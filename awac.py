import base64
from collections import deque
import numpy as np
from pathlib import Path
import pickle
import shutil
from tensorboardX import SummaryWriter
import time
import torch
from torch import distributions
import torch.nn.functional as F
import tqdm

# Custom imports
import vectorized_env as ve


class DEPRECATED_ReplayBuffer:
    def __init__(self, max_len=1e6, starting_s_a_r_s_d=None):
        self.max_len = int(max_len)
        # self.sample_shapes = sample_shapes
        # if len(sample_shapes) != 4:
        #     raise ValueError(f"sample_shapes must have length 4, for s, a, r, s', was {len(sample_shapes)}")
        self.s_buffer = deque(maxlen=self.max_len)
        self.a_buffer = deque(maxlen=self.max_len)
        self.r_buffer = deque(maxlen=self.max_len)
        self.next_s_buffer = deque(maxlen=self.max_len)
        self.d_buffer = deque(maxlen=self.max_len)
        if starting_s_a_r_s_d is not None:
            self.append_samples_batch(*starting_s_a_r_s_d)

    def get_samples_batch(self, sample_size):
        if sample_size > len(self.s_buffer):
            return (torch.cat(list(self.s_buffer)),
                    torch.cat(list(self.a_buffer)),
                    torch.cat(list(self.r_buffer)),
                    torch.cat(list(self.next_s_buffer)),
                    torch.cat(list(self.d_buffer)))
        else:
            # Too slow:
            # idxs = torch.multinomial(torch.ones(len(self.s_buffer)), sample_size)
            # This is better, but is sampling with replacement
            idxs = np.random.randint(len(self.s_buffer), size=(sample_size,))
            return (torch.cat([self.s_buffer[i] for i in idxs]),
                    torch.cat([self.a_buffer[i] for i in idxs]),
                    torch.cat([self.r_buffer[i] for i in idxs]),
                    torch.cat([self.next_s_buffer[i] for i in idxs]),
                    torch.cat([self.d_buffer[i] for i in idxs]))

    def append_samples_batch(self, s_batch, a_batch, r_batch, next_s_batch, d_batch):
        batch_len = s_batch.shape[0]
        assert a_batch.shape[0] == batch_len
        assert r_batch.shape[0] == batch_len
        assert next_s_batch.shape[0] == batch_len
        assert d_batch.shape[0] == batch_len
        self.s_buffer.extend(s_batch.split(1))
        self.a_buffer.extend(a_batch.split(1))
        self.r_buffer.extend(r_batch.split(1))
        self.next_s_buffer.extend(next_s_batch.split(1))
        self.d_buffer.extend(d_batch.split(1))
        """
        if len(self.s_buffer) > self.max_len:
            self.s_buffer = self.s_buffer[-self.max_len:]
            self.a_buffer = self.a_buffer[-self.max_len:]
            self.r_buffer = self.r_buffer[-self.max_len:]
            self.next_s_buffer = self.next_s_buffer[-self.max_len:]
            self.d_buffer = self.d_buffer[-self.max_len:]"""


class AWACVectorized:
    def __init__(self, model, optimizer, replay_buffer,
                 device=torch.device('cuda'), exp_folder=Path('runs/awac/TEMP'),
                 clip_grads=10., checkpoint_freq=10):
        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.device = device
        self.exp_folder = exp_folder.absolute()
        if str(self.exp_folder) in ('/Windows/Users/isaia/Documents/GitHub/Kaggle/Santa_2020/runs/awac/TEMP',
                                    '/home/pressmi/github_misc/Kaggle_Santa_2020/runs/awac/TEMP'):
            print('WARNING: Using TEMP exp_folder')
            if self.exp_folder.exists():
                shutil.rmtree(self.exp_folder)
        elif self.exp_folder.exists() and any(Path(self.exp_folder).iterdir()):
            raise RuntimeError(f'Experiment folder {self.exp_folder} already exists and is not empty')
        else:
            print(f'Saving results to {self.exp_folder}')
        self.exp_folder.mkdir(exist_ok=True)
        self.clip_grads = clip_grads
        self.checkpoint_freq = checkpoint_freq

        self.env = None
        self.true_ep_num = 0
        self.true_batch_num = 0
        self.summary_writer = SummaryWriter(self.exp_folder)

        if self.clip_grads is not None:
            if self.clip_grads <= 0:
                raise ValueError(f'Should not clip gradients to <= 0, was {self.clip_grads} - '
                                 'pass None to clip_grads to not clip gradients')
            for p in self.model.parameters():
                if p.requires_grad:
                    p.register_hook(lambda grad: torch.clamp(grad, -self.clip_grads, self.clip_grads))

    def train(self, batch_size, n_pretrain_batches, n_steps, train_batches_per_timestep,
              gamma=0.99, lagrange_multiplier=1., **env_kwargs):
        self.env = ve.KaggleMABEnvTorchVectorized(**env_kwargs)
        self.model.train()

        print('Pre-training:')
        for _ in tqdm.trange(n_pretrain_batches):
            start_time = time.time()
            self.train_batch(batch_size, gamma, lagrange_multiplier)
            self.summary_writer.add_scalar('DEBUG/batch_time_ms', time.time() - start_time, self.true_batch_num)

        print('Main training loop:')
        s, r, done, info_dict = self.env.reset()
        episode_reward_sums = r
        for _ in tqdm.trange(n_steps):
            start_time = time.time()
            if done:
                if self.true_ep_num % self.checkpoint_freq == 0:
                    self.save()
                self.log_episode(
                    episode_reward_sums.mean().cpu().clone() / self.env.r_norm,
                    info_dict
                )
                self.true_ep_num += 1
                s, r, done, info_dict = self.env.reset()
                episode_reward_sums = r

            a = self.model.sample_action(s.to(device=self.device).unsqueeze(0))
            next_s, r, done, info_dict = self.env.step(a.squeeze(0))
            self.replay_buffer.append_samples_batch(
                s.view(-1, *s.shape[-2:]).cpu().clone(),
                a.view(-1).detach().cpu().clone(),
                r.view(-1).cpu().clone(),
                next_s.view(-1, *next_s.shape[-2:]).cpu().clone(),
                torch.zeros(r.shape).view(-1) if done else torch.ones(r.shape).view(-1)
            )
            s = next_s
            episode_reward_sums += r
            for i in range(train_batches_per_timestep):
                self.train_batch(batch_size, gamma, lagrange_multiplier)
            self.summary_writer.add_scalar('DEBUG/batch_time_ms', time.time() - start_time, self.true_batch_num)
            self.summary_writer.add_scalar('DEBUG/replay_buffer_len', len(self.replay_buffer.s_buffer),
                                           self.true_batch_num)

        self.save(finished=True)

    def train_batch(self, batch_size, gamma, lagrange_multiplier):
        s_batch, a_batch, r_batch, next_s_batch, d_batch = self.replay_buffer.get_samples_batch(batch_size)
        s_batch = s_batch.to(device=self.device)
        a_batch = a_batch.to(device=self.device)
        r_batch = r_batch.to(device=self.device)
        d_batch = d_batch.to(device=self.device)
        next_s_batch = next_s_batch.to(device=self.device)
        logits, values = self.model(s_batch.to(device=self.device))
        _, v_next_s = self.model(next_s_batch.to(device=self.device))
        # Reward of 0 for terminal states
        v_t = (r_batch + gamma * v_next_s * (1 - d_batch)).detach()
        # Huber loss for critic
        critic_loss = F.smooth_l1_loss(values, v_t, reduction='none').view(-1)

        td = v_t - values
        log_probs = distributions.Categorical(F.softmax(logits, dim=-1)).log_prob(a_batch.view(-1))
        actor_loss = -(log_probs * torch.exp(td.view(-1) / lagrange_multiplier))

        total_loss = (critic_loss + actor_loss).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.log_batch(actor_loss.mean(), critic_loss.mean(), total_loss)
        self.true_batch_num += 1

    def log_batch(self, actor_loss, critic_loss, total_loss):
        self.summary_writer.add_scalar('batch/actor_loss', actor_loss.detach().cpu().numpy().item(),
                                       self.true_batch_num)
        self.summary_writer.add_scalar('batch/critic_loss', critic_loss.detach().cpu().numpy().item(),
                                       self.true_batch_num)
        self.summary_writer.add_scalar('batch/total_loss', total_loss.detach().cpu().numpy().item(),
                                       self.true_batch_num)

    def log_episode(self, episode_reward_sums, final_info_dict):
        self.summary_writer.add_scalar('episode/reward', episode_reward_sums.numpy().item(), self.true_ep_num)
        if self.env.opponent is None:
            self.summary_writer.add_scalar(
                'episode/pull_rewards_sum',
                final_info_dict['player_rewards_sums'].cpu().sum(dim=-1).mean().numpy().item(),
                self.true_ep_num
            )
        else:
            self.summary_writer.add_scalar(
                'episode/pull_rewards_sum',
                final_info_dict['player_rewards_sums'].cpu().sum(dim=-1)[:,0].mean().numpy().item(),
                self.true_ep_num
            )
        if self.true_ep_num % self.checkpoint_freq == 0:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.summary_writer.add_histogram(
                        f'params/{name}',
                        param.detach().cpu().clone().numpy(),
                        self.true_ep_num
                    )

    def save(self, finished=False):
        if finished:
            file_path_base = self.exp_folder / f'final_{self.true_ep_num}'
        else:
            file_path_base = self.exp_folder / str(self.true_ep_num)
        # Save model params
        self.model.cpu()
        state_dict_bytes = pickle.dumps({
            'model_state_dict': self.model.state_dict(),
        })
        serialized_string = base64.b64encode(state_dict_bytes)
        with open(f'{file_path_base}_cp.txt', 'w') as f:
            f.write(str(serialized_string))
        self.model.to(device=self.device)


class ReplayBuffer:
    def __init__(self, max_len=1e6, starting_s_a_r_s_d=None):
        self.max_len = int(max_len)
        # self.sample_shapes = sample_shapes
        # if len(sample_shapes) != 4:
        #     raise ValueError(f"sample_shapes must have length 4, for s, a, r, s', was {len(sample_shapes)}")
        self.s_buffer = []
        self.a_buffer = []
        self.r_buffer = []
        self.next_s_buffer = []
        self.d_buffer = []
        if starting_s_a_r_s_d is not None:
            self.append_samples_batch(*starting_s_a_r_s_d)

    def get_samples_batch(self, sample_size):
        if sample_size > len(self.s_buffer):
            return (torch.cat(self.s_buffer),
                    torch.cat(self.a_buffer),
                    torch.cat(self.r_buffer),
                    torch.cat(self.next_s_buffer),
                    torch.cat(self.d_buffer))
        else:
            # Too slow:
            # idxs = torch.multinomial(torch.ones(len(self.s_buffer)), sample_size)
            # This is better, but is sampling with replacement
            idxs = np.random.randint(len(self.s_buffer), size=(sample_size,))
            return (torch.cat([self.s_buffer[i] for i in idxs]),
                    torch.cat([self.a_buffer[i] for i in idxs]),
                    torch.cat([self.r_buffer[i] for i in idxs]),
                    torch.cat([self.next_s_buffer[i] for i in idxs]),
                    torch.cat([self.d_buffer[i] for i in idxs]))

    def append_samples_batch(self, s_batch, a_batch, r_batch, next_s_batch, d_batch):
        batch_len = s_batch.shape[0]
        assert a_batch.shape[0] == batch_len
        assert r_batch.shape[0] == batch_len
        assert next_s_batch.shape[0] == batch_len
        assert d_batch.shape[0] == batch_len
        self.s_buffer.extend(s_batch.split(1))
        self.a_buffer.extend(a_batch.split(1))
        self.r_buffer.extend(r_batch.split(1))
        self.next_s_buffer.extend(next_s_batch.split(1))
        self.d_buffer.extend(d_batch.split(1))
        if len(self.s_buffer) > self.max_len:
            self.s_buffer = self.s_buffer[-self.max_len:]
            self.a_buffer = self.a_buffer[-self.max_len:]
            self.r_buffer = self.r_buffer[-self.max_len:]
            self.next_s_buffer = self.next_s_buffer[-self.max_len:]
            self.d_buffer = self.d_buffer[-self.max_len:]