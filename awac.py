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


class ReplayBuffer:
    def __init__(self, s_shape, max_len=1e6, starting_s_a_r_d_s=None):
        self.max_len = int(max_len)
        self._s_buffer = torch.zeros(self.max_len, *s_shape)
        self._a_buffer = torch.zeros(self.max_len)
        self._r_buffer = torch.zeros(self.max_len)
        self._d_buffer = torch.zeros(self.max_len)
        self._next_s_buffer = torch.zeros(self.max_len, *s_shape)
        self.current_size = 0
        self._top = 0
        if starting_s_a_r_d_s is not None:
            self.append_samples_batch(*starting_s_a_r_d_s)

    def get_samples_batch(self, sample_size):
        # Sampling with replacement
        idxs = np.random.randint(self.current_size, size=(sample_size,))
        # Sampling without replacement is possible, but ~2.5x slower:
        # idxs = np.random.choice(self.current_size, size=sample_size, replace=(self.current_size < sample_size))
        return (self._s_buffer[idxs],
                self._a_buffer[idxs],
                self._r_buffer[idxs],
                self._d_buffer[idxs],
                self._next_s_buffer[idxs])

    def append_samples_batch(self, s_batch, a_batch, r_batch, d_batch, next_s_batch):
        batch_len = s_batch.shape[0]
        assert a_batch.shape[0] == batch_len
        assert r_batch.shape[0] == batch_len
        assert next_s_batch.shape[0] == batch_len
        assert d_batch.shape[0] == batch_len
        new_len = self._top + batch_len
        if new_len <= self.max_len:
            self._s_buffer[self._top:new_len] = s_batch
            self._a_buffer[self._top:new_len] = a_batch
            self._r_buffer[self._top:new_len] = r_batch
            self._d_buffer[self._top:new_len] = d_batch
            self._next_s_buffer[self._top:new_len] = next_s_batch
            self._top = new_len % self.max_len
            self.current_size = max(new_len, self.current_size)
        else:
            leftover_batch = new_len % self.max_len
            s_batch_split = s_batch.split((batch_len - leftover_batch, leftover_batch))
            a_batch_split = a_batch.split((batch_len - leftover_batch, leftover_batch))
            r_batch_split = r_batch.split((batch_len - leftover_batch, leftover_batch))
            d_batch_split = d_batch.split((batch_len - leftover_batch, leftover_batch))
            next_s_batch_split = next_s_batch.split((batch_len - leftover_batch, leftover_batch))
            self.append_samples_batch(s_batch_split[0],
                                      a_batch_split[0],
                                      r_batch_split[0],
                                      d_batch_split[0],
                                      next_s_batch_split[0])
            self.append_samples_batch(s_batch_split[1],
                                      a_batch_split[1],
                                      r_batch_split[1],
                                      d_batch_split[1],
                                      next_s_batch_split[1])


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
                torch.zeros(r.shape).view(-1) if done else torch.ones(r.shape).view(-1),
                next_s.view(-1, *next_s.shape[-2:]).cpu().clone()
            )
            s = next_s
            episode_reward_sums += r
            for i in range(train_batches_per_timestep):
                self.train_batch(batch_size, gamma, lagrange_multiplier)
            self.summary_writer.add_scalar('DEBUG/batch_time_ms', time.time() - start_time, self.true_batch_num)
            self.summary_writer.add_scalar('DEBUG/replay_buffer_len', self.replay_buffer.current_size,
                                           self.true_batch_num)
            self.summary_writer.add_scalar('DEBUG/replay_buffer__top', self.replay_buffer._top,
                                           self.true_batch_num)

        self.save(finished=True)

    def train_batch(self, batch_size, gamma, lagrange_multiplier):
        s_batch, a_batch, r_batch, d_batch, next_s_batch = self.replay_buffer.get_samples_batch(batch_size)
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
]