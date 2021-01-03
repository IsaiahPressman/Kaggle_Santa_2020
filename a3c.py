import base64
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import shutil
from tensorboardX import SummaryWriter
import torch
from torch import distributions
import torch.nn.functional as F
import tqdm

# Custom imports
import vectorized_env as ve
import vectorized_agents as va


class A3CVectorized:
    def __init__(self, model_constructor, optimizer, model=None, device=torch.device('cuda'),
                 exp_folder=Path('runs/TEMP'),
                 recurrent_model=False, clip_grads=10.,
                 play_against_past_selves=True, n_past_selves=4, checkpoint_freq=10, initial_opponent_pool=[],
                 opp_posterior_decay=0.95):
        self.model_constructor = model_constructor
        self.optimizer = optimizer
        if model is None:
            self.model = self.model_constructor()
        else:
            self.model = model
        self.device = device
        self.exp_folder = exp_folder.absolute()
        if str(self.exp_folder) in ('/Windows/Users/isaia/Documents/GitHub/Kaggle/Santa_2020/runs/TEMP',
                                    '/home/pressmi/github_misc/Kaggle_Santa_2020/runs/TEMP'):
            print('WARNING: Using TEMP exp_folder')
            if self.exp_folder.exists():
                shutil.rmtree(self.exp_folder)
        elif self.exp_folder.exists() and any(Path(self.exp_folder).iterdir()):
            raise RuntimeError(f'Experiment folder {self.exp_folder} already exists and is not empty')
        else:
            print(f'Saving results to {self.exp_folder}')
        self.exp_folder.mkdir(exist_ok=True)
        self.recurrent_model = recurrent_model
        self.clip_grads = clip_grads
        self.play_against_past_selves = play_against_past_selves
        self.n_past_selves = n_past_selves
        self.checkpoint_freq = checkpoint_freq
        self.initial_opponent_pool = initial_opponent_pool
        self.opp_posterior_decay = opp_posterior_decay

        self.env = None
        self.opp_a = np.ones(len(self.initial_opponent_pool))
        self.opp_b = np.ones(len(self.initial_opponent_pool))
        self.checkpoints = []
        self.true_ep_num = 0
        self.summary_writer = None

        if self.clip_grads is not None:
            if self.clip_grads <= 0:
                raise ValueError(f'Should not clip gradients to <= 0, was {self.clip_grads} - '
                                 'pass None to clip_grads to not clip gradients')
            for p in self.model.parameters():
                if p.requires_grad:
                    p.register_hook(lambda grad: torch.clamp(grad, -self.clip_grads, self.clip_grads))

    def train(self, n_episodes, batch_size=30, gamma=0.99, **env_kwargs):
        if self.play_against_past_selves and 'opponent' in env_kwargs.keys():
            raise RuntimeError('Cannot play against past selves when opponent is defined')
        self.env = None
        self.model.train()

        for ep_num in tqdm.trange(n_episodes):
            if self.recurrent_model:
                self.model.reset_hidden_states()
            if self.play_against_past_selves:
                opponents, opponent_idxs = self.sample_opponents()
                self.env = ve.KaggleMABEnvTorchVectorized(opponent=opponents, **env_kwargs)
            else:
                self.env = ve.KaggleMABEnvTorchVectorized(**env_kwargs)
            buffer_a, buffer_r, buffer_l, buffer_v = [], [], [], []
            next_s, r, _, _ = self.env.reset()
            episode_reward_sums = r
            actor_losses = []
            critic_losses = []
            step_count = 1
            a, (l, v) = self.model.sample_action(next_s.to(device=self.device).unsqueeze(0), train=True)
            while not self.env.done:
                next_s, r, done, _ = self.env.step(a.squeeze(0))

                buffer_a.append(a)
                buffer_r.append(r)
                buffer_l.append(l)
                buffer_v.append(v)

                if self.recurrent_model and (step_count % batch_size == 0 or done):
                    self.model.detach_hidden_states()
                a, (l, v) = self.model.sample_action(next_s.to(device=self.device).unsqueeze(0), train=True)

                if step_count % batch_size == 0 or done:
                    if done:
                        v_next_s = torch.zeros_like(buffer_r[-1])
                    else:
                        # _, v_next_s = self.model(next_s.to(device=self.device).unsqueeze(0))
                        v_next_s = v.detach().squeeze(0)
                    v_next_s.to(device=self.device)

                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        v_next_s = r + gamma * v_next_s
                        buffer_v_target.append(v_next_s)
                    buffer_v_target.reverse()

                    actions = torch.cat(buffer_a).to(device=self.device)
                    v_t = torch.stack(buffer_v_target).to(device=self.device)
                    logits = torch.cat(buffer_l).to(device=self.device)
                    values = torch.cat(buffer_v).to(device=self.device)
                    # print(f'actions.shape: {actions.shape}, v_t.shape: {v_t.shape}')

                    # print(f'logits.shape: {logits.shape}, values.shape: {values.shape}')
                    td = v_t - values
                    # Huber loss
                    critic_loss = F.smooth_l1_loss(v_t, values, reduction='none').view(-1)

                    probs = F.softmax(logits, dim=-1)
                    real_batch_size, n_envs, n_players, n_bandits = probs.shape
                    m = distributions.Categorical(probs.view(real_batch_size * n_envs * n_players, n_bandits))
                    # print(f'm.log_prob(actions.view(real_batch_size * n_envs * n_players)).shape: '
                    #       f'{m.log_prob(actions.view(real_batch_size * n_envs * n_players)).shape}, '
                    #       f'td.shape: {td.shape}')
                    actor_loss = -(m.log_prob(actions.view(-1)) * td.detach().view(-1))
                    total_loss = (critic_loss + actor_loss).mean()

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
                    buffer_a, buffer_r, buffer_l, buffer_v = [], [], [], []
                    actor_losses.append(actor_loss.detach().mean().clone().cpu())
                    critic_losses.append(critic_loss.detach().mean().clone().cpu())
                episode_reward_sums += r
                step_count += 1
            if self.play_against_past_selves:
                if opponent_idxs is not None:
                    self.update_opponent_scores(opponent_idxs, self.env.player_rewards_sums.cpu())
                if self.true_ep_num % self.checkpoint_freq == 0:
                    self.checkpoint()
            if self.true_ep_num % self.checkpoint_freq == 0:
                self.save()
            self.log(
                episode_reward_sums.mean().clone().cpu() / self.env.r_norm,
                torch.stack(actor_losses),
                torch.stack(critic_losses)
            )
            self.true_ep_num += 1
        self.save(finished=True)

    def sample_opponents(self):
        if len(self.initial_opponent_pool) + len(self.checkpoints) == 0:
            print('No opponents sampled')
            return None, None
        else:
            opponents = []
            opponent_idxs = []
            for i in range(self.n_past_selves):
                # Sample opponents randomly if the agent is losing > some fraction of it's games, otherwise use thompson sampling
                if np.sum(self.opp_b - 1) <= np.sum(self.opp_a - 1) / 5.:
                    if i == 0:
                        print('Sampling opponents randomly')
                    selected_opp_idx = np.random.randint(len(self.opp_a))
                else:
                    if i == 0:
                        print('Sampling opponents using Thompson sampling')
                    selected_opp_idx = np.random.beta(self.opp_a, self.opp_b).argmax()
                opponent_idxs.append(selected_opp_idx)
                opponents.append(self.get_opponent(selected_opp_idx))
                if selected_opp_idx < len(self.initial_opponent_pool):
                    try:
                        print(
                            f'Opponent {i}: {opponents[-1].name} with alpha {self.opp_a[selected_opp_idx]:.2f} '
                            f'and beta {self.opp_b[selected_opp_idx]:.2f}')
                    except AttributeError:
                        print(
                            f'Opponent {i}: {opponents[-1]} with alpha {self.opp_a[selected_opp_idx]:.2f} '
                            f'and beta {self.opp_b[selected_opp_idx]:.2f}')
                else:
                    print(
                        f'Opponent {i}: Checkpoint #{selected_opp_idx - len(self.initial_opponent_pool)} with '
                        f'alpha {self.opp_a[selected_opp_idx]:.2f} and beta {self.opp_b[selected_opp_idx]:.2f}')
            print()
            return va.MultiAgent(opponents), opponent_idxs

    def update_opponent_scores(self, curr_opp_idxs, player_rewards_sums):
        # player_rewards_sums.shape is (n_envs, n_players, n_bandits)
        # Decay posteriors
        self.opp_a = np.maximum(self.opp_a * self.opp_posterior_decay, 1.)
        self.opp_b = np.maximum(self.opp_b * self.opp_posterior_decay, 1.)

        # Update posteriors with new w/l stats
        rewards_sums = player_rewards_sums.sum(dim=2)
        game_scores = torch.zeros(rewards_sums.shape)
        winners_idxs = rewards_sums.argmax(dim=1)
        draws_mask = rewards_sums[:, 0] == rewards_sums[:, 1]
        game_scores[torch.arange(game_scores.shape[0]), winners_idxs] = 1.
        game_scores[draws_mask] = 0.5
        game_scores = [gs.numpy() for gs in game_scores.chunk(self.n_past_selves)]
        for match_idx, opp_idx in enumerate(curr_opp_idxs):
            self.opp_a[opp_idx] += game_scores[match_idx][:, 1].sum()
            self.opp_b[opp_idx] += game_scores[match_idx][:, 0].sum()

    def checkpoint(self):
        self.model.cpu()
        self.checkpoints.append(self.model.state_dict())
        self.opp_a = np.append(self.opp_a, 1.)
        self.opp_b = np.append(self.opp_b, 1.)
        self.model.to(device=self.device)

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
        # Save skill estimates
        if self.play_against_past_selves:
            checkpoint_idxs = np.arange(len(self.opp_a)) - len(self.initial_opponent_pool)
            checkpoint_idxs = np.where(
                checkpoint_idxs < 0,
                np.nan,
                checkpoint_idxs
            )
            df = pd.DataFrame({
                'opp_a': self.opp_a,
                'opp_b': self.opp_b,
                'est_opp_skill': self.opp_a / (self.opp_a + self.opp_b),
                'checkpoint_idxs': checkpoint_idxs
            })
            df.to_csv(f'{file_path_base}_skill_estimates.csv')

    def log(self, episode_reward_sums, actor_losses, critic_losses):
        # Lazily initialize summary_writer
        if self.summary_writer is None:
            self.summary_writer = SummaryWriter(self.exp_folder)
        self.summary_writer.add_scalar('episode/reward', episode_reward_sums.numpy().item(), self.true_ep_num)
        self.summary_writer.add_scalar('episode/actor_loss', actor_losses.mean().numpy().item(), self.true_ep_num)
        self.summary_writer.add_scalar('episode/critic_loss', critic_losses.mean().numpy().item(), self.true_ep_num)
        assert len(actor_losses) == len(critic_losses)
        for i, (a_l, c_l) in enumerate(zip(actor_losses.numpy(), critic_losses.numpy())):
            batch_num = self.true_ep_num * len(actor_losses) + i
            self.summary_writer.add_scalar('batch/actor_loss', a_l.item(), batch_num)
            self.summary_writer.add_scalar('batch/critic_loss', c_l.item(), batch_num)
        if self.true_ep_num % self.checkpoint_freq == 0:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.summary_writer.add_histogram(f'params/{name}',
                                                      param.clone().cpu().data.numpy(),
                                                      self.true_ep_num)

    def get_opponent(self, idx):
        if idx < 0:
            raise IndexError(f'Negative indexing is not supported')
        elif idx < len(self.initial_opponent_pool):
            return self.initial_opponent_pool[idx]
        elif idx < len(self.initial_opponent_pool) + len(self.checkpoints):
            checkpoint_idx = idx - len(self.initial_opponent_pool)
            checkpoint_opp = self.model_constructor()
            checkpoint_opp.load_state_dict(self.checkpoints[checkpoint_idx])
            checkpoint_opp.to(device=self.device)
            checkpoint_opp.eval()
            return va.RLModelWrapperAgent(checkpoint_opp, self.env.obs_type)
        else:
            raise IndexError(f'Index {idx} is out of bounds')
