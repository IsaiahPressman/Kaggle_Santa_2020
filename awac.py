import base64
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


class AWACVectorized:
    def __init__(self, model, optimizer, replay_buffer,
                 validation_env_kwargs_dicts=(), deterministic_validation_policy=True,
                 device=torch.device('cuda'), exp_folder=Path('runs/awac/TEMP'),
                 clip_grads=10., checkpoint_freq=10):
        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
        self.validation_env_kwargs_dicts = validation_env_kwargs_dicts
        opponent_names = []
        for i, d in enumerate(self.validation_env_kwargs_dicts):
            if d.get('reward_type', ve.END_OF_GAME_TRUE) != ve.END_OF_GAME_TRUE:
                raise ValueError(f'Validation envs should have reward_type: {ve.END_OF_GAME_TRUE}, was '
                                 f'{d["reward_type"]} for env {i}')
            d['reward_type'] = ve.END_OF_GAME_TRUE
            opp = d.get('opponent')
            if opp is not None:
                opponent_names.append(opp.name)
            else:
                opponent_names.append('None')
        if len(opponent_names) != len(np.unique(opponent_names)):
            raise ValueError(f'Duplicate opponents encountered in validation_env_kwargs_dicts : {opponent_names}')
        self.deterministic_validation_policy = deterministic_validation_policy
        self.device = device
        self.exp_folder = exp_folder.absolute()
        if self.exp_folder.name == 'TEMP':
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
        self.epoch_counter = 0
        self.episode_counter = 0
        self.batch_counter = 0
        self.train_step_counter = 0
        self.validation_counter = 0
        self.validation_step_counters = [0] * len(validation_env_kwargs_dicts)
        self.summary_writer = SummaryWriter(self.exp_folder)

        if self.clip_grads is not None:
            if self.clip_grads <= 0:
                raise ValueError(f'Should not clip gradients to <= 0, was {self.clip_grads} - '
                                 'pass None to clip_grads to not clip gradients')
            for p in self.model.parameters():
                if p.requires_grad:
                    p.register_hook(lambda grad: torch.clamp(grad, -self.clip_grads, self.clip_grads))

    def train(self, env, batch_size, n_pretrain_batches, n_epochs, n_steps_per_epoch,
              n_train_batches_per_epoch=None, gamma=0.99, lagrange_multiplier=1.):
        self.env = env
        if n_train_batches_per_epoch is None:
            n_train_batches_per_epoch = n_steps_per_epoch

        self.model.train()
        print(f'Pre-training on {n_pretrain_batches} batches')
        for batch in tqdm.trange(n_pretrain_batches):
            self.train_on_batch(batch_size, gamma, lagrange_multiplier)

        self.run_validation()
        print(f'\nRunning main training loop with {n_epochs} epochs')
        for epoch in range(n_epochs):
            epoch_start_time = time.time()
            self.model.eval()
            print(f'Epoch #{epoch}:')
            print(f'Sampling {n_steps_per_epoch} time-steps from the environment')
            s, r, done, info_dict = self.env.reset()
            episode_reward_sums = r
            for step in tqdm.trange(n_steps_per_epoch):
                start_time = time.time()
                a = self.model.sample_action(s.to(device=self.device).unsqueeze(0))
                next_s, r, done, info_dict = self.env.step(a.squeeze(0))
                self.replay_buffer.append_samples_batch(
                    s.view(-1, *s.shape[-2:]).cpu().clone(),
                    a.view(-1).detach().cpu().clone(),
                    r.view(-1).cpu().clone(),
                    torch.ones(r.shape).view(-1) if done else torch.zeros(r.shape).view(-1),
                    next_s.view(-1, *next_s.shape[-2:]).cpu().clone()
                )
                self.summary_writer.add_scalar(f'Info/replay_buffer_size',
                                               len(self.replay_buffer),
                                               self.train_step_counter)
                self.summary_writer.add_scalar(f'Info/replay_buffer__top',
                                               self.replay_buffer._top,
                                               self.train_step_counter)
                s = next_s
                episode_reward_sums += r
                if done:
                    self.log_train_episode(
                        episode_reward_sums.mean().cpu().clone() / self.env.r_norm,
                        info_dict
                    )
                    self.episode_counter += 1
                    s, r, done, info_dict = self.env.reset()
                    episode_reward_sums = r
                self.summary_writer.add_scalar('Time/exploration_step_time_ms',
                                               (time.time() - start_time) * 1000,
                                               self.train_step_counter)
                self.train_step_counter += 1

            self.model.train()
            print(f'Training on {n_train_batches_per_epoch} batches from the replay buffer')
            for batch in tqdm.trange(n_train_batches_per_epoch):
                self.train_on_batch(batch_size, gamma, lagrange_multiplier)

            self.log_epoch()
            self.epoch_counter += 1
            if self.epoch_counter % self.checkpoint_freq == 0:
                self.run_validation()
                self.save()
            self.summary_writer.add_scalar('Time/epoch_time_minutes',
                                           (time.time() - epoch_start_time) / 60.,
                                           self.epoch_counter - 1)
            print()
        self.save(finished=True)

    def train_on_batch(self, batch_size, gamma, lagrange_multiplier):
        start_time = time.time()
        s_batch, a_batch, r_batch, d_batch, next_s_batch = self.replay_buffer.get_samples_batch(batch_size)
        s_batch = s_batch.to(device=self.device)
        a_batch = a_batch.to(device=self.device)
        r_batch = r_batch.to(device=self.device)
        d_batch = d_batch.to(device=self.device)
        next_s_batch = next_s_batch.to(device=self.device)
        logits, values = self.model(s_batch.to(device=self.device))
        _, v_next_s = self.model(next_s_batch.to(device=self.device))
        # Reward of 0 for terminal states
        v_t = (r_batch + gamma * v_next_s * (1. - d_batch)).detach()
        # Huber loss for critic
        critic_loss = F.smooth_l1_loss(values, v_t, reduction='none').view(-1)

        td = v_t - values
        log_probs = distributions.Categorical(F.softmax(logits, dim=-1)).log_prob(a_batch.view(-1))
        weights = torch.exp(td / lagrange_multiplier).view(-1)
        # weights = F.softmax(td.view(-1) / lagrange_multiplier, dim=-1)
        actor_loss = -(log_probs * weights.detach())

        total_loss = (critic_loss + actor_loss).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.log_batch(actor_loss.mean(), critic_loss.mean(), total_loss)
        self.summary_writer.add_scalar('Time/batch_time_ms',
                                       (time.time() - start_time) * 1000,
                                       self.batch_counter)
        self.batch_counter += 1

    def log_batch(self, actor_loss, critic_loss, total_loss):
        self.summary_writer.add_scalar('Batch/actor_loss', actor_loss.detach().cpu().numpy().item(),
                                       self.batch_counter)
        self.summary_writer.add_scalar('Batch/critic_loss', critic_loss.detach().cpu().numpy().item(),
                                       self.batch_counter)
        self.summary_writer.add_scalar('Batch/total_loss', total_loss.detach().cpu().numpy().item(),
                                       self.batch_counter)

    def log_train_episode(self, episode_reward_sums, final_info_dict):
        self.summary_writer.add_scalar(f'Episode/reward_{self.env.reward_type}',
                                       episode_reward_sums.numpy().item(),
                                       self.episode_counter)
        if self.env.opponent is None:
            pull_rewards_sum = final_info_dict['player_rewards_sums'].sum(dim=-1).cpu()
        else:
            pull_rewards_sum = final_info_dict['player_rewards_sums'].sum(dim=-1)[:, 0].cpu()
        self.summary_writer.add_histogram(
            'Episode/agent_pull_rewards',
            pull_rewards_sum,
            self.episode_counter
        )
        self.summary_writer.add_histogram(
            'Episode/p1_pull_rewards',
            final_info_dict['player_rewards_sums'].sum(dim=-1)[:, 0].cpu().numpy(),
            self.episode_counter
        )
        self.summary_writer.add_histogram(
            'Episode/p2_pull_rewards',
            final_info_dict['player_rewards_sums'].sum(dim=-1)[:, 1].cpu().numpy(),
            self.episode_counter
        )
        self.summary_writer.add_scalar(
            'Episode/mean_agent_pull_rewards',
            pull_rewards_sum.mean().numpy().item(),
            self.episode_counter
        )
        self.summary_writer.add_scalar(
            'Episode/mean_p1_pull_rewards',
            final_info_dict['player_rewards_sums'].sum(dim=-1)[:, 0].mean().cpu().numpy().item(),
            self.episode_counter
        )
        self.summary_writer.add_scalar(
            'Episode/mean_p2_pull_rewards',
            final_info_dict['player_rewards_sums'].sum(dim=-1)[:, 1].mean().cpu().numpy().item(),
            self.episode_counter
        )

    def log_epoch(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.summary_writer.add_histogram(
                    f'params/{name}',
                    param.detach().cpu().clone().numpy(),
                    self.epoch_counter
                )

    def log_validation_episodes(self, episode_reward_sums, final_info_dicts):
        assert len(episode_reward_sums) == len(final_info_dicts)
        n_val_envs = len(episode_reward_sums)
        for i, ers, fid in zip(range(n_val_envs), episode_reward_sums, final_info_dicts):
            opponent = self.validation_env_kwargs_dicts[i].get('opponent')
            if opponent is not None:
                opponent = opponent.name
            env_name = opponent

            self.summary_writer.add_histogram(
                f'Validation/{env_name}_game_results',
                ers.numpy(),
                self.validation_counter
            )
            self.summary_writer.add_histogram(
                f'Validation/{env_name}_hero_pull_rewards',
                fid['player_rewards_sums'].sum(dim=-1)[:, 0].cpu().numpy(),
                self.validation_counter
            )
            self.summary_writer.add_histogram(
                f'Validation/{env_name}_villain_pull_rewards',
                fid['player_rewards_sums'].sum(dim=-1)[:, 1].cpu().numpy(),
                self.validation_counter
            )
            self.summary_writer.add_scalar(
                f'Validation/{env_name}_win_percent',
                # W/D/L values are 1/0/-1, so they need to be scaled to 1/0.5/0 to be represented as a percent
                (ers.mean().numpy().item() + 1) / 2. * 100.,
                self.validation_counter
            )
            self.summary_writer.add_scalar(
                f'Validation/{env_name}_mean_hero_pull_rewards',
                fid['player_rewards_sums'].sum(dim=-1)[:, 0].mean().cpu().numpy().item(),
                self.validation_counter
            )
            self.summary_writer.add_scalar(
                f'Validation/{env_name}_mean_villain_pull_rewards',
                fid['player_rewards_sums'].sum(dim=-1)[:, 1].mean().cpu().numpy().item(),
                self.validation_counter
            )

    def run_validation(self):
        self.model.eval()
        if len(self.validation_env_kwargs_dicts) > 0:
            print(f'Validating model performance in {len(self.validation_env_kwargs_dicts)} environments')
            episode_reward_sums = []
            final_info_dicts = []
            for i in tqdm.trange(len(self.validation_env_kwargs_dicts)):
                if 'opponent' in self.validation_env_kwargs_dicts[i].keys():
                    self.validation_env_kwargs_dicts[i]['opponent'].reset()
                # Lazily construct validation envs to conserve GPU memory
                val_env = ve.KaggleMABEnvTorchVectorized(**self.validation_env_kwargs_dicts[i])
                s, r, done, info_dict = val_env.reset()
                episode_reward_sums.append(r)
                while not done:
                    start_time = time.time()
                    if self.deterministic_validation_policy:
                        a = self.model.choose_best_action(s.to(device=self.device).unsqueeze(0))
                    else:
                        a = self.model.sample_action(s.to(device=self.device).unsqueeze(0))
                    next_s, r, done, info_dict = val_env.step(a.squeeze(0))
                    s = next_s
                    episode_reward_sums[-1] += r
                    self.summary_writer.add_scalar(f'Time/val_env{i}_step_time_ms',
                                                   (time.time() - start_time) * 1000,
                                                   self.validation_step_counters[i])
                    self.validation_step_counters[i] += 1
                episode_reward_sums[-1] = episode_reward_sums[-1].mean(dim=-1).cpu().clone() / val_env.r_norm
                final_info_dicts.append(info_dict)
            self.log_validation_episodes(
                episode_reward_sums,
                final_info_dicts
            )
        self.validation_counter += 1

    def save(self, finished=False):
        if finished:
            file_path_base = self.exp_folder / f'final_{self.epoch_counter}'
        else:
            file_path_base = self.exp_folder / str(self.epoch_counter)
        # Save model params
        self.model.cpu()
        state_dict_bytes = pickle.dumps({
            'model_state_dict': self.model.state_dict(),
        })
        serialized_string = base64.b64encode(state_dict_bytes)
        with open(f'{file_path_base}_cp.txt', 'w') as f:
            f.write(str(serialized_string))
        self.model.to(device=self.device)


class BasicReplayBuffer:
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
            # Randomly shuffle initial experiences
            shuffled_idxs = np.arange(self.current_size)
            np.random.shuffle(shuffled_idxs)
            shuffled_idxs = np.append(shuffled_idxs, np.arange(self.current_size, self.max_len))
            self._s_buffer = self._s_buffer[torch.from_numpy(shuffled_idxs)]
            self._a_buffer = self._a_buffer[torch.from_numpy(shuffled_idxs)]
            self._r_buffer = self._r_buffer[torch.from_numpy(shuffled_idxs)]
            self._d_buffer = self._d_buffer[torch.from_numpy(shuffled_idxs)]
            self._next_s_buffer = self._next_s_buffer[torch.from_numpy(shuffled_idxs)]

    def get_samples_batch(self, sample_size):
        # Sampling with replacement
        idxs = torch.randint(self.current_size, size=(sample_size,))
        # Sampling without replacement is possible, but quite a bit slower:
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
        assert d_batch.shape[0] == batch_len
        assert next_s_batch.shape[0] == batch_len
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

    def __len__(self):
        return self.current_size


class EveryStepObsReplayBuffer:
    def __init__(self, s_shape, max_len=1e6, is_ravelled=True,
                 env_n_steps=1999, starting_s_a_r_d_s=None):
        self.max_len = int(max_len - (max_len % env_n_steps))
        self.is_ravelled = is_ravelled
        self.env_n_steps = env_n_steps
        self._max_len_full_obs = self.max_len // self.env_n_steps
        self._full_observations = [torch.zeros(*s_shape, dtype=torch.float) for _ in range(self._max_len_full_obs)]
        self._a_buffer = torch.zeros(self.max_len)
        self._r_buffer = torch.zeros(self.max_len)
        self._d_buffer = torch.zeros(self.max_len)
        self._a_batch_cache = []
        self._r_batch_cache = []
        self._d_batch_cache = []
        self._full_observations_top = 0
        if starting_s_a_r_d_s is not None:
            raise ValueError('starting_s_a_r_d_s is not yet implemented')

    def get_samples_batch(self, sample_size):
        # Sampling with replacement
        idxs = np.random.randint(len(self), size=(sample_size,))

        return (self._get_s(idxs),
                self._a_buffer[idxs],
                self._r_buffer[idxs],
                self._d_buffer[idxs],
                self._get_next_s(idxs))

    def append_samples_batch(self, s_batch, a_batch, r_batch, d_batch, next_s_batch):
        self._a_batch_cache.append(a_batch)
        self._r_batch_cache.append(r_batch)
        self._d_batch_cache.append(d_batch)
        if d_batch.to(dtype=torch.bool).all():
            batch_len = s_batch.shape[0]
            assert next_s_batch.shape[0] == batch_len
            a_batch = torch.stack(self._a_batch_cache, dim=-1)
            r_batch = torch.stack(self._r_batch_cache, dim=-1)
            d_batch = torch.stack(self._d_batch_cache, dim=-1)
            assert a_batch.shape == (batch_len, self.env_n_steps)
            assert r_batch.shape == (batch_len, self.env_n_steps)
            assert d_batch.shape == (batch_len, self.env_n_steps)
            self._a_batch_cache = []
            self._r_batch_cache = []
            self._d_batch_cache = []
            a_batch = a_batch.view(-1)
            r_batch = r_batch.view(-1)
            d_batch = d_batch.view(-1)
            new_num_full_observations = self._full_observations_top + batch_len
            if new_num_full_observations <= self._max_len_full_obs:
                a_r_d_top = len(self)
                new_a_r_d_len = new_num_full_observations * self.env_n_steps
                for full_s in next_s_batch:
                    self._full_observations[self._full_observations_top] = full_s
                    self._full_observations_top = (self._full_observations_top + 1) % self._max_len_full_obs
                self._a_buffer[a_r_d_top:new_a_r_d_len] = a_batch
                self._r_buffer[a_r_d_top:new_a_r_d_len] = r_batch
                self._d_buffer[a_r_d_top:new_a_r_d_len] = d_batch
            else:
                a_r_d_top = len(self)
                a_r_d_cut = (self.max_len - a_r_d_top)
                idx = 0
                while self._full_observations_top != 0:
                    self._full_observations[self._full_observations_top] = next_s_batch[idx]
                    self._full_observations_top = (self._full_observations_top + 1) % self._max_len_full_obs
                    idx += 1
                self._a_buffer[a_r_d_top:self.max_len] = a_batch[:a_r_d_cut]
                self._r_buffer[a_r_d_top:self.max_len] = r_batch[:a_r_d_cut]
                self._d_buffer[a_r_d_top:self.max_len] = d_batch[:a_r_d_cut]

                a_r_d_top = self._full_observations_top * self.env_n_steps
                new_a_r_d_len = (new_num_full_observations * self.env_n_steps) % self.max_len
                while idx != 0:
                    self._full_observations[self._full_observations_top] = next_s_batch[idx]
                    self._full_observations_top = (self._full_observations_top + 1) % self._max_len_full_obs
                    idx = (idx + 1) % batch_len
                self._a_buffer[a_r_d_top:self.max_len] = a_batch[a_r_d_cut:]
                self._r_buffer[a_r_d_top:self.max_len] = r_batch[a_r_d_cut:]
                self._d_buffer[a_r_d_top:self.max_len] = d_batch[a_r_d_cut:]

    def _get_s(self, idxs):
        states = []
        for i in idxs:
            full_obs_index = int(i // self.env_n_steps)
            step_idx = int(i % self.env_n_steps)
            states.append(self._get_obs(full_obs_index, step_idx))
        return torch.stack(states, dim=0)

    def _get_next_s(self, idxs):
        states = []
        for i in idxs:
            full_obs_index = int(i // self.env_n_steps)
            step_idx = int(i % self.env_n_steps)
            states.append(self._get_obs(full_obs_index, step_idx + 1))
        return torch.stack(states, dim=0)

    def _get_obs(self, full_obs_index, step_idx):
        full_obs = self._full_observations[full_obs_index]
        orig_shape = full_obs.shape
        if self.is_ravelled:
            full_obs = full_obs.view(*full_obs.shape[:-1], self.env_n_steps, -1)
        mask = torch.where(
            torch.arange(self.env_n_steps) < step_idx,
            torch.ones(self.env_n_steps, dtype=full_obs.dtype),
            torch.zeros(self.env_n_steps, dtype=full_obs.dtype)
        ).view(1, self.env_n_steps, 1)
        full_obs = full_obs * mask
        return full_obs.view(*orig_shape) if self.is_ravelled else full_obs

    def __len__(self):
        return self._full_observations_top * self.env_n_steps

    # Only used for logging compatibility
    @property
    def _top(self):
        return self._full_observations_top
