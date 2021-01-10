import base64
import math
import numpy as np
from pathlib import Path
import pickle
import random
import shutil
from tensorboardX import SummaryWriter
import time
import torch
from torch import distributions
import torch.nn.functional as F
import tqdm

# Custom imports
import vectorized_env as ve


class NFSPVectorized:
    def __init__(self, policy_models, q_models, q_target_models, m_rl_kwargs, m_sl_kwargs,
                 policy_opts=None, q_opts=None,
                 eta=0.1, starting_epsilon=0.08, gamma=0.99,
                 validation_env_kwargs_dicts=(),
                 device=torch.device('cuda'),
                 exp_folder=Path('runs/nfsp/TEMP'),
                 checkpoint_freq=5):
        assert len(policy_models) == 2
        assert len(q_models) == 2
        assert len(q_target_models) == 2
        for i in range(2):
            q_target_models[i].load_state_dict(q_models[i].state_dict())
        if policy_opts is None:
            policy_opts = [torch.optim.Adam(pm) for pm in policy_models]
        else:
            assert len(policy_opts) == len(policy_models)
        if q_opts is None:
            q_opts = [torch.optim.Adam(qm) for qm in q_models]
        else:
            assert len(q_opts) == len(q_models)
        self.nfsp_agents = []
        for i in range(2):
            self.nfsp_agents.append(
                NFSPTrainAgent(
                    CircularReplayBuffer(m_rl_kwargs),
                    ReservoirReplayBuffer(m_sl_kwargs),
                    policy_models[i],
                    policy_opts[i],
                    q_models[i],
                    q_target_models[i],
                    q_opts[i],
                    eta=eta,
                    device=device
                )
            )
        self.starting_epsilon = starting_epsilon
        self.gamma = gamma
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
        self.checkpoint_freq = checkpoint_freq

        self.env = None
        self.epoch_counter = 0
        self.episode_counter = 0
        self.batch_counter = 0
        self.train_step_counter = 0
        self.validation_counter = 0
        self.validation_step_counters = [0] * len(validation_env_kwargs_dicts)
        self.summary_writer = SummaryWriter(self.exp_folder)

    def new_episode(self):
        for agent in self.nfsp_agents:
            agent.eval()
            agent.new_episode()
        return self.env.reset()

    def train(self, env, batch_size, n_epochs, n_expl_steps_per_epoch, n_train_batches_per_epoch,
              n_epochs_q_target_update=1):
        self.env = env
        assert self.env.opponent is None

        print(f'\nRunning main training loop with {n_epochs} epochs')
        for epoch in range(n_epochs):
            print(f'Epoch #{epoch}:')
            if self.epoch_counter % n_epochs_q_target_update == 0:
                for agent in self.nfsp_agents:
                    agent.update_q_target()
            print(f'Sampling {n_expl_steps_per_epoch} time-steps from the environment')
            s, r, done, info_dict = self.new_episode()
            episode_reward_sums = r
            for step in tqdm.trange(n_expl_steps_per_epoch):
                start_time = time.time()
                a = torch.cat([
                    agent.sample_action(
                        player_s,
                        epsilon=self.starting_epsilon / math.sqrt(self.epoch_counter)
                    ) for agent, player_s in zip(self.nfsp_agents, s.to(device=self.device).chunk(2, dim=1))
                ], dim=1)
                next_s, r, done, info_dict = self.env.step(a)
                d_tensor = torch.ones(r.shape).view(-1) if done else torch.zeros(r.shape).view(-1)
                for agent, player_s, player_a, player_r, player_d, player_next_s in zip(
                    self.nfsp_agents,
                    s.chunk(2, dim=1),
                    a.chunk(2, dim=1),
                    r.chunk(2, dim=1),
                    d_tensor.chunk(2, dim=1),
                    next_s.chunk(2, dim=1),
                ):
                    agent.log_s_a_r_d_next_s(
                        player_s.view(-1, *s.shape[-2:]),
                        player_a.view(-1).detach(),
                        player_r.view(-1),
                        player_d,
                        player_next_s.view(-1, *next_s.shape[-2:])
                    )
                s = next_s
                episode_reward_sums += r
                if done:
                    self.log_train_episode(
                        episode_reward_sums.mean().cpu().clone() / self.env.r_norm,
                        info_dict
                    )
                    self.episode_counter += 1
                    s, r, done, info_dict = self.new_episode()
                    episode_reward_sums = r
                self.summary_writer.add_scalar('time/exploration_step_time_ms',
                                               (time.time() - start_time) * 1000,
                                               self.train_step_counter)
                self.train_step_counter += 1

            for agent in self.nfsp_agents:
                agent.train()
            print(f'Training policies on {n_train_batches_per_epoch} batches from the replay buffer')
            for i, agent in enumerate(self.nfsp_agents):
                losses, step_times = agent.train_policy(batch_size, n_train_batches_per_epoch)
                self.log_train_batches(f'agent{i}_policy', losses, step_times)
            print(f'Training Q-networks on {n_train_batches_per_epoch} batches from the replay buffer')
            for i, agent in enumerate(self.nfsp_agents):
                losses, step_times = agent.train_q(batch_size, n_train_batches_per_epoch, self.gamma)
                self.log_train_batches(f'agent{i}_q', losses, step_times)

            self.log_epoch()
            self.epoch_counter += 1
            if self.epoch_counter % self.checkpoint_freq == 0:
                self.run_validation()
                self.save()
            print()
        self.save(finished=True)

    def log_train_batches(self, agent_name, losses, step_times):
        for i, loss, step_time in zip(range(len(losses)), losses, step_times):
            self.summary_writer.add_scalar(f'batch/{agent_name}_loss', loss.numpy().item(),
                                           self.batch_counter - len(losses) + i)
            self.summary_writer.add_scalar(f'time/{agent_name}_step_time_ms', step_time * 1000,
                                           self.batch_counter - len(losses) + i)

    def log_train_episode(self, episode_reward_sums, final_info_dict):
        self.summary_writer.add_scalar(f'episode/reward_{self.env.reward_type}',
                                       episode_reward_sums.numpy().item(),
                                       self.episode_counter)
        pull_rewards_sum = final_info_dict['player_rewards_sums'].cpu().sum(dim=-1)
        self.summary_writer.add_histogram(
            'episode/average_pull_rewards',
            pull_rewards_sum,
            self.episode_counter
        )
        self.summary_writer.add_histogram(
            'episode/agent1_pull_rewards',
            final_info_dict['player_rewards_sums'].cpu().sum(dim=-1)[:, 0].numpy(),
            self.episode_counter
        )
        self.summary_writer.add_histogram(
            'episode/agent2_pull_rewards',
            final_info_dict['player_rewards_sums'].cpu().sum(dim=-1)[:, 1].numpy(),
            self.episode_counter
        )
        self.summary_writer.add_scalar(
            'episode/mean_agent_pull_rewards',
            pull_rewards_sum.mean().numpy().item(),
            self.episode_counter
        )
        self.summary_writer.add_scalar(
            'episode/mean_agent1_pull_rewards',
            final_info_dict['player_rewards_sums'].cpu().sum(dim=-1)[:, 0].mean().numpy().item(),
            self.episode_counter
        )
        self.summary_writer.add_scalar(
            'episode/mean_agent2_pull_rewards',
            final_info_dict['player_rewards_sums'].cpu().sum(dim=-1)[:, 1].mean().numpy().item(),
            self.episode_counter
        )

    def log_epoch(self):
        for i, model in enumerate([agent.policy for agent in self.nfsp_agents]):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.summary_writer.add_histogram(
                        f'params/agent{i}.policy.{name}',
                        param.detach().cpu().clone().numpy(),
                        self.epoch_counter
                    )
        for i, model in enumerate([agent.q for agent in self.nfsp_agents]):
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.summary_writer.add_histogram(
                        f'params/agent{i}.policy.{name}',
                        param.detach().cpu().clone().numpy(),
                        self.epoch_counter
                    )

    def log_validation_episodes(self, agent_name, episode_reward_sums, final_info_dicts):
        assert len(episode_reward_sums) == len(final_info_dicts)
        n_val_envs = len(episode_reward_sums)
        for i, ers, fid in zip(range(n_val_envs), episode_reward_sums, final_info_dicts):
            opponent = self.validation_env_kwargs_dicts[i].get('opponent')
            if opponent is not None:
                opponent = opponent.name
            env_name = opponent

            self.summary_writer.add_histogram(
                f'validation/{agent_name}_{env_name}_game_results',
                ers.numpy(),
                self.validation_counter
            )
            self.summary_writer.add_histogram(
                f'validation/{agent_name}_{env_name}_hero_pull_rewards',
                fid['player_rewards_sums'].sum(dim=-1).cpu()[:, 0].numpy(),
                self.validation_counter
            )
            self.summary_writer.add_histogram(
                f'validation/{agent_name}_{env_name}_villain_pull_rewards',
                fid['player_rewards_sums'].sum(dim=-1).cpu()[:, 1].numpy(),
                self.validation_counter
            )
            self.summary_writer.add_scalar(
                f'validation/{agent_name}_{env_name}_win_percent',
                # W/D/L values are 1/0/-1, so they need to be scaled to 1/0.5/0 to be represented as a percent
                (ers.mean().numpy().item() + 1) / 2. * 100.,
                self.validation_counter
            )
            self.summary_writer.add_scalar(
                f'validation/{agent_name}_{env_name}_mean_hero_pull_rewards',
                fid['player_rewards_sums'].sum(dim=-1).cpu()[:, 0].mean().numpy().item(),
                self.validation_counter
            )
            self.summary_writer.add_scalar(
                f'validation/{agent_name}_{env_name}_mean_villain_pull_rewards',
                fid['player_rewards_sums'].sum(dim=-1).cpu()[:, 1].mean().numpy().item(),
                self.validation_counter
            )

    def run_validation(self, validation_agent_idx):

        for validation_agent_idx, validation_agent in enumerate(self.nfsp_agents):
            validation_agent.eval()
            if len(self.validation_env_kwargs_dicts) > 0:
                print(f'Validating model performance in {len(self.validation_env_kwargs_dicts)} environments')
                episode_reward_sums = []
                final_info_dicts = []
                for i in tqdm.trange(len(self.validation_env_kwargs_dicts)):
                    # Lazily construct validation envs to conserve GPU memory
                    val_env = ve.KaggleMABEnvTorchVectorized(**self.validation_env_kwargs_dicts[i])
                    s, r, done, info_dict = val_env.reset()
                    episode_reward_sums.append(r)
                    while not done:
                        start_time = time.time()
                        a = validation_agent.get_policy_action(s.to(device=self.device))
                        next_s, r, done, info_dict = val_env.step(a)
                        s = next_s
                        episode_reward_sums[-1] += r
                        self.summary_writer.add_scalar(f'time/val_env{i}_policy_step_time_ms',
                                                       (time.time() - start_time) * 1000,
                                                       self.validation_step_counters[i])
                        self.validation_step_counters[i] += 1
                    episode_reward_sums[-1] = episode_reward_sums[-1].mean(dim=-1).cpu().clone() / val_env.r_norm
                    final_info_dicts.append(info_dict)
                self.log_validation_episodes(
                    f'agent{validation_agent_idx}_policy',
                    episode_reward_sums,
                    final_info_dicts
                )

                episode_reward_sums = []
                final_info_dicts = []
                for i in tqdm.trange(len(self.validation_env_kwargs_dicts)):
                    # Lazily construct validation envs to conserve GPU memory
                    val_env = ve.KaggleMABEnvTorchVectorized(**self.validation_env_kwargs_dicts[i])
                    s, r, done, info_dict = val_env.reset()
                    episode_reward_sums.append(r)
                    while not done:
                        start_time = time.time()
                        a = validation_agent.get_q_action(s.to(device=self.device))
                        next_s, r, done, info_dict = val_env.step(a)
                        s = next_s
                        episode_reward_sums[-1] += r
                        self.summary_writer.add_scalar(f'time/val_env{i}_q_step_time_ms',
                                                       (time.time() - start_time) * 1000,
                                                       self.validation_step_counters[i])
                        self.validation_step_counters[i] += 1
                    episode_reward_sums[-1] = episode_reward_sums[-1].mean(dim=-1).cpu().clone() / val_env.r_norm
                    final_info_dicts.append(info_dict)
                self.log_validation_episodes(
                    f'agent{validation_agent_idx}_q',
                    episode_reward_sums,
                    final_info_dicts
                )
        self.validation_counter += 1

    def save(self, finished=False):
        if finished:
            file_path_base = self.exp_folder / f'final_{self.epoch_counter}'
        else:
            file_path_base = self.exp_folder / str(self.epoch_counter)
        for i, agent in enumerate(self.nfsp_agents):
            agent.save(f'{file_path_base}_agent{i}_policy_cp.txt', f'{file_path_base}_agent{i}_q_cp.txt', self.device)


class NFSPTrainAgent:
    def __init__(self, m_rl, m_sl,
                 policy_model, policy_opt,
                 q_model, q_target_model, q_opt,
                 eta, device):
        self.m_rl = m_rl
        self.m_sl = m_sl
        self.policy = policy_model
        self.policy_opt = policy_opt
        self.q = q_model
        self.q_target = q_target_model
        self.q_target.eval()
        self.q_opt = q_opt
        self.eta = eta
        self.device = device

        self.e_greedy_policy = None
        self.new_episode()

    def new_episode(self, n_envs=1):
        self.e_greedy_policy = torch.rand(n_envs) < self.eta

    def sample_action(self, states, epsilon):
        greedy_actions = self.get_q_action(states, epsilon)
        mixed_strategy = self.get_policy_action(states)
        return torch.where(
            self.e_greedy_policy,
            greedy_actions,
            mixed_strategy
        )

    def get_policy_action(self, states):
        return self.policy.sample_action(states)

    def get_q_action(self, states, epsilon=0.):
        return self.q.sample_action_epsilon_greedy(states, epsilon)

    def log_s_a_r_d_next_s(self, s, a, r, d, next_s):
        s = s.cpu().clone()
        a = a.cpu().clone()
        r = r.cpu().clone()
        d = d.cpu().clone()
        next_s = next_s.cpu().clone()
        self.m_rl.append_samples_batch(s, a, r, d, next_s)
        self.m_sl.append_samples_batch(
            s[self.e_greedy_policy],
            a[self.e_greedy_policy],
            r[self.e_greedy_policy],
            d[self.e_greedy_policy],
            next_s[self.e_greedy_policy])

    def train_policy(self, batch_size, n_batches):
        losses = []
        step_times = []
        for batch in tqdm.trange(n_batches):
            start_time = time.time()
            s_batch, a_batch = self.m_sl.sample(batch_size)
            logits = self.policy(s_batch.to(device=self.device))
            log_probs = distributions.Categorical(F.softmax(logits, dim=-1)).log_prob(a_batch.view(-1))
            loss = -log_probs.mean()

            self.policy_opt.zero_grad()
            loss.backward()
            self.policy_opt.step()
            # Used for logging
            losses.append(loss.detach().cpu().clone())
            step_times.append(time.time() - start_time)
        return losses, step_times

    def train_q(self, batch_size, n_batches, gamma):
        losses = []
        step_times = []
        for batch in tqdm.trange(n_batches):
            start_time = time.time()
            s_batch, a_batch, r_batch, d_batch, next_s_batch = self.m_rl.get_samples_batch(batch_size)
            s_batch = s_batch.to(device=self.device)
            a_batch = a_batch.to(device=self.device)
            r_batch = r_batch.to(device=self.device)
            d_batch = d_batch.to(device=self.device)
            next_s_batch = next_s_batch.to(device=self.device)
            q_values = self.q(s_batch.to(device=self.device)).gather(1, a_batch.unsqueeze(1))
            v_next_s = self.q_target(next_s_batch.to(device=self.device)).max(dim=-1)
            # Reward of 0 for terminal states
            v_next_s = (r_batch + gamma * v_next_s * (1. - d_batch))
            # Huber loss for critic
            loss = F.smooth_l1_loss(q_values, v_next_s.detach(), reduction='none').mean()

            self.q_opt.zero_grad()
            loss.backward()
            self.q_opt.step()
            # Used for logging
            losses.append(loss.detach().cpu().clone())
            step_times.append(time.time() - start_time)
        return losses, step_times

    def update_q_target(self):
        self.q_target.load_state_dict(self.q)

    def save(self, policy_file_path, q_file_path, current_device):
        assert policy_file_path != q_file_path
        self.policy.cpu()
        state_dict_bytes = pickle.dumps({
            'model_state_dict': self.policy.state_dict(),
        })
        serialized_string = base64.b64encode(state_dict_bytes)
        with open(policy_file_path, 'w') as f:
            f.write(str(serialized_string))
        self.policy.to(device=current_device)

        self.q.cpu()
        state_dict_bytes = pickle.dumps({
            'model_state_dict': self.q.state_dict(),
        })
        serialized_string = base64.b64encode(state_dict_bytes)
        with open(q_file_path, 'w') as f:
            f.write(str(serialized_string))
        self.q.to(device=current_device)

    def eval(self):
        self.policy.eval()
        self.q.eval()

    def train(self):
        self.policy.train()
        self.q.train()


class CircularReplayBuffer:
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

    def __len__(self):
        return self.current_size


class ReservoirReplayBuffer:
    def __init__(self, s_shape, max_len=1e6):
        self.max_len = int(max_len)
        self._s_buffer = torch.zeros(self.max_len, *s_shape)
        self._a_buffer = torch.zeros(self.max_len)
        self.current_size = 0
        self._top = 0

    def sample(self, sample_size):
        if self.current_size <= sample_size:
            return self._s_buffer, self._a_buffer
        else:
            # Efficient Reservoir Sampling
            # http://erikerlandson.github.io/blog/2015/11/20/very-fast-reservoir-sampling/
            n = self.current_size
            reservoir_idxs = np.arange(sample_size)
            threshold = sample_size * 4
            idx = sample_size
            while idx < n and idx <= threshold:
                m = random.randint(0, idx)
                if m < sample_size:
                    reservoir_idxs[m] = idx
                idx += 1

            while idx < n:
                p = float(sample_size) / idx
                u = random.random()
                g = math.floor(math.log(u) / math.log(1 - p))
                idx = idx + g
                if idx < n:
                    k = random.randint(0, sample_size - 1)
                    reservoir_idxs[k] = idx
                idx += 1

            return (self._s_buffer[torch.from_numpy(reservoir_idxs)],
                    self._a_buffer[torch.from_numpy(reservoir_idxs)])

    def append_samples_batch(self, s_batch, a_batch):
        batch_len = s_batch.shape[0]
        assert a_batch.shape[0] == batch_len
        new_len = self._top + batch_len
        if new_len <= self.max_len:
            self._s_buffer[self._top:new_len] = s_batch
            self._a_buffer[self._top:new_len] = a_batch
            self._top = new_len % self.max_len
            self.current_size = max(new_len, self.current_size)
        else:
            leftover_batch = new_len % self.max_len
            s_batch_split = s_batch.split((batch_len - leftover_batch, leftover_batch))
            a_batch_split = a_batch.split((batch_len - leftover_batch, leftover_batch))
            self.append_samples_batch(s_batch_split[0], a_batch_split[0])
            self.append_samples_batch(s_batch_split[1], a_batch_split[1])

    def __len__(self):
        return self.current_size
