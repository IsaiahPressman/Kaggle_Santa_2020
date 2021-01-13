from functools import wraps
import torch

EVERY_STEP_TRUE = 'every_step_true'
EVERY_STEP_EV = 'every_step_ev'
EVERY_STEP_EV_ZEROSUM = 'every_step_ev_zerosum'
END_OF_GAME_TRUE = 'end_of_game_true'
# END_OF_GAME_EV = 3

SUMMED_OBS = 'summed_obs'
SUMMED_OBS_WITH_TIMESTEP = 'summed_obs_with_timestep'
LAST_STEP_OBS = 'last_step_obs'
EVERY_STEP_OBS = 'every_step_obs'
EVERY_STEP_OBS_RAVELLED = 'every_step_obs_ravelled'

REWARD_TYPES = (
    EVERY_STEP_TRUE,
    EVERY_STEP_EV,
    EVERY_STEP_EV_ZEROSUM,
    END_OF_GAME_TRUE
)

OBS_TYPES = (
    SUMMED_OBS,
    SUMMED_OBS_WITH_TIMESTEP,
    LAST_STEP_OBS,
    EVERY_STEP_OBS,
    EVERY_STEP_OBS_RAVELLED
)


# A vectorized and GPU-compatible recreation of the kaggle "MAB" environment
class KaggleMABEnvTorchVectorized:
    def __init__(
        self,
        # Kaggle MAB env params
        n_bandits=100, n_steps=1999, decay_rate=0.97, sample_resolution=100,
        # Custom params
        n_envs=1,
        n_players=2,
        reward_type=EVERY_STEP_TRUE,
        obs_type=SUMMED_OBS,
        opponent=None,
        opponent_obs_type=None,
        normalize_reward=False,
        env_device=torch.device('cuda'),
        out_device=torch.device('cuda'),
    ):
        # Assert parameter conditions
        assert 0 <= decay_rate <= 1.
        assert reward_type in REWARD_TYPES
        if reward_type in (END_OF_GAME_TRUE, EVERY_STEP_EV_ZEROSUM):
            assert n_players >= 2
        else:
            assert n_players >= 1
        self.n_bandits = n_bandits
        self.n_steps = n_steps
        self.decay_rate = decay_rate
        self.sample_resolution = sample_resolution
        
        self.n_envs = n_envs
        self.n_players = n_players
        if n_players > 2:
            raise ValueError('n_players > 2 is not currently supported')
        self.reward_type = reward_type
        if self.reward_type == EVERY_STEP_EV_ZEROSUM:
            assert self.n_players == 2
        self.obs_type = obs_type
        assert self.obs_type in OBS_TYPES
        self.opponent = opponent
        if self.opponent is not None:
            assert self.n_players == 2
        if opponent_obs_type is None:
            self.opponent_obs_type = self.obs_type
        else:
            self.opponent_obs_type = opponent_obs_type
        assert self.opponent_obs_type in OBS_TYPES
        if not normalize_reward or self.reward_type in (END_OF_GAME_TRUE,):
            self.r_norm = 1.
        else:
            self.r_norm = 1. / (torch.sum(self.decay_rate ** torch.arange(self.n_steps, dtype=torch.float32)) * \
                                torch.arange(self.n_bandits, dtype=torch.float32).sum() / \
                                (self.n_bandits * self.n_players))
        self.env_device = env_device
        self.out_device = out_device

        if self.obs_type in (SUMMED_OBS, SUMMED_OBS_WITH_TIMESTEP):
            self.obs_norm = self.n_bandits / self.n_steps
        elif self.obs_type in (LAST_STEP_OBS, EVERY_STEP_OBS, EVERY_STEP_OBS_RAVELLED):
            self.obs_norm = 1.
        else:
            raise ValueError(f'Unsupported obs_type: {self.obs_type}')
        if self.opponent_obs_type in (SUMMED_OBS, SUMMED_OBS_WITH_TIMESTEP):
            self.opponent_obs_norm = self.n_bandits / self.n_steps
        elif self.opponent_obs_type in (LAST_STEP_OBS, EVERY_STEP_OBS, EVERY_STEP_OBS_RAVELLED):
            self.opponent_obs_norm = 1.
        else:
            raise ValueError(f'Unsupported opponent obs_type: {self.opponent_obs_type}')
        self.timestep = None
        self.orig_thresholds = None
        self.player_n_pulls = None
        self.player_rewards_sums = None
        self.last_pulls = None
        self.last_rewards = None
        self.all_pulls_onehot = None
        self.all_pull_rewards_onehot = None
        self.store_every_step = (self.obs_type in (EVERY_STEP_OBS, EVERY_STEP_OBS_RAVELLED) or
                                 self.opponent_obs_type in (EVERY_STEP_OBS, EVERY_STEP_OBS_RAVELLED))
        self.reset()
        
    def _single_player_decorator(f):
        @wraps(f)
        def wrapped(self, *args, **kwargs):
            if self.opponent is not None:
                return [out[:,0].unsqueeze(1) if torch.is_tensor(out) else out for out in f(self, *args, **kwargs)]
            else:
                return f(self, *args, **kwargs)
        return wrapped
    
    def _out_device_decorator(f):
        @wraps(f)
        def wrapped(self, *args, **kwargs):
            return [out.to(self.out_device) if torch.is_tensor(out) else out for out in f(self, *args, **kwargs)]
        return wrapped
    
    @_single_player_decorator
    @_out_device_decorator
    def reset(self):
        if self.opponent is not None:
            self.opponent.reset()
        self.timestep = 0
        self.orig_thresholds = torch.randint(
            self.sample_resolution + 1,
            size=(self.n_envs, self.n_bandits),
            dtype=torch.float64,
            device=self.env_device
        )
        self.player_n_pulls = torch.zeros((self.n_envs, self.n_players, self.n_bandits),
                                          device=self.env_device, dtype=torch.float)
        self.player_rewards_sums = torch.zeros_like(self.player_n_pulls)
        self.last_pulls = torch.zeros_like(self.player_n_pulls)
        self.last_rewards = torch.zeros_like(self.player_n_pulls)
        # For EVERY_STEP_OBS, store the action and pull_reward history
        # This is memory intensive, so is only done when necessary
        if self.store_every_step:
            self.all_pulls_onehot = torch.zeros((self.n_envs, self.n_players, self.n_bandits, self.n_steps),
                                                device=self.env_device, dtype=torch.float)
            self.all_pull_rewards_onehot = torch.zeros_like(self.all_pulls_onehot)
            # Deprecated sparse implementation:
            """
            self.all_pull_indices = torch.empty((4, 0), dtype=torch.long, device=self.env_device)
            self.all_pull_rewards = torch.empty(0, dtype=torch.float, device=self.env_device)
            """

        rewards = torch.zeros((self.n_envs, self.n_players), device=self.env_device) * self.r_norm
        return self.obs, rewards, self.done, self.info_dict
    
    @_single_player_decorator
    @_out_device_decorator
    def step(self, actions):
        actions = actions.to(self.env_device)
        if self.opponent is not None:
            if self.opponent_obs_type == SUMMED_OBS:
                opp_obs = self._get_summed_obs()
            elif self.opponent_obs_type == SUMMED_OBS_WITH_TIMESTEP:
                opp_obs = self._get_summed_obs_with_timestep()
            elif self.opponent_obs_type == LAST_STEP_OBS:
                opp_obs = self._get_last_step_obs()
            elif self.opponent_obs_type == EVERY_STEP_OBS:
                opp_obs = self._get_every_step_obs()
            elif self.opponent_obs_type == EVERY_STEP_OBS_RAVELLED:
                opp_obs = self._get_every_step_obs_ravelled()
            else:
                raise ValueError(f'Unsupported opponent obs_type {self.opponent_obs_type}')
            opp_obs = opp_obs * self.opponent_obs_norm
            opp_actions = self.opponent(opp_obs[:, 1].unsqueeze(1))
            actions = torch.cat([actions, opp_actions], dim=1)
        if actions.shape != (self.n_envs, self.n_players):
            raise ValueError(f'actions.shape was: {actions.shape}, should have been {(self.n_envs, self.n_players)}')
        assert not self.done
        self.timestep += 1
        actions = actions.to(self.env_device).detach()
        
        # Compute agent rewards
        selected_thresholds = self.thresholds.gather(-1, actions)
        pull_rewards = torch.randint(
            self.sample_resolution,
            size=selected_thresholds.shape,
            dtype=selected_thresholds.dtype,
            device=self.env_device) < selected_thresholds
        selected_thresholds = selected_thresholds.float()
        
        # Update player_n_pulls and player_rewards_sums
        envs_idxs = torch.arange(self.n_envs).repeat_interleave(self.n_players)
        players_idxs = torch.arange(self.n_players).repeat(self.n_envs)
        self.player_n_pulls[
            envs_idxs,
            players_idxs,
            actions.view(-1)
        ] += 1.
        self.player_rewards_sums[
            envs_idxs,
            players_idxs,
            actions.view(-1)
        ] += pull_rewards.view(-1)
        # Used when obs_type == LAST_STEP_OBS
        self.last_pulls = torch.zeros_like(self.last_pulls)
        self.last_rewards = torch.zeros_like(self.last_rewards)
        self.last_pulls[
            envs_idxs,
            players_idxs,
            actions.view(-1)
        ] = 1.
        self.last_rewards[
            envs_idxs,
            players_idxs,
            actions.view(-1)
        ] += pull_rewards.view(-1)
        # Used when obs_type == EVERY_STEP_OBS
        if self.store_every_step:
            timestep_idxs = torch.zeros_like(envs_idxs) + self.timestep - 1
            self.all_pulls_onehot[
                envs_idxs,
                players_idxs,
                actions.view(-1),
                timestep_idxs
            ] += 1.
            self.all_pull_rewards_onehot[
                envs_idxs,
                players_idxs,
                actions.view(-1),
                timestep_idxs
            ] += pull_rewards.view(-1)
            # Deprecated sparse implementation:
            """
            envs_idxs = envs_idxs.to(device=self.env_device)
            players_idxs = players_idxs.to(device=self.env_device)
            timestep_idxs = torch.zeros_like(envs_idxs) + self.timestep - 1
            this_step_indices = torch.stack([
                envs_idxs,
                players_idxs,
                actions.view(-1),
                timestep_idxs
            ], dim=0).to(self.env_device)
            self.all_pull_indices = torch.cat([
                self.all_pull_indices,
                this_step_indices
            ], dim=1)
            self.all_pull_rewards = torch.cat([
                self.all_pull_rewards,
                pull_rewards.view(-1).float()
            ], dim=0)
            """
        
        if self.reward_type == EVERY_STEP_TRUE:
            rewards = pull_rewards
        elif self.reward_type == EVERY_STEP_EV:
            rewards = selected_thresholds / self.sample_resolution
        elif self.reward_type == EVERY_STEP_EV_ZEROSUM:
            rewards_ev = selected_thresholds / self.sample_resolution
            rewards = torch.stack([
                rewards_ev[:, 0] - rewards_ev[:, 1],
                rewards_ev[:, 1] - rewards_ev[:, 0]
            ], dim=1)
        elif self.reward_type == END_OF_GAME_TRUE:
            rewards = torch.zeros_like(actions).float()
            if self.timestep == self.n_steps:
                rewards_sums = self.player_rewards_sums.sum(dim=2)
                winners_idxs = rewards_sums.argmax(dim=1)
                draws_mask = rewards_sums[:, 0] == rewards_sums[:, 1]
                rewards[torch.arange(rewards.shape[0]), winners_idxs] = 1.
                rewards[torch.arange(rewards.shape[0]), (1 - winners_idxs)] = -1.
                rewards[draws_mask] = 0.

        rewards = rewards * self.r_norm
        # State, reward, done, info_dict
        return self.obs, rewards, self.done, self.info_dict
    
    @property
    def obs(self):
        if self.obs_type == SUMMED_OBS:
            obs = self._get_summed_obs()
        elif self.obs_type == SUMMED_OBS_WITH_TIMESTEP:
            obs = self._get_summed_obs_with_timestep()
        elif self.obs_type == LAST_STEP_OBS:
            obs = self._get_last_step_obs()
        elif self.obs_type == EVERY_STEP_OBS:
            obs = self._get_every_step_obs()
        elif self.obs_type == EVERY_STEP_OBS_RAVELLED:
            obs = self._get_every_step_obs_ravelled()
        else:
            raise ValueError(f'Unsupported obs_type: {self.obs_type}')
        return obs * self.obs_norm

    def _get_summed_obs(self):
        # Each actor receives a tensor of shape: (1, 1, n_bandits, n_players+1) (including pull_rewards)
        # The overall obs tensor shape is: (n_envs, n_players, n_bandits, n_players+1)
        # Reshape player_n_pulls such that each player receives a pulls tensor of shape: (1, 1, n_bandits, n_players)
        # The final axis contains the player's num_pulls first and other player actions listed afterwards
        # This is currently not implemented for more than 2 players
        if self.n_players == 1:
            obs = torch.stack([
                self.player_n_pulls,
                self.player_rewards_sums
            ], dim=-1)
        else:
            player_n_pulls_player_relative = torch.stack([
                self.player_n_pulls,
                self.player_n_pulls[:, [1, 0], :]
            ], dim=-1)
            obs = torch.cat([
                player_n_pulls_player_relative,
                self.player_rewards_sums.unsqueeze(-1)
            ], dim=-1)
        return obs.detach()

    def _get_summed_obs_with_timestep(self):
        # Each actor receives a tensor of shape: (1, 1, n_bandits, n_players+2) (including pull_rewards and timestep)
        # The overall obs tensor shape is: (n_envs, n_players, n_bandits, n_players+2)
        # Reshape player_n_pulls such that each player receives a pulls tensor of shape: (1, 1, n_bandits, n_players)
        # The final axis contains the player's num_pulls first and other player actions listed afterwards
        # This is currently not implemented for more than 2 players
        timestep_broadcasted = (torch.zeros((self.n_envs, self.n_players, self.n_bandits),
                                            device=self.env_device) + self.timestep) / self.n_bandits
        if self.n_players == 1:
            obs = torch.stack([
                self.player_n_pulls,
                self.player_rewards_sums,
                timestep_broadcasted
            ], dim=-1)
        else:
            player_n_pulls_player_relative = torch.stack([
                self.player_n_pulls,
                self.player_n_pulls[:, [1, 0], :]
            ], dim=-1)
            obs = torch.cat([
                player_n_pulls_player_relative,
                self.player_rewards_sums.unsqueeze(-1),
                timestep_broadcasted.unsqueeze(-1)
            ], dim=-1)
        return obs.detach()

    def _get_last_step_obs(self):
        # Return an observation with only information about the last timestep, useful for RNNs
        # Each actor receives a tensor of shape: (1, 1, n_bandits, n_players+1)
        # The overall obs tensor shape is: (n_envs, n_players, n_bandits, n_players+1)
        # Unlike with SUMMED_OBS, each value is either 0. or 1., depending only on what happened in the last step
        if self.n_players == 1:
            obs = torch.stack([
                self.last_pulls,
                self.last_rewards
            ], dim=-1)
        else:
            last_pulls_relative = torch.stack([
                self.last_pulls,
                self.last_pulls[:, [1, 0], :]
            ], dim=-1)
            obs = torch.cat([
                last_pulls_relative,
                self.last_rewards.unsqueeze(-1)
            ], dim=-1)
        return obs.detach()
    
    def _get_every_step_obs(self):
        # Return an observation with information about every timestep
        # Each actor receives a tensor of shape: (1, 1, n_bandits, n_steps, n_players+1)
        # The overall obs tensor shape is: (n_envs, n_players, n_bandits, n_steps, n_players+1)
        # The output is a mostly sparse tensor where each value is either 0. or 1.
        if not self.store_every_step:
            raise RuntimeError('This environment is not storing every_step information')
        # Deprecated sparse implementation:
        """
        all_pulls_onehot = torch.sparse.FloatTensor(
            self.all_pull_indices,
            torch.ones(self.all_pull_indices.shape[1], dtype=torch.float, device=self.env_device),
            torch.Size([self.n_envs, self.n_players, self.n_bandits, self.n_steps])
        ).to_dense()
        all_rewards = torch.sparse.FloatTensor(
            self.all_pull_indices,
            self.all_pull_rewards,
            torch.Size([self.n_envs, self.n_players, self.n_bandits, self.n_steps])
        ).to_dense()
        """
        if self.n_players == 1:
            obs = torch.stack([
                self.all_pulls_onehot,
                self.all_pull_rewards_onehot
            ], dim=-1)
        else:
            all_pulls_relative = torch.stack([
                self.all_pulls_onehot,
                self.all_pulls_onehot[:, [1, 0], :]
            ], dim=-1)
            obs = torch.cat([
                all_pulls_relative,
                self.all_pull_rewards_onehot.unsqueeze(-1)
            ], dim=-1)
        return obs.detach()

    def _get_every_step_obs_ravelled(self):
        # Same as every_step_obs, but with each bandit's sample "ravelled"
        # Each actor receives a tensor of shape: (1, 1, n_bandits, n_steps * (n_players+1))
        # The overall obs tensor shape is: (n_envs, n_players, n_bandits, n_steps * n(_players+1))
        return self._get_every_step_obs().view(self.n_envs, self.n_players, self.n_bandits, -1)

    @property
    def thresholds(self):
        return self.orig_thresholds * (self.decay_rate ** self.player_n_pulls.sum(dim=1).double())
    
    @property
    def done(self):
        return self.timestep >= self.n_steps
    
    @property
    def info_dict(self):
        info_dict = {
            'thresholds': self.thresholds,
            'player_rewards_sums': self.player_rewards_sums,
        }
        return info_dict
