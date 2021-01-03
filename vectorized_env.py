from functools import wraps
import torch

EVERY_STEP_TRUE = 'every_step_true'
EVERY_STEP_EV = 'every_step_ev'
EVERY_STEP_EV_ZEROSUM = 'every_step_ev_zerosum'
END_OF_GAME_TRUE = 'end_of_game_true'
#END_OF_GAME_EV = 3

SUMMED_OBS = 'summed_obs'
LAST_STEP_OBS = 'last_step_obs'
#ONEHOT_OBS = 2


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
        normalize_reward=True,
        env_device=torch.device('cuda'),
        out_device=torch.device('cuda'),
    ):
        # Assert parameter conditions
        assert 0 <= decay_rate <= 1.
        assert reward_type in (EVERY_STEP_TRUE, EVERY_STEP_EV, EVERY_STEP_EV_ZEROSUM, END_OF_GAME_TRUE)
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
        self.opponent = opponent
        if self.opponent is not None:
            assert self.n_players == 2
        if opponent_obs_type is None:
            self.opponent_obs_type = self.obs_type
        else:
            self.opponent_obs_type = opponent_obs_type
        if not normalize_reward or self.reward_type in (END_OF_GAME_TRUE,):
            self.r_norm = 1.
        else:
            self.r_norm = 1. / (torch.sum(self.decay_rate ** torch.arange(self.n_steps, dtype=torch.float32)) * \
                                torch.arange(self.n_bandits, dtype=torch.float32).sum() \
                                / (self.n_bandits * self.n_players))
        self.env_device = env_device
        self.out_device = out_device

        if self.obs_type == SUMMED_OBS:
            self.obs_norm = self.n_bandits / self.n_steps
        else:
            self.obs_norm = 1.
        self.timestep = None
        self.orig_thresholds = None
        self.player_n_pulls = None
        self.player_rewards_sums = None
        self.last_pulls = None
        self.last_rewards = None
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
        self.timestep = 0
        self.orig_thresholds = torch.randint(
            self.sample_resolution + 1,
            size=(self.n_envs, self.n_bandits),
            dtype=torch.float32,
            device=self.env_device
        )
        self.player_n_pulls = torch.zeros((self.n_envs, self.n_players, self.n_bandits), device=self.env_device)
        self.player_rewards_sums = torch.zeros_like(self.player_n_pulls)
        self.last_pulls = torch.zeros_like(self.player_n_pulls)
        self.last_rewards = torch.zeros_like(self.player_n_pulls)
        # self.pulls_onehot = torch.zeros((self.n_envs, self.n_players, self.n_bandits, self.n_steps),
        #                                 device=self.env_device)
        # self.rewards_onehot = torch.zeros_like(self.pulls_onehot)
        
        rewards = torch.zeros((self.n_envs, self.n_players), device=self.env_device) * self.r_norm
        return self.obs, rewards, self.done, self.info_dict
    
    @_single_player_decorator
    @_out_device_decorator
    def step(self, actions):
        if self.opponent is not None:
            if self.opponent_obs_type == SUMMED_OBS:
                opp_obs = self.get_summed_obs()
            elif self.opponent_obs_type == LAST_STEP_OBS:
                opp_obs = self.get_last_step_obs()
            opp_actions = self.opponent(opp_obs[:, 1].unsqueeze(1))
            actions = torch.cat([actions, opp_actions], dim=1)
        assert actions.shape == (self.n_envs, self.n_players), f'actions.shape was: {actions.shape}'
        assert not self.done
        self.timestep += 1
        actions = actions.to(self.env_device).detach()
        
        # Compute agent rewards
        selected_thresholds = self.thresholds.gather(-1, actions)
        pull_rewards = torch.randint(
            self.sample_resolution,
            size=selected_thresholds.shape,
            dtype=torch.float32,
            device=self.env_device) < selected_thresholds
        
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
        
        # Return (obs, reward, done) tuple
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
                draws_mask = rewards_sums[:,0] == rewards_sums[:,1]
                rewards[torch.arange(rewards.shape[0]), winners_idxs] = 1.
                rewards[draws_mask] = 0.5

        rewards = rewards * self.r_norm
        # State, reward, done, info_dict
        return self.obs, rewards, self.done, self.info_dict
    
    @property
    def obs(self):
        if self.obs_type == SUMMED_OBS:
            obs = self.get_summed_obs()
        elif self.obs_type == LAST_STEP_OBS:
            obs = self.get_last_step_obs()
        return obs

    def get_summed_obs(self):
        # Reshape player_n_pulls such that each player receives a tensor of shape (1,1,n_bandits,n_players)
        # The overall obs tensor is then of shape (1,1,n_bandits,n_players+1) with rewards
        # The final axis contains the player's num_pulls first and other player actions listed afterwards
        # This is currently not implemented for more than 2 players
        if self.n_players == 1:
            obs = torch.stack([
                self.player_n_pulls,
                self.player_rewards_sums
            ], dim=-1)
        elif self.n_players == 2:
            player_n_pulls_player_relative = torch.stack([
                self.player_n_pulls,
                self.player_n_pulls[:, [1, 0], :]
            ], dim=-1)
            obs = torch.cat([
                player_n_pulls_player_relative,
                self.player_rewards_sums.unsqueeze(-1)
            ], dim=-1)
        return obs.detach() * self.obs_norm

    def get_last_step_obs(self):
        # Return an observation with only information about the last timestep, useful for RNNs
        # The returned observation for each player is a tensor of shape (1,1,n_bandits,n_players+1)
        # Unlike with SUMMED_OBS, each value is either 0 or 1
        if self.n_players == 1:
            obs = torch.stack([
                self.last_pulls,
                self.last_rewards
            ], dim=-1)
        elif self.n_players == 2:
            last_pulls_relative = torch.stack([
                self.last_pulls,
                self.last_pulls[:, [1, 0], :]
            ], dim=-1)
            obs = torch.cat([
                last_pulls_relative,
                self.last_rewards.unsqueeze(-1)
            ], dim=-1)
        return obs.detach() * self.obs_norm

    @property
    def thresholds(self):
        return self.orig_thresholds * (self.decay_rate ** self.player_n_pulls.sum(dim=1))
    
    @property
    def done(self):
        return self.timestep >= self.n_steps
    
    @property
    def info_dict(self):
        info_dict = {
            'thresholds': self.thresholds,
            'true_player_rewards_sums': self.player_rewards_sums,
        }
        return info_dict
