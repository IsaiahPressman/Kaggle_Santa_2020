from functools import wraps
import torch

EVERY_STEP_TRUE = 0
EVERY_STEP_EV = 1
END_OF_GAME_TRUE = 2
#END_OF_GAME_EV = 3


# A vectorized and GPU-compatible recreation of the kaggle "MAB" environment
class KaggleMABEnvTorchVectorized():
    def __init__(
        self,
        # Kaggle MAB env params
        n_bandits=100, n_steps=1999, decay_rate=0.97, sample_resolution=100,
        # Custom params
        n_envs=1,
        n_players=2,
        opponent=None,
        reward_type=EVERY_STEP_TRUE,
        normalize_reward=True,
        env_device=torch.device('cuda'),
        out_device=torch.device('cuda'),
    ):
        # Assert parameter conditions
        assert 0 <= decay_rate <= 1.
        assert reward_type in (EVERY_STEP_TRUE, EVERY_STEP_EV, END_OF_GAME_TRUE)
        if reward_type in (END_OF_GAME_TRUE,):
            assert n_players >= 2
        else:
            assert n_players >= 1
        self.n_bandits = n_bandits
        self.n_steps = n_steps
        self.decay_rate = decay_rate
        self.sample_resolution = sample_resolution
        
        self.n_envs = n_envs
        self.n_players = n_players
        self.opponent = opponent
        if self.opponent is not None:
            assert n_players == 2
        self.reward_type = reward_type
        if not normalize_reward or self.reward_type in (END_OF_GAME_TRUE,):
            self.r_norm = 1.
        else:
            self.r_norm = 1. / (torch.sum(self.decay_rate ** torch.arange(self.n_steps, dtype=torch.float32)) * torch.arange(self.n_bandits, dtype=torch.float32).sum() / (self.n_bandits * self.n_players))
        self.env_device = env_device
        self.out_device = out_device
        
        self.obs_norm = self.n_bandits / self.n_steps
        self.timestep = None
        self.orig_thresholds = None
        self.player_n_pulls = None
        self.player_rewards_sums = None
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
        self.orig_thresholds = torch.randint(self.sample_resolution + 1, size=(self.n_envs, self.n_bandits), dtype=torch.float32, device=self.env_device)
        self.player_n_pulls = torch.zeros((self.n_envs, self.n_players, self.n_bandits), device=self.env_device)
        self.player_rewards_sums = torch.zeros_like(self.player_n_pulls)
        
        rewards = torch.zeros((self.n_envs, self.n_players), device=self.env_device) * self.r_norm
        return self.obs, rewards, self.done, self.info_dict
    
    @_single_player_decorator
    @_out_device_decorator
    def step(self, actions):
        if self.opponent is not None:
            opp_actions = self.opponent(self.obs[:,1].unsqueeze(1))
            actions = torch.cat([actions, opp_actions], dim=1)
        assert actions.shape == (self.n_envs, self.n_players), f'actions.shape was: {actions.shape}'
        assert not self.done
        self.timestep += 1
        actions = actions.to(self.env_device)
        
        # Compute agent rewards
        selected_thresholds = self.thresholds.gather(-1, actions)
        pull_rewards = torch.randint(self.sample_resolution, size=selected_thresholds.shape, dtype=torch.float32, device=self.env_device) < selected_thresholds
        
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
        
        # Return (obs, reward, done) tuple
        if self.reward_type == EVERY_STEP_TRUE:
            rewards = pull_rewards
        elif self.reward_type == EVERY_STEP_EV:
            rewards = selected_thresholds
        elif self.reward_type == END_OF_GAME_TRUE:
            rewards = torch.zeros_like(actions).float()
            if self.timestep == self.n_steps:
                rewards_sums = self.player_rewards_sums.sum(dim=2)
                winners = rewards_sums.argmax(dim=1)
                rewards[winners] = 1.
        
        rewards = rewards * self.r_norm
        # State, reward, done, info_dict
        return self.obs, rewards, self.done, self.info_dict
    
    @property
    def obs(self):
        # Duplicate and reshape player_n_pulls such that each player receives a tensor of shape (1,1,n_bandits,n_players)
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
                self.player_n_pulls[:,[1,0],:]
            ], dim=-1)
            obs = torch.cat([
                player_n_pulls_player_relative,
                self.player_rewards_sums.unsqueeze(-1)
            ], dim=-1)
        else:
            raise RuntimeError('n_players > 2 is not currently supported by obs() due to relative player pulls info')
        return obs * self.obs_norm
    
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
