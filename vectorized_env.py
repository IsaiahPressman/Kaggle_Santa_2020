from functools import wraps
import torch

EVERY_STEP_TRUE = 'every_step_true'
EVERY_STEP_EV = 'every_step_ev'
EVERY_STEP_EV_ZEROSUM = 'every_step_ev_zerosum'
END_OF_GAME_TRUE = 'end_of_game_true'

SUMMED_OBS = 'summed_obs'
SUMMED_OBS_NOISE='summed_obs_noise'
SUMMED_OBS_WITH_TIMESTEP = 'summed_obs_with_timestep'
LAST_STEP_OBS = 'last_step_obs'
EVERY_STEP_OBS = 'every_step_obs'
EVERY_STEP_OBS_RAVELLED = 'every_step_obs_ravelled'
SUMMED_AND_LAST_TEN='summed_and_last_ten'
SUMMED_AND_LAST_TEN = 'summed_and_last_ten'
LAST_60_EVENTS_OBS = 'last_60_events_obs'
LAST_60_EVENTS_AND_SUMMED_OBS = 'last_60_events_and_summed_obs'
DECAYING_OBS = 'decaying_obs'
SUMMED_AND_DECAY='summed_and_decay'
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
    SUMMED_OBS_NOISE,
    EVERY_STEP_OBS,
    EVERY_STEP_OBS_RAVELLED,
    SUMMED_AND_LAST_TEN,
    SUMMED_AND_DECAY,
    LAST_60_EVENTS_OBS,
    LAST_60_EVENTS_AND_SUMMED_OBS,
    DECAYING_OBS
)

NO_INFO_VAL = 0.


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
        n_decay=5,
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
        self.n_decay=n_decay
        self.n_envs = n_envs
        self.n_players = n_players
        if n_players > 2:
            raise ValueError('n_players > 2 is not currently supported')
        self.reward_type = reward_type
        self.obs_type = obs_type
        assert self.obs_type in OBS_TYPES, f'obs_type "{self.obs_type}" is not recognized'
        self.opponent = opponent
        if self.opponent is not None:
            assert self.n_players == 2
        if opponent_obs_type is None:
            self.opponent_obs_type = self.obs_type
        else:
            self.opponent_obs_type = opponent_obs_type
        assert self.opponent_obs_type in OBS_TYPES, f'opponent_obs_type "{self.opponent_obs_type}" is not recognized'
        if not normalize_reward or self.reward_type in (EVERY_STEP_EV_ZEROSUM, END_OF_GAME_TRUE):
            self.r_norm = 1.
        else:
            self.r_norm = 1. / (torch.sum(self.decay_rate ** torch.arange(self.n_steps, dtype=torch.float32)) * \
                                torch.arange(self.n_bandits, dtype=torch.float32).sum() / \
                                (self.n_bandits * self.n_players))
        self.env_device = env_device
        self.out_device = out_device

        self.obs_norm_dict = {
            SUMMED_OBS: self.n_bandits / self.n_steps,
            SUMMED_OBS_WITH_TIMESTEP: self.n_bandits / self.n_steps,
            LAST_STEP_OBS: 1.,
            SUMMED_OBS_NOISE: self.n_bandits / self.n_steps,
            EVERY_STEP_OBS: 1.,
            EVERY_STEP_OBS_RAVELLED: 1.,
            SUMMED_AND_LAST_TEN:1.,
            LAST_60_EVENTS_OBS: 1.,
            LAST_60_EVENTS_AND_SUMMED_OBS: 1.,
            DECAYING_OBS: 1.,
            SUMMED_AND_DECAY: 1.
        }
        if self.obs_type not in self.obs_norm_dict.keys():
            raise RuntimeError(f'obs_type "{self.obs_type}" does not have a defined obs_norm')
        if self.opponent_obs_type not in self.obs_norm_dict.keys():
            raise RuntimeError(f'opponent_obs_type "{self.opponent_obs_type}" does not have a defined obs_norm')
        self.timestep = None
        self.orig_thresholds = None
        self.player_n_pulls = None
        self.player_rewards_sums = None
        self.last_pulls = None
        self.last_rewards = None
        self.all_pulls_onehot = None
        self.all_pull_rewards_onehot = None
        self.store_every_step = (
                self.obs_type in (EVERY_STEP_OBS, EVERY_STEP_OBS_RAVELLED) or
                self.opponent_obs_type in (EVERY_STEP_OBS, EVERY_STEP_OBS_RAVELLED)
        )
        self.last_10_pulls = None
        self.last_10_rewards = None
        self.store_last_ten = self.obs_type == SUMMED_AND_LAST_TEN or self.opponent_obs_type == SUMMED_AND_LAST_TEN
        self.last_60_pull_events = None
        self.last_60_reward_events = None
        self.last_60_event_timestamps = None
        self.last_60_event_indices = None
        self.store_events = (
                self.obs_type in (LAST_60_EVENTS_OBS, LAST_60_EVENTS_AND_SUMMED_OBS) or
                self.opponent_obs_type in (LAST_60_EVENTS_OBS, LAST_60_EVENTS_AND_SUMMED_OBS)
        )
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
            dtype=torch.float,
            device=self.env_device
        )
        self.player_n_pulls = torch.zeros((self.n_envs, self.n_players, self.n_bandits),
                                          device=self.env_device, dtype=torch.float)
        self.player_rewards_sums = torch.zeros_like(self.player_n_pulls)
        self.last_pulls = torch.zeros_like(self.player_n_pulls)
        self.last_rewards = torch.zeros_like(self.player_n_pulls)

        self.decay_pulls=torch.zeros((self.n_envs, self.n_players, self.n_bandits,self.n_decay),
                                          device=self.env_device, dtype=torch.float)
        self.decay_rewards=torch.zeros((self.n_envs, self.n_players, self.n_bandits,self.n_decay),
                                    device=self.env_device, dtype=torch.float)
        # For EVERY_STEP_OBS, store the action and pull_reward history
        # This is memory intensive, so is only done when necessary
        if self.store_every_step:
            self.all_pulls_onehot = torch.zeros((self.n_envs, self.n_players, self.n_bandits, self.n_steps),
                                                device=self.env_device, dtype=torch.float) + NO_INFO_VAL
            self.all_pull_rewards_onehot = torch.zeros_like(self.all_pulls_onehot) + NO_INFO_VAL
        if self.store_last_ten:
            self.last_10_pulls = torch.zeros((self.n_envs, self.n_players, self.n_bandits, 10),
                                             device=self.env_device, dtype=torch.float) + NO_INFO_VAL
            self.last_10_rewards = torch.zeros_like(self.last_10_pulls)
        if self.store_events:
            self.last_60_pull_events = torch.zeros((self.n_envs, self.n_players, self.n_bandits, 60),
                                                   device=self.env_device, dtype=torch.float) + NO_INFO_VAL
            self.last_60_reward_events = torch.zeros_like(self.last_60_pull_events) + NO_INFO_VAL
            self.last_60_event_timestamps = torch.zeros_like(self.last_60_pull_events) + NO_INFO_VAL
            self.last_60_event_indices = torch.zeros((self.n_envs, self.n_bandits),
                                                     device=self.env_device, dtype=torch.long)

        rewards = torch.zeros((self.n_envs, self.n_players), device=self.env_device) * self.r_norm
        return self.obs, rewards, self.done, self.info_dict
    
    @_single_player_decorator
    @_out_device_decorator
    def step(self, actions):
        actions = actions.to(self.env_device)
        if self.opponent is not None:
            if self.opponent_obs_type == SUMMED_OBS:
                opp_obs = self._get_summed_obs()
            if self.opponent_obs_type == SUMMED_AND_DECAY:
                opp_obs = self._get_summed_and_decay_obs()
            elif self.opponent_obs_type == SUMMED_OBS_WITH_TIMESTEP:
                opp_obs = self._get_summed_obs_with_timestep() 
            elif self.opponent_obs_type == SUMMED_OBS_NOISE:
                opp_obs = self._get_summed_obs_noise() 
            elif self.opponent_obs_type == LAST_STEP_OBS:
                opp_obs = self._get_last_step_obs()
            elif self.opponent_obs_type == EVERY_STEP_OBS:
                opp_obs = self._get_every_step_obs()
            elif self.opponent_obs_type == EVERY_STEP_OBS_RAVELLED:
                opp_obs = self._get_every_step_obs_ravelled()
            elif self.opponent_obs_type == SUMMED_AND_LAST_TEN:
                opp_obs = self._get_summed_obs_and_last_ten()
            elif self.opponent_obs_type == LAST_60_EVENTS_OBS:
                opp_obs = self._get_last_60_events_obs()
            elif self.opponent_obs_type == LAST_60_EVENTS_AND_SUMMED_OBS:
                opp_obs = self._get_last_60_events_and_summed_obs()
            elif self.opponent_obs_type == DECAYING_OBS:
                opp_obs = self._get_decaying_obs()
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
        pull_rewards = pull_rewards.to(dtype=torch.float)
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
        self.last_pulls[
            envs_idxs,
            players_idxs,
            actions.view(-1)
        ] = 1.
        self.last_rewards[
            envs_idxs,
            players_idxs,
            actions.view(-1)
        ] = pull_rewards.view(-1)
        decay_rates=torch.exp(-1*torch.arange(self.n_decay,device=self.env_device).float()).reshape(1,1,1,-1)
        self.decay_pulls=self.decay_pulls+self.last_pulls.unsqueeze(dim=-1)*decay_rates
        self.decay_rewards =self.decay_rewards+self.last_rewards.unsqueeze(dim=-1)*decay_rates
        self.decay_pulls=self.decay_pulls*(1-decay_rates)
        self.decay_rewards=self.decay_rewards*(1-decay_rates)
        # Used when obs_type == EVERY_STEP_OBS
        if self.store_every_step:
            timestep_idxs = torch.zeros_like(envs_idxs) + self.timestep - 1
            self.all_pulls_onehot[
                envs_idxs,
                players_idxs,
                actions.view(-1),
                timestep_idxs
            ] = 1.
            self.all_pull_rewards_onehot[
                envs_idxs,
                players_idxs,
                actions.view(-1),
                timestep_idxs
            ] = pull_rewards.view(-1)
        if self.store_last_ten:
            self.last_10_pulls = torch.cat([
                self.last_10_pulls,
                self.last_pulls.clone().unsqueeze(-1)
            ], dim=-1)[:, :, :, 1:]
            self.last_10_rewards = torch.cat([
                self.last_10_rewards,
                self.last_rewards.clone().unsqueeze(-1)
            ], dim=-1)[:, :, :, 1:]
        if self.store_events:
            event_idxs = self.last_60_event_indices.gather(-1, actions)
            self.last_60_pull_events[
                envs_idxs,
                players_idxs,
                actions[:, [1, 0]].view(-1),
                event_idxs[:, [1, 0]].view(-1)
            ] = 0.
            self.last_60_pull_events[
                envs_idxs,
                players_idxs,
                actions.view(-1),
                event_idxs.view(-1)
            ] = 1.
            self.last_60_reward_events[
                envs_idxs,
                players_idxs,
                actions[:, [1, 0]].view(-1),
                event_idxs[:, [1, 0]].view(-1)
            ] = 0.
            self.last_60_reward_events[
                envs_idxs,
                players_idxs,
                actions.view(-1),
                event_idxs.view(-1)
            ] = pull_rewards.view(-1)
            self.last_60_event_timestamps[
                envs_idxs,
                :,
                actions.view(-1),
                event_idxs.view(-1)
            ] = float(self.timestep) / self.n_steps
            self.last_60_event_indices.scatter_(-1, actions, event_idxs + 1)
            # This will lose information after the 60th event, but this should happen rarely
            self.last_60_event_indices = torch.where(
                self.last_60_event_indices > 59,
                torch.zeros_like(self.last_60_event_indices) + 59,
                self.last_60_event_indices
            )

        if self.reward_type == EVERY_STEP_TRUE:
            rewards = pull_rewards
        elif self.reward_type == EVERY_STEP_EV:
            rewards = selected_thresholds.ceil() / self.sample_resolution
        elif self.reward_type == EVERY_STEP_EV_ZEROSUM:
            rewards_ev = selected_thresholds.ceil() / self.sample_resolution
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
        else:
            raise ValueError(f'Unrecognized reward_type: {self.reward_type}')

        rewards = rewards * self.r_norm
        # State, reward, done, info_dict
        return self.obs, rewards, self.done, self.info_dict
    
    @property
    def obs(self):
        if self.obs_type == SUMMED_OBS:
            obs = self._get_summed_obs()
        if self.obs_type == SUMMED_AND_DECAY:
            obs = self._get_summed_and_decay_obs()
        elif self.obs_type == SUMMED_OBS_WITH_TIMESTEP:
            obs = self._get_summed_obs_with_timestep()
        elif self.obs_type == SUMMED_OBS_NOISE:
            obs = self._get_summed_obs_noise() 
        elif self.obs_type == LAST_STEP_OBS:
            obs = self._get_last_step_obs()
        elif self.obs_type == EVERY_STEP_OBS:
            obs = self._get_every_step_obs()
        elif self.obs_type == EVERY_STEP_OBS_RAVELLED:
            obs = self._get_every_step_obs_ravelled()
        elif self.obs_type == SUMMED_AND_LAST_TEN:
            obs = self._get_summed_obs_and_last_ten()
        elif self.obs_type == LAST_60_EVENTS_OBS:
            obs = self._get_last_60_events_obs()
        elif self.obs_type == LAST_60_EVENTS_AND_SUMMED_OBS:
            obs = self._get_last_60_events_and_summed_obs()
        elif self.obs_type == DECAYING_OBS:
            obs = self._get_decaying_obs()
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

    def _get_summed_obs_noise(self):
        # Reshape player_n_pulls such that each player receives a tensor of shape (1,1,n_bandits,n_players)
        # The overall obs tensor is then of shape (1,1,n_bandits,n_players+2) (including rewards and timestep)
        # The final axis contains the player's num_pulls first and other player actions listed afterwards
        # This is currently not implemented for more than 2 players
        # Rescale the noise so that it is the intended noise value after being normalized
        noise = torch.randn_like(self.player_rewards_sums) * self.n_steps / self.n_bandits
        if self.n_players == 1:
            obs = torch.stack([
                self.player_n_pulls,
                self.player_rewards_sums,
                noise
            ], dim=-1)
        else:
            player_n_pulls_player_relative = torch.stack([
                self.player_n_pulls,
                self.player_n_pulls[:, [1, 0], :]
            ], dim=-1)
            obs = torch.cat([
                player_n_pulls_player_relative,
                self.player_rewards_sums.unsqueeze(-1),
                noise.unsqueeze(-1)
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
        # The overall obs tensor shape is: (n_envs, n_players, n_bandits, n_steps * (n_players+1))
        return self._get_every_step_obs().view(self.n_envs, self.n_players, self.n_bandits, -1)
    
    def _get_summed_and_decay_obs(self):
        summed_obs=self._get_summed_obs()/self.n_steps
        result=torch.cat([summed_obs,self.decay_pulls,self.decay_rewards],dim=3)
        return result
    
    def _get_summed_obs_and_last_ten(self):
        # The overall obs tensor shape is: (n_envs, n_players, n_bandits, 3+10 * n(_players+1))
        """time = self.timestep
        all_steps = self._get_every_step_obs()[:, :, :, max(0, time-10):time, :].view(
            self.n_envs,
            self.n_players,
            self.n_bandits,
            -1
        )
        all_steps = torch.nn.functional.pad(all_steps, (max(30-time*3, 0), 0), "constant", NO_INFO_VAL)
        summed_obs = self._get_summed_obs()
        result = torch.cat([all_steps, summed_obs], dim=3)
        return result"""
        if not self.store_last_ten:
            raise RuntimeError('This environment is not storing last_ten information')
        if self.n_players == 1:
            obs = torch.stack([
                self.last_10_pulls,
                self.last_10_rewards
            ], dim=-1)
        else:
            last_10_pulls_relative = torch.stack([
                self.last_10_pulls,
                self.last_10_pulls[:, [1, 0], :]
            ], dim=-1)
            obs = torch.cat([
                last_10_pulls_relative,
                self.last_10_rewards.unsqueeze(-1)
            ], dim=-1)
        obs = torch.cat([
            obs.view(self.n_envs, self.n_players, self.n_bandits, -1),
            self._get_summed_obs() * self.obs_norm_dict[SUMMED_OBS]
        ], dim=-1)
        return obs.detach()

    def _get_last_60_events_obs(self):
        # Return an observation with information about the last 60 "events" per arm
        # An event is when either player pulls that arm
        # Each actor receives a tensor of shape: (1, 1, n_bandits, 60 * (n_players+2))
        # The overall obs tensor shape is: (n_envs, n_players, n_bandits, 60 * (n_players+2))
        # 60 * (n_players+2): 60 events * (my_pull, opp_pull, my_reward, timestamp)
        # The output is a mostly sparse tensor, where each pull and reward value is either NO_INFO_VAL, 0. or 1.
        # NO_INFO_VAL indicates that the event has not yet happened
        # The timestamps are in the range 0. to 1., depending on when the event occurred
        if not self.store_events:
            raise RuntimeError('This environment is not storing last_60_events information')
        if self.n_players == 1:
            obs = torch.stack([
                self.last_60_pull_events,
                self.last_60_reward_events,
                self.last_60_event_timestamps
            ], dim=-1)
        else:
            pull_events_relative = torch.stack([
                self.last_60_pull_events,
                self.last_60_pull_events[:, [1, 0], :]
            ], dim=-1)
            obs = torch.cat([
                pull_events_relative,
                self.last_60_reward_events.unsqueeze(-1),
                self.last_60_event_timestamps.unsqueeze(-1),
            ], dim=-1)
        return obs.view(self.n_envs, self.n_players, self.n_bandits, -1)

    def _get_last_60_events_and_summed_obs(self):
        # _get_last_60_events_obs and _get_summed_obs_with_timestep concatenated
        return torch.cat([
            self._get_last_60_events_obs(),
            self._get_summed_obs_with_timestep() * self.obs_norm_dict[SUMMED_OBS_WITH_TIMESTEP]
        ], dim=-1)

    def _get_decaying_obs(self):
        assert False, 'TODO: not yet implemented'

    @property
    def thresholds(self):
        return self.orig_thresholds * (self.decay_rate ** self.player_n_pulls.sum(dim=1))

    @property
    def obs_norm(self):
        return self.obs_norm_dict[self.obs_type]

    @property
    def opponent_obs_norm(self):
        return self.obs_norm_dict[self.opponent_obs_type]

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
