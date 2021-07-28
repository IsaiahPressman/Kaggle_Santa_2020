import base64
from copy import copy
import numpy as np
import pickle
import time
import torch
from torch import distributions
import tqdm

import graph_nns as gnn
import vectorized_env as ve


def run_vectorized_vs(p1, p2, p1_name=None, p2_name=None, display_out=True, **env_kwargs):
    env_kwargs = copy(env_kwargs)
    assert 'opponent' not in env_kwargs.keys(), 'Pass opponent as p2 arg, not as opponent kwarg'
    if 'obs_type' not in env_kwargs.keys():
        env_kwargs['obs_type'] = p1.obs_type
    if 'opp_obs_type' not in env_kwargs.keys():
        env_kwargs['opp_obs_type'] = p2.obs_type
    if p1_name is None:
        try:
            p1_name = p1.name
        except AttributeError:
            pass
    if p2_name is None:
        try:
            p2_name = p2.name
        except AttributeError:
            pass

    p1.reset()
    p2.reset()
    rewards = []
    vs_env = ve.KaggleMABEnvTorchVectorized(opponent=p2, **env_kwargs)
    s, r, _, _ = vs_env.reset()
    rewards.append(r)
    iterator = tqdm.trange if display_out else range
    for i in iterator(vs_env.n_steps):
        s, r, _, _ = vs_env.step(p1(s))
        rewards.append(r)
    p1_scores, p2_scores = vs_env.player_rewards_sums.sum(-1).chunk(2, dim=1)
    p1_mean_score = (torch.sum(p1_scores > p2_scores) + torch.sum(p1_scores == p2_scores) * 0.5) / vs_env.n_envs
    if display_out:
        print(f'{p1_name} -vs- {p2_name}')
        print(f'Mean scores: {p1_scores.mean():.2f} - {p2_scores.mean():.2f}')
        print(f'Match score: {torch.sum(p1_scores > p2_scores)} - '
              f'{torch.sum(p1_scores == p2_scores)} - '
              f'{torch.sum(p1_scores < p2_scores)} '
              f'({torch.sum(p1_scores > p2_scores) * 100. / vs_env.n_envs:.1f}% - '
              f'{torch.sum(p1_scores == p2_scores) * 100. / vs_env.n_envs:.1f}% - '
              f'{torch.sum(p1_scores < p2_scores) * 100. / vs_env.n_envs:.1f}%)')
        time.sleep(0.5)
    return p1_mean_score.cpu().numpy().item(), torch.stack(rewards, dim=0)


class VectorizedAgent:
    def __init__(self):
        self.name = None
        self.obs_type = None

    def __call__(self, states):
        return None

    def __eq__(self, other):
        if isinstance(other, VectorizedAgent):
            return self.name == other.name
        else:
            return False

    def reset(self):
        pass


class AlwaysFirstAgent(VectorizedAgent):
    def __init__(self):
        super().__init__()
        self.name = 'AlwaysFirstAgent'
        self.obs_type = ve.SUMMED_OBS
        
    def __call__(self, states):
        return torch.zeros(size=states.shape[:-2], dtype=torch.long, device=states.device)

    
class BasicThompsonSampling(VectorizedAgent):
    def __init__(self, obs_norm=100. / 1999., n_bandits=100):
        super().__init__()
        self.obs_norm = obs_norm
        self.n_bandits = n_bandits
        self.name = 'BasicThompsonSampling'
        self.obs_type = ve.SUMMED_OBS
        
    def __call__(self, states):
        assert states.shape[-2:] == (self.n_bandits, 3)
        states = states / self.obs_norm
        
        my_pulls, _, post_a = states.chunk(3, dim=-1)
        post_b = my_pulls - post_a
        
        actions = distributions.beta.Beta(post_a.view(-1, 100) + 1, post_b.view(-1, 100) + 1).sample().argmax(dim=-1)
        return actions.view(states.shape[:-2])

    
class MultiAgent(VectorizedAgent):
    def __init__(self, agents, envs_dim=0):
        super().__init__()
        self.agents = agents
        # Splits batch observations among agents along envs_dim
        # envs_dim = 0 assumes that the observations are of the shape:
        # n_envs, n_players, n_bandits, features
        self.envs_dim = envs_dim
        self.name = 'MultiAgent'
        obs_types_unique = np.unique([a.obs_type for a in self.agents])
        assert len(obs_types_unique) == 1
        self.obs_type = obs_types_unique.item()
    
    def __call__(self, states):
        states_chunked = states.chunk(len(self.agents), dim=self.envs_dim)
        if len(states_chunked) != len(self.agents):
            raise ValueError(f'There were only {states.shape[self.envs_dim]} envs for {len(self.agents)} agents, '
                             f'from states tensor of shape {states.shape}')
        else:
            return torch.cat([a(s) for a, s in zip(self.agents, states_chunked)], dim=self.envs_dim)

    def reset(self):
        for agent in self.agents:
            agent.reset()


class PullVegasSlotMachines(VectorizedAgent):
    def __init__(self, obs_norm=100. / 1999., n_bandits=100):
        super().__init__()
        self.obs_norm = obs_norm
        self.n_bandits = n_bandits
        self.name = 'PullVegasSlotMachines'
        self.obs_type = ve.SUMMED_OBS
    
    def __call__(self, states):
        assert states.shape[-2:] == (self.n_bandits, 3)
        states = states / self.obs_norm
        
        # Take random action on the first step
        if states.sum() == 0:
            actions = torch.randint(self.n_bandits, size=states.shape[:-2], device=states.device)
        else:
            my_pulls, opp_pulls, wins = states.chunk(3, dim=-1)
            losses = my_pulls - wins
            ev = ((wins+1) - losses + opp_pulls - (opp_pulls>0)*1.5) / ((wins+1) + losses + opp_pulls)
            actions = ev.squeeze(-1).argmax(dim=-1)
        return actions


class PullVegasSlotMachinesImproved(VectorizedAgent):
    def __init__(self, obs_norm=100. / 1999., n_bandits=100):
        super().__init__()
        self.obs_norm = obs_norm
        self.n_bandits = n_bandits
        self.name = 'PullVegasSlotMachinesImproved'
        self.obs_type = ve.SUMMED_OBS

        self.last_states = None
        self.wins = None
        self.losses = None
        self.opp_pulls = None
        self.opp_last_action = None
        self.opp_continues = None
        self.my_actions_list = []

    def __call__(self, states):
        assert states.shape[-2:] == (self.n_bandits, 3)
        states = states / self.obs_norm
        if self.last_states is None:
            self.last_states = torch.zeros_like(states)
        _, opp_pull, win = [t.squeeze(-1) for t in (states - self.last_states).chunk(3, dim=-1)]
        self.my_pulls, self.opp_pulls, self.wins = [t.squeeze(-1) for t in states.chunk(3, dim=-1)]
        self.losses = self.my_pulls - self.wins
        self.wins += 1.
        self.last_states = states.clone()
        if self.opp_last_action is None:
            self.opp_last_action = torch.zeros_like(win)
            self.opp_continues = torch.zeros_like(win)
        else:
            self.opp_pulls += opp_pull
            if len(self.my_actions_list) >= 2:
                self.opp_continues = torch.where(
                    (opp_pull == 1) & (self.opp_last_action == 1),
                    self.opp_continues + 1.,
                    torch.zeros_like(self.opp_continues)
                )

        win = win.sum(dim=-1)
        next_bandits = self.get_next_bandit()
        if len(self.my_actions_list) == 0:
            actions = torch.randint(self.n_bandits, size=states.shape[:-2], device=states.device)
        elif len(self.my_actions_list) < 3:
            actions = torch.where(
                win == 1,
                self.my_actions_list[-1],
                next_bandits
            )
        else:
            actions_if_three_in_a_row = torch.where(
                (self.my_actions_list[-1] == self.my_actions_list[-2]) & (
                        self.my_actions_list[-1] == self.my_actions_list[-3]),
                torch.where(
                    torch.rand(size=next_bandits.shape, device=states.device) < 0.5,
                    self.my_actions_list[-1],
                    next_bandits
                ),
                next_bandits
            )
            actions = torch.where(
                win == 1,
                self.my_actions_list[-1],
                actions_if_three_in_a_row
            )
        self.my_actions_list.append(actions)
        self.opp_last_action = opp_pull.clone()
        return actions

    def get_next_bandit(self):
        ev = (self.wins - self.losses + self.opp_pulls - (self.opp_pulls > 0)*1.5 + self.opp_continues) \
            / (self.wins + self.losses + self.opp_pulls) \
            * torch.pow(0.97, self.wins + self.losses + self.opp_pulls)
        return ev.argmax(dim=-1)

    def reset(self):
        self.last_states = None
        self.wins = None
        self.losses = None
        self.opp_pulls = None
        self.opp_last_action = None
        self.opp_continues = None
        self.my_actions_list = []

    
class RandomAgent(VectorizedAgent):
    def __init__(self, n_bandits=100):
        super().__init__()
        self.n_bandits = n_bandits
        self.name = 'RandomAgent'
        self.obs_type = ve.SUMMED_OBS
        
    def __call__(self, states):
        assert states.shape[-2] == self.n_bandits
        return torch.randint(self.n_bandits, size=states.shape[:-2], device=states.device)


# Saved RL agents
class SavedRLAgent(VectorizedAgent):
    def __init__(self, agent_name, device=torch.device('cuda'), deterministic_policy=False):
        super().__init__()
        self.agent_name = agent_name
        self.name = f'SavedRLAgent: {self.agent_name}'
        if deterministic_policy:
            self.name += '_deterministic'
        if self.agent_name == 'a3c_agent_v0':
            self.model = gnn.GraphNNActorCritic(
                in_features=3,
                n_nodes=100,
                n_hidden_layers=1,
                layer_sizes=16,
                layer_class=gnn.FullyConnectedGNNLayer,
                skip_connection_n=0
            )
            self.obs_type = ve.SUMMED_OBS
        elif self.agent_name == 'a3c_agent_v1':
            self.model = gnn.GraphNNActorCritic(
                in_features=3,
                n_nodes=100,
                n_hidden_layers=3,
                layer_sizes=16,
                layer_class=gnn.FullyConnectedGNNLayer,
                skip_connection_n=0
            )
            self.obs_type = ve.SUMMED_OBS
        elif self.agent_name == 'a3c_agent_v2':
            self.model = gnn.GraphNNActorCritic(
                in_features=3,
                n_nodes=100,
                n_hidden_layers=4,
                layer_sizes=16,
                layer_class=gnn.FullyConnectedGNNLayer,
                skip_connection_n=1
            )
            self.obs_type = ve.SUMMED_OBS
        elif self.agent_name == 'a3c_agent_v3':
            self.model = gnn.GraphNNActorCritic(
                in_features=3,
                n_nodes=100,
                n_hidden_layers=4,
                layer_sizes=16,
                layer_class=gnn.FullyConnectedGNNLayer,
                skip_connection_n=1
            )
            self.obs_type = ve.SUMMED_OBS
        elif self.agent_name == 'a3c_agent_v4-162':
            self.model = gnn.GraphNNActorCritic(
                in_features=3,
                n_nodes=100,
                n_hidden_layers=4,
                layer_sizes=16,
                layer_class=gnn.FullyConnectedGNNLayer,
                skip_connection_n=1
            )
            self.obs_type = ve.SUMMED_OBS
        elif self.agent_name == 'a3c_agent_small_8_32-790':
            self.model = gnn.GraphNNActorCritic(
                in_features=3,
                n_nodes=100,
                n_hidden_layers=8,
                layer_sizes=32,
                layer_class=gnn.SmallFullyConnectedGNNLayer,
                skip_connection_n=1
            )
            self.obs_type = ve.SUMMED_OBS
        elif self.agent_name == 'awac_agent_small_8_64_32_1_norm_v1-230':
            self.model = gnn.GraphNNActorCritic(
                in_features=4,
                n_nodes=100,
                n_hidden_layers=8,
                layer_sizes=([64]*4) + ([32]*5),
                layer_class=gnn.SmallFullyConnectedGNNLayer,
                skip_connection_n=1,
                normalize=True,
                deprecated_version=True
            )
            self.obs_type = ve.SUMMED_OBS_WITH_TIMESTEP
        elif self.agent_name == 'a3c_agent_small_8_64_32_2_v1-11100':
            self.model = gnn.GraphNNActorCritic(
                in_features=244,
                n_nodes=100,
                n_hidden_layers=8,
                layer_sizes=[64]*4 + [32]*5,
                layer_class=gnn.SmallFullyConnectedGNNLayer,
                normalize=False,
                skip_connection_n=2
            )
            self.obs_type = ve.LAST_60_EVENTS_AND_SUMMED_OBS_RAVELLED
        elif self.agent_name == 'a3c_agent_small_8_64_32_2_v2-30':
            self.model = gnn.GraphNNActorCritic(
                in_features=244,
                n_nodes=100,
                n_hidden_layers=8,
                layer_sizes=[64] * 4 + [32] * 5,
                layer_class=gnn.SmallFullyConnectedGNNLayer,
                normalize=False,
                skip_connection_n=2
            )
            self.obs_type = ve.LAST_60_EVENTS_AND_SUMMED_OBS_RAVELLED
        else:
            raise ValueError(f'Unrecognized agent_name: {self.agent_name}')
        with open(f'saved_rl_models/{self.agent_name}.txt', 'r') as f:
            sd = pickle.loads(base64.b64decode(f.readline()[2:-1].encode()))['model_state_dict']
        self.model.load_state_dict(sd)
        self.model.to(device=device)
        self.model.eval()
        self.device = device
        if deterministic_policy:
            self.act_func = self.model.choose_best_action
        else:
            self.act_func = self.model.sample_action
        
    def __call__(self, states):
        states_device = states.device
        actions = self.act_func(states.to(device=self.device).unsqueeze(0)).squeeze(0)
        return actions.to(device=states_device)


class SavedRLAgentEnsemble(VectorizedAgent):
    def __init__(self, ensemble_name, device=torch.device('cuda'), weight_logits=False, deterministic_policy=False):
        super().__init__()
        self.ensemble_name = ensemble_name
        self.name = f'SavedRLAgentEnsemble: {self.ensemble_name}'
        if weight_logits:
            self.name += '_weight_logits'
        else:
            self.name += '_weight_probs'
        if deterministic_policy:
            self.name += '_deterministic'
        else:
            self.name += '_stochastic'
        models = []
        if self.ensemble_name == 'a3c_agent_small_8_32':
            for model_checkpoint in (410, 490, 540, 610, 790):
                model = gnn.GraphNNActorCritic(
                    in_features=3,
                    n_nodes=100,
                    n_hidden_layers=8,
                    layer_sizes=32,
                    layer_class=gnn.SmallFullyConnectedGNNLayer,
                    skip_connection_n=1
                )
                run_folder = 'a3c/small_8_32/'
                with open(f'runs/{run_folder}/{model_checkpoint}_cp.txt', 'r') as f:
                    serialized_string = f.readline()[2:-1].encode()
                state_dict_bytes = base64.b64decode(serialized_string)
                loaded_state_dicts = pickle.loads(state_dict_bytes)
                model.load_state_dict(loaded_state_dicts['model_state_dict'])
                models.append(model)
            self.obs_type = ve.SUMMED_OBS
        elif self.ensemble_name == 'awac_agent_small_8_64_32_1_norm':
            for model_checkpoint in (230, 355, 540, 600, 755):
                model = gnn.GraphNNActorCritic(
                    in_features=4,
                    n_nodes=100,
                    n_hidden_layers=8,
                    layer_sizes=([64]*4) + ([32]*5),
                    layer_class=gnn.SmallFullyConnectedGNNLayer,
                    skip_connection_n=1,
                    normalize=True,
                    deprecated_version=True
                )
                run_folder = 'awac/small_8_64_32_1_norm_v1'
                with open(f'runs/{run_folder}/{model_checkpoint}_cp.txt', 'r') as f:
                    serialized_string = f.readline()[2:-1].encode()
                state_dict_bytes = base64.b64decode(serialized_string)
                loaded_state_dicts = pickle.loads(state_dict_bytes)
                model.load_state_dict(loaded_state_dicts['model_state_dict'])
                models.append(model)
            self.obs_type = ve.SUMMED_OBS_WITH_TIMESTEP
        elif self.ensemble_name == 'a3c_agent_small_8_64_32_2':
            for model_checkpoint, run_folder in (
                    (6500, 'a3c/small_8_64_32_2_v1/'),
                    (11100, 'a3c/small_8_64_32_2_v1/'),
                    (12500, 'a3c/small_8_64_32_2_v1/'),
                    (17500, 'a3c/small_8_64_32_2_v1/'),
                    (19200, 'a3c/small_8_64_32_2_v1/'),
                    (30, 'a3c/small_8_64_32_2_v2/')
            ):
                model = gnn.GraphNNActorCritic(
                    in_features=244,
                    n_nodes=100,
                    n_hidden_layers=8,
                    layer_sizes=[64]*4 + [32]*5,
                    layer_class=gnn.SmallFullyConnectedGNNLayer,
                    normalize=False,
                    skip_connection_n=2
                )
                with open(f'runs/{run_folder}/{model_checkpoint}_cp.txt', 'r') as f:
                    serialized_string = f.readline()[2:-1].encode()
                state_dict_bytes = base64.b64decode(serialized_string)
                loaded_state_dicts = pickle.loads(state_dict_bytes)
                model.load_state_dict(loaded_state_dicts['model_state_dict'])
                models.append(model)
            self.obs_type = ve.LAST_60_EVENTS_AND_SUMMED_OBS_RAVELLED
        else:
            raise ValueError(f'Unrecognized ensemble_name: {self.ensemble_name}')
        self.ensemble_model = gnn.GraphNNActorCriticEnsemble(models, weight_logits=weight_logits)
        self.ensemble_model.to(device=device)
        self.ensemble_model.eval()
        self.device = device
        if deterministic_policy:
            self.act_func = self.ensemble_model.choose_best_action
        else:
            self.act_func = self.ensemble_model.sample_action

    def __call__(self, states):
        states_device = states.device
        actions = self.act_func(states.to(device=self.device).unsqueeze(0)).squeeze(0)
        return actions.to(device=states_device)


class SavedRLAgentMultiObsEnsemble(VectorizedAgent):
    def __init__(self, ensemble_names, device=torch.device('cuda'), weight_logits=False, deterministic_policy=False):
        super().__init__()
        self.ensemble_names = ensemble_names
        self.name = f'SavedRLAgentMultiObsEnsemble: {"__".join(self.ensemble_names)}'
        if weight_logits:
            self.name += '_weight_logits'
        else:
            self.name += '_weight_probs'
        if deterministic_policy:
            self.name += '_deterministic'
        else:
            self.name += '_stochastic'
        self.weight_logits = weight_logits
        self.deterministic_policy = deterministic_policy
        obs_types_to_models_dict = {}
        for en in ensemble_names:
            ensemble = SavedRLAgentEnsemble(
                en,
                device=device,
                weight_logits=weight_logits,
                deterministic_policy=deterministic_policy
            )
            obs_type = ensemble.obs_type
            if obs_type not in obs_types_to_models_dict.keys():
                obs_types_to_models_dict[obs_type] = []
            for model in ensemble.ensemble_model.models:
                obs_types_to_models_dict[obs_type].append(model)
        self.ensemble_model = gnn.GraphNNActorCriticMultiObsEnsemble(
            obs_types_to_models_dict,
            weight_logits=weight_logits
        )
        self.ensemble_model.to(device=device)
        self.ensemble_model.eval()
        self.device = device
        self.obs_type = self.ensemble_model.obs_types
        if deterministic_policy:
            self.act_func = self.ensemble_model.choose_best_action
        else:
            self.act_func = self.ensemble_model.sample_action

    def __call__(self, states):
        states_device = states[0].device
        actions = self.act_func([s.to(device=self.device).unsqueeze(0) for s in states]).squeeze(0)
        return actions.to(device=states_device)

    @property
    def ensemble_name(self):
        return '__'.join(self.ensemble_names)


class MultiObsFixedTimeEnsemble:
    def __init__(self, timesteps, agents, agents_obs_types):
        super().__init__()
        self.timesteps = timesteps
        self.agents = agents
        self.agents_obs_types = agents_obs_types
        self.name = f'MultiObsFixedTimeEnsemble: {"__".join([a.name for a in self.agents])}'
        assert len(self.timesteps) == len(self.agents)
        assert len(self.agents) == len(self.agents_obs_types)
        assert self.timesteps[0] == 0, 'self.timesteps describes the starting timestep for using each model'
        assert self.timesteps == sorted(self.timesteps), 'self.timesteps should be in ascending order'

        self.current_step = 0
        self.obs_type = []
        for ot in self.agents_obs_types:
            if type(ot) != str:
                for ot_ in ot:
                    if ot_ not in self.obs_type:
                        self.obs_type.append(ot_)
            else:
                if ot not in self.obs_type:
                    self.obs_type.append(ot)
        self.obs_types_idx_dict = {}
        for idx, ot in enumerate(self.obs_type):
            self.obs_types_idx_dict[ot] = idx
        self.reset()

    def __call__(self, states):
        current_agent_idx = None
        for idx, t in reversed(list(enumerate(self.timesteps))):
            if self.current_step >= t:
                current_agent_idx = idx
                break
        self.current_step += 1
        states = self.extract_agent_obs_type(current_agent_idx, states)
        return self.agents[current_agent_idx](states)

    def extract_agent_obs_type(self, agent_idx, states):
        target_obs_type = self.agents_obs_types[agent_idx]
        if type(target_obs_type) == str:
            idx = self.obs_types_idx_dict[target_obs_type]
            return states[idx]
        else:
            new_states = []
            for ot in target_obs_type:
                idx = self.obs_types_idx_dict[ot]
                new_states.append(states[idx])
            return new_states

    def reset(self):
        self.current_step = 0
        for a in self.agents:
            a.reset()


class RLModelWrapperAgent(VectorizedAgent):
    def __init__(self, model, obs_type, name=None, deterministic_policy=False):
        super().__init__()
        self.model = model
        if name is None:
            self.name = f'Wrapped RL model'
        else:
            self.name = f'Wrapped RL model: {name}'
        self.obs_type = obs_type
        if deterministic_policy:
            self.act_func = self.model.choose_best_action
        else:
            self.act_func = self.model.sample_action

    def __call__(self, states):
        if type(states) in (list, tuple):
            states = [s.unsqueeze(0) for s in states]
        else:
            states = states.unsqueeze(0)
        return self.act_func(states).squeeze(0)
