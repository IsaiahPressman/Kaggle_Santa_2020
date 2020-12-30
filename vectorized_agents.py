import base64
from copy import copy
import pickle
import time
import torch
from torch import distributions
import tqdm

import graph_nns as gnn
import vectorized_env as ve


def run_vectorized_vs(p1, p2, p1_name, p2_name, *env_args, **env_kwargs):
    env_kwargs = copy(env_kwargs)
    assert 'opponent' not in env_kwargs.keys(), 'Pass opponent as p2 arg, not as opponent kwarg'
    if 'obs_type' not in env_kwargs.keys():
        env_kwargs['obs_type'] = p1.obs_type
    if 'opponent_obs_type' not in env_kwargs.keys():
        env_kwargs['opponent_obs_type'] = p2.obs_type
    vs_env = ve.KaggleMABEnvTorchVectorized(*env_args, opponent=p2, **env_kwargs)
    s, _, _, _ = vs_env.reset()
    for i in tqdm.trange(vs_env.n_steps):
        s, _, _, _ = vs_env.step(p1(s))
    p1_scores, p2_scores = vs_env.player_rewards_sums.sum(-1).chunk(2, dim=1)
    print(f'{p1_name} -vs- {p2_name}')
    print(f'Mean scores: {p1_scores.mean():.2f} - {p2_scores.mean():.2f}')
    print(f'Match score: {torch.sum(p1_scores > p2_scores)} - '
          f'{torch.sum(p1_scores == p2_scores)} - '
          f'{torch.sum(p1_scores < p2_scores)} '
          f'({torch.sum(p1_scores > p2_scores) * 100. / vs_env.n_envs:.1f}% - '
          f'{torch.sum(p1_scores == p2_scores) * 100. / vs_env.n_envs:.1f}% - '
          f'{torch.sum(p1_scores < p2_scores) * 100. / vs_env.n_envs:.1f}%)')
    time.sleep(0.5)


class VectorizedAgent:
    def __init__(self):
        self.name = None
        self.obs_type = None

    def __call__(self, states):
        return None


class AlwaysFirstAgent(VectorizedAgent):
    def __init__(self):
        super().__init__()
        self.name = 'AlwaysFirstAgent'
        self.obs_type = ve.SUMMED_OBS
        
    def __call__(self, states):
        return torch.zeros(size=states.shape[:-2], dtype=torch.long, device=states.device)

    
class BasicThompsonSampling(VectorizedAgent):
    def __init__(self, obs_norm, n_bandits=100):
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
        self.obs_type = ve.SUMMED_OBS
    
    def __call__(self, states):
        states_chunked = states.chunk(len(self.agents), dim=self.envs_dim)
        if len(states_chunked) != len(self.agents):
            raise ValueError(f'There were only {states.shape[self.envs_dim]} envs for {len(self.agents)} agents, '
                             f'from states tensor of shape {states.shape}')
        else:
            return torch.cat([a(s) for a, s in zip(self.agents, states_chunked)], dim=self.envs_dim)


class PullVegasSlotMachines(VectorizedAgent):
    def __init__(self, obs_norm, n_bandits=100):
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
            ev = (wins - losses + opp_pulls - (opp_pulls>0)*1.5) / (wins + losses + opp_pulls)
            actions = ev.squeeze(-1).argmax(dim=-1)
        return actions

    
class RandomAgent(VectorizedAgent):
    def __init__(self, n_bandits=100):
        super().__init__()
        self.n_bandits = n_bandits
        self.name = 'RandomAgent'
        self.obs_type = ve.SUMMED_OBS
        
    def __call__(self, states):
        assert states.shape[-2:] == (self.n_bandits, 3)
        return torch.randint(self.n_bandits, size=states.shape[:-2], device=states.device)


# Saved RL agents
class SavedRLAgent(VectorizedAgent):
    def __init__(self, agent_name, device=torch.device('cuda'), deterministic_policy=False):
        super().__init__()
        self.name = f'SavedRLAgent: {agent_name}'
        if agent_name == 'a3c_agent_v0':
            self.model = gnn.GraphNNA3C(
                in_features=3,
                n_nodes=100,
                n_hidden_layers=1,
                layer_sizes=16,
                layer_class=gnn.FullyConnectedGNNLayer,
                skip_connection_n=0
            )
            ss_filename = 'rl_agents/ss_a3c_agent_v0.txt'
            self.obs_type = ve.SUMMED_OBS
        elif agent_name == 'a3c_agent_v1':
            self.model = gnn.GraphNNA3C(
                in_features=3,
                n_nodes=100,
                n_hidden_layers=3,
                layer_sizes=16,
                layer_class=gnn.FullyConnectedGNNLayer,
                skip_connection_n=0
            )
            ss_filename = 'rl_agents/ss_a3c_agent_v1.txt'
            self.obs_type = ve.SUMMED_OBS
        elif agent_name == 'a3c_agent_v2':
            self.model = gnn.GraphNNA3C(
                in_features=3,
                n_nodes=100,
                n_hidden_layers=4,
                layer_sizes=16,
                layer_class=gnn.FullyConnectedGNNLayer,
                skip_connection_n=1
            )
            ss_filename = 'rl_agents/ss_a3c_agent_v2.txt'
            self.obs_type = ve.SUMMED_OBS
        elif agent_name == 'a3c_agent_v3':
            self.model = gnn.GraphNNA3C(
                in_features=3,
                n_nodes=100,
                n_hidden_layers=4,
                layer_sizes=16,
                layer_class=gnn.FullyConnectedGNNLayer,
                skip_connection_n=1
            )
            ss_filename = 'rl_agents/ss_a3c_agent_v3.txt'
            self.obs_type = ve.SUMMED_OBS
        elif agent_name == 'a3c_agent_v4-162':
            self.model = gnn.GraphNNA3C(
                in_features=3,
                n_nodes=100,
                n_hidden_layers=4,
                layer_sizes=16,
                layer_class=gnn.FullyConnectedGNNLayer,
                skip_connection_n=1
            )
            ss_filename = 'runs/v4/cp_162.txt'
            self.obs_type = ve.SUMMED_OBS
        else:
            raise ValueError(f'Unrecognized agent_name: {agent_name}')
        with open(ss_filename, 'r') as f:
            sd = pickle.loads(base64.b64decode(f.readline()[2:-1].encode()))['model_state_dict']
        self.model.load_state_dict(sd)
        self.model.to(device=device)
        self.model.eval()
        if deterministic_policy:
            self.act_func = self.model.choose_best_action
        else:
            self.act_func = self.model.sample_action
        
    def __call__(self, states):
        return self.act_func(states.unsqueeze(0)).squeeze(0)


class RLModelWrapperAgent(VectorizedAgent):
    def __init__(self, model, obs_type, deterministic_policy=False):
        super().__init__()
        self.model = model
        self.name = f'Wrapped RL model: {self.model}'
        self.obs_type = obs_type
        if deterministic_policy:
            self.act_func = self.model.choose_best_action
        else:
            self.act_func = self.model.sample_action

    def __call__(self, states):
        return self.act_func(states.unsqueeze(0)).squeeze(0)
