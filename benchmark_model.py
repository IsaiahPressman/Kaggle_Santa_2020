import base64
import os
import pickle

import torch

# Custom imports
import graph_nns as gnn
import vectorized_agents as va
import vectorized_env as ve

DEVICE = torch.device('cuda')
RUN_FOLDER = 'a3c/TEMP'
MODEL_CHECKPOINT = '140'
AGENT_OBS_TYPE = ve.SUMMED_OBS
USE_DETERMINISTIC_POLICY = True

if DEVICE == torch.device('cpu'):
    os.environ['OMP_NUM_THREADS'] = '4'

graph_nn_kwargs = dict(
    in_features=3,
    n_nodes=100,
    n_hidden_layers=2,
    preprocessing_layer=False,
    layer_sizes=[32],
    layer_class=gnn.SmallFullyConnectedGNNLayer,
    normalize=False,
    skip_connection_n=1
)

model = gnn.GraphNNActorCritic(
    **graph_nn_kwargs
)

with open(f'runs/{RUN_FOLDER}/{MODEL_CHECKPOINT}_cp.txt', 'r') as f:
    serialized_string = f.readline()[2:-1].encode()
state_dict_bytes = base64.b64decode(serialized_string)
loaded_state_dicts = pickle.loads(state_dict_bytes)
model.load_state_dict(loaded_state_dicts['model_state_dict'])
model.to(device=DEVICE)
model.train()
model_wrapped = va.RLModelWrapperAgent(
    model,
    AGENT_OBS_TYPE,
    name=f'new_model checkpoint #{MODEL_CHECKPOINT}',
    deterministic_policy=USE_DETERMINISTIC_POLICY
)

benchmark_env_kwargs = dict(
    n_envs=1000 if DEVICE == torch.device('cuda') else 50,
    env_device=DEVICE,
    out_device=DEVICE
)
important_env_kwargs = dict(**benchmark_env_kwargs)
important_env_kwargs['n_envs'] = benchmark_env_kwargs['n_envs'] * 2

rl_agent_opp_kwargs = dict(
    device=DEVICE,
    deterministic_policy=True
)

# Benchmark the model against various hand-crafted algorithms and previously trained RL agents
va.run_vectorized_vs(model_wrapped, va.BasicThompsonSampling(), **benchmark_env_kwargs)
va.run_vectorized_vs(model_wrapped, va.PullVegasSlotMachines(), **benchmark_env_kwargs)
va.run_vectorized_vs(model_wrapped, va.SavedRLAgent('a3c_agent_v0', **rl_agent_opp_kwargs), **benchmark_env_kwargs)
va.run_vectorized_vs(model_wrapped, va.SavedRLAgent('a3c_agent_v1', **rl_agent_opp_kwargs), **benchmark_env_kwargs)
va.run_vectorized_vs(model_wrapped, va.SavedRLAgent('a3c_agent_v2', **rl_agent_opp_kwargs), **benchmark_env_kwargs)
va.run_vectorized_vs(model_wrapped, va.SavedRLAgent('a3c_agent_v3', **rl_agent_opp_kwargs), **benchmark_env_kwargs)
va.run_vectorized_vs(model_wrapped, va.SavedRLAgent('a3c_agent_v4-162', **rl_agent_opp_kwargs), **benchmark_env_kwargs)
va.run_vectorized_vs(model_wrapped, va.PullVegasSlotMachinesImproved(), **important_env_kwargs)
va.run_vectorized_vs(model_wrapped, va.SavedRLAgent('a3c_agent_small_8_32-790', **rl_agent_opp_kwargs),
                     **important_env_kwargs)
