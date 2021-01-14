import base64
from pathlib import Path
import pickle
import torch
from torch import nn

# Custom imports
from a3c import A3CVectorized
import graph_nns as gnn
import vectorized_env as ve
import vectorized_agents as va

DEVICE = torch.device('cuda')
OBS_NORM = 100. / 1999.

graph_nn_kwargs = dict(
    in_features=3,
    n_nodes=100,
    n_hidden_layers=2,
    layer_sizes=16,
    layer_class=gnn.SqueezeExictationGNNLayer,#gnn.FullyConnectedGNNLayer,
    normalize=False,
    skip_connection_n=1 
)
model = gnn.GraphNNActorCritic(**graph_nn_kwargs)
"""
with open(f'runs/small_16_32_v1/570_cp.txt', 'r') as f:
    serialized_string = f.readline()[2:-1].encode()
state_dict_bytes = base64.b64decode(serialized_string)
loaded_state_dicts = pickle.loads(state_dict_bytes)
model.load_state_dict(loaded_state_dicts['model_state_dict'])
"""
model.to(device=DEVICE)
optimizer = torch.optim.Adam(model.parameters())

env_kwargs = dict(
    env_device=DEVICE,
    out_device=DEVICE,
    normalize_reward=False,
    reward_type=ve.EVERY_STEP_EV_ZEROSUM,
    obs_type=ve.SUMMED_OBS,
    #opponent_obs_type=ve.SUMMED_OBS
)
rl_alg_kwargs = dict(
    batch_size=30,
    gamma=0.99
)

def model_constructor():
    return gnn.GraphNNActorCritic(**graph_nn_kwargs)

rl_agent_opp_kwargs = dict(
    device=DEVICE,
    deterministic_policy=True
)
initial_opponent_pool = [
    va.BasicThompsonSampling(OBS_NORM),
    va.PullVegasSlotMachines(OBS_NORM),
    #va.SavedRLAgent('a3c_agent_v0', device=DEVICE),
    #va.SavedRLAgent('a3c_agent_v1', **rl_agent_opp_kwargs),
    #va.SavedRLAgent('a3c_agent_v2', **rl_agent_opp_kwargs),
    #va.SavedRLAgent('a3c_agent_v3', **rl_agent_opp_kwargs),
    #va.SavedRLAgent('a3c_agent_v4-162', **rl_agent_opp_kwargs),
    #va.SavedRLAgent('a3c_agent_small_8_32-790', **rl_agent_opp_kwargs),
]

#folder_name = f"small_{graph_nn_kwargs['n_hidden_layers']}_{graph_nn_kwargs['layer_sizes']}_v2"
folder_name = 'SE3'
a3c_alg = A3CVectorized(model_constructor, optimizer, env_kwargs['obs_type'], model=model, device=DEVICE,
                        exp_folder=Path(f'runs/a3c/{folder_name}'),
                        recurrent_model=False,
                        clip_grads=10.,
                        play_against_past_selves=True,
                        n_past_selves=4,
                        checkpoint_freq=10,
                        initial_opponent_pool=initial_opponent_pool,
                        opp_posterior_decay=0.95)

env_kwargs['n_envs'] = 150
try:
    with torch.autograd.set_detect_anomaly(True):
        a3c_alg.train(n_episodes=1000, **rl_alg_kwargs, **env_kwargs)
except KeyboardInterrupt:
    if a3c_alg.true_ep_num > a3c_alg.checkpoint_freq:
        print('KeyboardInterrupt: saving model')
        a3c_alg.save(finished=True)
