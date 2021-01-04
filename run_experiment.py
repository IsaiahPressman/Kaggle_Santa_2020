import base64
from importlib import reload
from jupyterthemes import jtplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import seaborn as sns
import shutil
from tensorboardX import SummaryWriter
import torch
from torch import distributions, nn
import torch.nn.functional as F
import time
import tqdm

# Custom imports
from a3c import A3CVectorized
import graph_nns as gnn
import vectorized_env as ve
import vectorized_agents as va

DEVICE = torch.device('cuda')
OBS_NORM = 100. / 1999.

use_rnn = False
def custom_layer_class_factory(*args, **kwargs):
    def recurrent_layer_class_factory(*args, **kwargs):
        return nn.LSTM(*args, num_layers=2, **kwargs)
    return gnn.SmallRecurrentGNNLayer(*args, recurrent_layer_class=recurrent_layer_class_factory, **kwargs)

graph_nn_kwargs = dict(
    in_features=3,
    n_nodes=100,
    n_hidden_layers=16,
    layer_sizes=32,
    layer_class=gnn.SmallRecurrentGNNLayer if use_rnn else gnn.SmallFullyConnectedGNNLayer,
    #layer_class=custom_layer_class_factory,
    skip_connection_n=1
)
model = gnn.GraphNNA3C(**graph_nn_kwargs)

with open(f'runs/small_16_32_v1/570_cp.txt', 'r') as f:
    serialized_string = f.readline()[2:-1].encode()
state_dict_bytes = base64.b64decode(serialized_string)
loaded_state_dicts = pickle.loads(state_dict_bytes)
model.load_state_dict(loaded_state_dicts['model_state_dict'])

model.to(device=DEVICE)
optimizer = torch.optim.Adam(model.parameters())

env_kwargs = dict(
    env_device=DEVICE,
    out_device=DEVICE,
    normalize_reward=False,
    reward_type=ve.EVERY_STEP_EV_ZEROSUM,
    obs_type=ve.LAST_STEP_OBS if use_rnn else ve.SUMMED_OBS,
    opponent_obs_type=ve.SUMMED_OBS
)
rl_alg_kwargs = dict(
    batch_size=30 if use_rnn else 30,
    gamma=0.99
)

def model_constructor():
    return gnn.GraphNNA3C(**graph_nn_kwargs)

rl_agent_opp_kwargs = dict(
    device=DEVICE,
    deterministic_policy=True
)
initial_opponent_pool = [
    #va.BasicThompsonSampling(OBS_NORM),
    va.PullVegasSlotMachines(OBS_NORM),
    #va.SavedRLAgent('a3c_agent_v0', device=DEVICE),
    va.SavedRLAgent('a3c_agent_v1', **rl_agent_opp_kwargs),
    va.SavedRLAgent('a3c_agent_v2', **rl_agent_opp_kwargs),
    va.SavedRLAgent('a3c_agent_v3', **rl_agent_opp_kwargs),
    va.SavedRLAgent('a3c_agent_v4-162', **rl_agent_opp_kwargs),
    va.SavedRLAgent('a3c_agent_small_8_32-790', **rl_agent_opp_kwargs),
]

folder_name = f"small_{graph_nn_kwargs['n_hidden_layers']}_{graph_nn_kwargs['layer_sizes']}_v2"
a3c_alg = A3CVectorized(model_constructor, optimizer, model=model, device=DEVICE,
                        exp_folder=Path(f'runs/a3c/{folder_name}'),
                        recurrent_model=use_rnn,
                        clip_grads=10.,
                        play_against_past_selves=True,
                        n_past_selves=4,
                        checkpoint_freq=10,
                        initial_opponent_pool=initial_opponent_pool,
                        opp_posterior_decay=0.95)

env_kwargs['n_envs'] = 200
try:
    a3c_alg.train(n_episodes=1000, **rl_alg_kwargs, **env_kwargs)
except KeyboardInterrupt:
    if a3c_alg.true_ep_num > a3c_alg.checkpoint_freq:
        print('KeyboardInterrupt: saving model')
        a3c_alg.save(finished=True)
