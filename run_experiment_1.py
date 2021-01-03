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

use_rnn = True
def custom_layer_class_factory(*args, **kwargs):
    def recurrent_layer_class_factory(*args, **kwargs):
        return nn.LSTM(*args, num_layers=2, **kwargs)
    return gnn.SmallRecurrentGNNLayer(*args, recurrent_layer_class=recurrent_layer_class_factory, **kwargs)

graph_nn_kwargs = dict(
    in_features=3,
    n_nodes=100,
    n_hidden_layers=3,
    layer_sizes=24,
    #layer_class=gnn.SmallRecurrentGNNLayer if use_rnn else gnn.SmallFullyConnectedGNNLayer,
    layer_class=custom_layer_class_factory,
    skip_connection_n=1
)
model = gnn.GraphNNA3C(**graph_nn_kwargs)
model.to(device=DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

env_kwargs = dict(
    n_envs=512,
    env_device=DEVICE,
    out_device=DEVICE,
    normalize_reward=False,
    reward_type=ve.EVERY_STEP_EV,
    obs_type=ve.LAST_STEP_OBS if use_rnn else ve.SUMMED_OBS,
    opponent_obs_type=ve.SUMMED_OBS
)
rl_alg_kwargs = dict(
    batch_size=30 if use_rnn else 30,
    gamma=0.99
)

def model_constructor():
    return gnn.GraphNNA3C(**graph_nn_kwargs)

# initial_opponent_pool = []
initial_opponent_pool = [
    #va.BasicThompsonSampling(OBS_NORM),
    #va.PullVegasSlotMachines(OBS_NORM),
    #va.SavedRLAgent('a3c_agent_v0', device=DEVICE),
    #va.SavedRLAgent('a3c_agent_v1', device=DEVICE),
    #va.SavedRLAgent('a3c_agent_v2', device=DEVICE),
    va.SavedRLAgent('a3c_agent_v3', device=DEVICE),
]

a3c_alg = A3CVectorized(model_constructor, optimizer, model=model, device=DEVICE,
                        exp_folder=Path('runs/LSTM_3_2_layer_v2'),
                        recurrent_model=use_rnn,
                        clip_grads=10.,
                        play_against_past_selves=False,
                        n_past_selves=4,
                        checkpoint_freq=10,
                        initial_opponent_pool=initial_opponent_pool,
                        opp_posterior_decay=0.95)

env_kwargs['n_envs'] = 400
env_kwargs['opponent'] = va.SavedRLAgent('a3c_agent_v0', device=DEVICE)
a3c_alg.train(n_episodes=1000, **rl_alg_kwargs, **env_kwargs)