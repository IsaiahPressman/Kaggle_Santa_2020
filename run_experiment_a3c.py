import base64
from copy import copy
import numpy as np
from pathlib import Path
import pickle
import shutil
import torch
from torch import nn

# Adabelief optimizer: https://github.com/juntang-zhuang/Adabelief-Optimizer
from adabelief_pytorch import AdaBelief

# Custom imports
from a3c import A3CVectorized
import graph_nns as gnn
import vectorized_env as ve
import vectorized_agents as va

DEVICE = torch.device('cuda')

preprocessing_layer = gnn.GraphNNPreprocessingLayer(gnn.GraphNNResidualBase([
    gnn.SmallFullyConnectedGNNLayer(60, 4, 32),
    #gnn.SmallFullyConnectedGNNLayer(60, 32, 32)
], skip_connection_n=1), mean_out=True, ravel_out=False)

graph_nn_kwargs = dict(
    n_nodes=100,
    in_features=32,
    n_hidden_layers=2,
    #preprocessing_layer=preprocessing_layer, ###########################################################
    preprocessing_layer=preprocessing_layer,
    layer_sizes=[64],
    layer_class=gnn.FullyConnectedGNNLayer,
    normalize=False,
    skip_connection_n=1
)
model = gnn.GraphNNActorCritic(**graph_nn_kwargs)
"""
with open(f'runs/a3c/smallMean_8_32_1_v1/2150_cp.txt', 'r') as f:
    serialized_string = f.readline()[2:-1].encode()
state_dict_bytes = base64.b64decode(serialized_string)
loaded_state_dicts = pickle.loads(state_dict_bytes)
model.load_state_dict(loaded_state_dicts['model_state_dict'])
"""
model.to(device=DEVICE)
#optimizer = torch.optim.Adadelta(model.parameters())

optimizer = AdaBelief(model.parameters(),
                      lr=1e-4,
                      betas=(0.9, 0.999),
                      eps=1e-10,
                      #weight_decay=1e-5,
                      weight_decay=0.,
                      weight_decouple=False,
                      rectify=True,
                      fixed_decay=False,
                      amsgrad=False,
                      print_change_log=False)

env_kwargs = dict(
    env_device=DEVICE,
    out_device=DEVICE,
    normalize_reward=False,
    reward_type=ve.EVERY_STEP_EV_ZEROSUM,
    obs_type=ve.LAST_60_EVENTS_OBS
)
validation_env_kwargs_base = dict(
    n_envs=1000,
    env_device=DEVICE,
    out_device=DEVICE,
    obs_type=env_kwargs['obs_type']
)
validation_opponent_env_kwargs = [
    #dict(
    #    opponent=va.BasicThompsonSampling(),
    #    opponent_obs_type=ve.SUMMED_OBS
    #),
    dict(
        opponent=va.PullVegasSlotMachinesImproved(),
        opponent_obs_type=ve.SUMMED_OBS
    ),
    dict(
        opponent=va.SavedRLAgent('a3c_agent_small_8_32-790', device=DEVICE, deterministic_policy=True),
        opponent_obs_type=ve.SUMMED_OBS
    ),
    dict(
        opponent=va.SavedRLAgent('awac_agent_small_8_64_32_1_norm_v2-750', device=DEVICE, deterministic_policy=True),
        opponent_obs_type=ve.SUMMED_OBS_WITH_TIMESTEP
    )
]
validation_env_kwargs_dicts = []
for opponent_kwargs in validation_opponent_env_kwargs:
    validation_env_kwargs_dicts.append(copy(validation_env_kwargs_base))
    validation_env_kwargs_dicts[-1].update(opponent_kwargs)
rl_alg_kwargs = dict(
    clip_grads=None,
    play_against_past_selves=False,
    n_past_selves=4,
    checkpoint_freq=100,
    log_params_full=True,
    opp_posterior_decay=0.99,
    recurrent_model=(graph_nn_kwargs['layer_class'] == gnn.SmallRecurrentGNNLayer),
    run_separate_validation=True,
    validation_env_kwargs_dicts=validation_env_kwargs_dicts,
    deterministic_validation_policy=False
)
rl_train_kwargs = dict(
    batch_size=100,
    gamma=0.999
)

def model_constructor():
    return gnn.GraphNNActorCritic(**graph_nn_kwargs)

rl_agent_opp_kwargs = dict(
    device=DEVICE,
    deterministic_policy=False
)
initial_opponent_pool = [
    #va.RandomAgent(),
    va.BasicThompsonSampling(),
    va.PullVegasSlotMachinesImproved(),
    #va.SavedRLAgent('a3c_agent_v0', device=DEVICE),
    #va.SavedRLAgent('a3c_agent_v1', **rl_agent_opp_kwargs),
    #va.SavedRLAgent('a3c_agent_v2', **rl_agent_opp_kwargs),
    #va.SavedRLAgent('a3c_agent_v3', **rl_agent_opp_kwargs),
    #va.SavedRLAgent('a3c_agent_v4-162', **rl_agent_opp_kwargs),
    va.SavedRLAgent('a3c_agent_small_8_32-790', **rl_agent_opp_kwargs),
    va.SavedRLAgent('awac_agent_small_8_64_32_1_norm_v2-750', **rl_agent_opp_kwargs)
]
if type(optimizer) == torch.optim.SGD:
    optimizer_name = 'sgd_'
elif type(optimizer) == torch.optim.Adam:
    optimizer_name = 'adam_'
elif type(optimizer) == torch.optim.Adadelta:
    optimizer_name = 'adadelta_'
elif type(optimizer) == AdaBelief:
    optimizer_name = 'adabelief_'
if graph_nn_kwargs['layer_class'] == gnn.SmallFullyConnectedGNNLayer:
    layer_class = 'small_'
elif graph_nn_kwargs['layer_class'] == gnn.AttentionGNNLayer:
    layer_class = 'attention_'
elif graph_nn_kwargs['layer_class'] == gnn.SmallRecurrentGNNLayer:
    layer_class = 'smallRecurrent_'
elif graph_nn_kwargs['layer_class'] == gnn.MaxAndMeanGNNLayer:
    layer_class = 'maxAndMean_'
elif graph_nn_kwargs['layer_class'] == gnn.SmallMeanGNNLayer:
    layer_class = 'smallMean_'
else:
    layer_class = ''
with_preprocessing = 'preprocessing_' if graph_nn_kwargs['preprocessing_layer'] else ''
is_normalized = 'norm_' if graph_nn_kwargs['normalize'] else ''
folder_name = f"{optimizer_name}{with_preprocessing}{layer_class}{graph_nn_kwargs['n_hidden_layers']}_" \
              f"{'_'.join(np.flip(np.sort(np.unique(graph_nn_kwargs['layer_sizes']))).astype(str))}_" \
              f"{graph_nn_kwargs['skip_connection_n']}_{is_normalized}v2"
a3c_alg = A3CVectorized(model_constructor, optimizer, env_kwargs['obs_type'],
                        model=model, device=DEVICE,
                        exp_folder=Path(f'runs/a3c/{folder_name}'),
                        **rl_alg_kwargs)
this_script = Path(__file__).absolute()
shutil.copy(this_script, a3c_alg.exp_folder / f'_{this_script.name}')

env_kwargs['n_envs'] = 20
#env_kwargs['opponent'] = va.MultiAgent(initial_opponent_pool)
#env_kwargs['opponent'] = va.BasicThompsonSampling()
#env_kwargs['opponent_obs_type'] = ve.SUMMED_OBS
try:
    a3c_alg.train(n_episodes=100000, **rl_train_kwargs, **env_kwargs)
except KeyboardInterrupt:
    if a3c_alg.true_ep_num > a3c_alg.checkpoint_freq:
        print('KeyboardInterrupt: saving model')
        a3c_alg.save(finished=True)
