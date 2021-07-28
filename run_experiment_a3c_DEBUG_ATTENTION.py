from apex import amp
import base64
import numpy as np
from pathlib import Path
import pickle
import shutil
import torch

# Custom imports
from a3c import A3CVectorized
import graph_nns as gnn
import vectorized_env as ve
import vectorized_agents as va

DEVICE = torch.device('cpu')
OBS_NORM = 100. / 1999.

graph_nn_kwargs = dict(
    in_features=4,
    n_nodes=100,
    n_hidden_layers=4,
    preprocessing_layer=False,
    layer_sizes=[32],
    layer_class=gnn.AttentionGNNLayer,
    normalize=False,
    skip_connection_n=1
)
model = gnn.GraphNNActorCritic(**graph_nn_kwargs)
"""
with open(f'runs/a3c/small_8_32_1_norm_v1/final_1230_cp.txt', 'r') as f:
    serialized_string = f.readline()[2:-1].encode()
state_dict_bytes = base64.b64decode(serialized_string)
loaded_state_dicts = pickle.loads(state_dict_bytes)
model.load_state_dict(loaded_state_dicts['model_state_dict'])
"""
model.to(device=DEVICE)
optimizer = torch.optim.Adam(model.parameters(),
                             #weight_decay=5e-6
                             )


env_kwargs = dict(
    env_device=DEVICE,
    out_device=DEVICE,
    normalize_reward=False,
    reward_type=ve.EVERY_STEP_EV_ZEROSUM,
    obs_type=ve.SUMMED_OBS_WITH_TIMESTEP
)
rl_alg_kwargs = dict(
    batch_size=30,
    gamma=0.99
)
rl_train_kwargs = dict(
    clip_grads=10.,
    play_against_past_selves=True,
    n_past_selves=4,
    checkpoint_freq=20,
    log_params_full=False,
    opp_posterior_decay=0.99,
    use_amp=False
)

if rl_train_kwargs['use_amp']:
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

def model_constructor():
    return gnn.GraphNNActorCritic(**graph_nn_kwargs)

rl_agent_opp_kwargs = dict(
    device=DEVICE,
    deterministic_policy=False
)
initial_opponent_pool = [
    #va.RandomAgent(),
    va.BasicThompsonSampling(OBS_NORM),
    va.PullVegasSlotMachinesImproved(OBS_NORM),
    #va.SavedRLAgent('a3c_agent_v0', device=DEVICE),
    va.SavedRLAgent('a3c_agent_v1', **rl_agent_opp_kwargs),
    va.SavedRLAgent('a3c_agent_v2', **rl_agent_opp_kwargs),
    #va.SavedRLAgent('a3c_agent_v3', **rl_agent_opp_kwargs),
    #va.SavedRLAgent('a3c_agent_v4-162', **rl_agent_opp_kwargs),
    va.SavedRLAgent('a3c_agent_small_8_32-790', **rl_agent_opp_kwargs),
]

is_small = 'small_' if graph_nn_kwargs['layer_class'] == gnn.SmallFullyConnectedGNNLayer else ''
with_preprocessing = 'preprocessing_' if graph_nn_kwargs['preprocessing_layer'] else ''
is_normalized = 'norm_' if graph_nn_kwargs['normalize'] else ''
folder_name = f"{with_preprocessing}{is_small}{graph_nn_kwargs['n_hidden_layers']}_" \
              f"{'_'.join(np.flip(np.sort(np.unique(graph_nn_kwargs['layer_sizes']))).astype(str))}_" \
              f"{graph_nn_kwargs['skip_connection_n']}_{is_normalized}v2"
a3c_alg = A3CVectorized(model_constructor, optimizer, env_kwargs['obs_type'],
                        model=model, device=DEVICE,
                        #exp_folder=Path(f'runs/a3c/{folder_name}'),
                        initial_opponent_pool=initial_opponent_pool,
                        **rl_train_kwargs)
this_script = Path(__file__).absolute()
shutil.copy(this_script, a3c_alg.exp_folder / f'_{this_script.name}')

env_kwargs['n_envs'] = 100
#env_kwargs['opponent'] = va.BasicThompsonSampling(OBS_NORM)
#env_kwargs['opp_obs_type'] = ve.SUMMED_OBS
try:
    a3c_alg.train(n_episodes=10000, **rl_alg_kwargs, **env_kwargs)
except KeyboardInterrupt:
    if a3c_alg.true_ep_num > a3c_alg.checkpoint_freq:
        print('KeyboardInterrupt: saving model')
        a3c_alg.save(finished=True)
