import base64
import copy
import numpy as np
from pathlib import Path
import pickle
import shutil
import torch

# Custom imports
from nfsp import NFSPVectorized, CircularReplayBuffer, ReservoirReplayBuffer
import graph_nns as gnn
from replays_to_database import load_s_a_r_d_s
import vectorized_env as ve
import vectorized_agents as va


DEVICE = torch.device('cuda')
OBS_NORM = 100. / 1999.

replay_s_a_r_d_s = load_s_a_r_d_s(
    'episode_scraping/latest_250_replays_database_SUMMED_OBS_WITH_TIMESTEP/'
)

graph_nn_policy_kwargs = dict(
    in_features=4,
    n_nodes=100,
    n_hidden_layers=4,
    layer_sizes=[20],
    layer_class=gnn.FullyConnectedGNNLayer,
    #preprocessing_layer=True,
    skip_connection_n=1,
    normalize=True
)
graph_nn_q_kwargs = copy.copy(graph_nn_policy_kwargs)

policy_opt_kwargs = dict(
    # weight_decay=5e-6
)
q_opt_kwargs = copy.copy(policy_opt_kwargs)

env_kwargs = dict(
    env_device=DEVICE,
    out_device=DEVICE,
    normalize_reward=False,
    reward_type=ve.EVERY_STEP_EV_ZEROSUM,
    obs_type=ve.SUMMED_OBS_WITH_TIMESTEP,
)

rl_alg_kwargs = dict(
    eta=0.1,
    starting_epsilon=0.08,
    gamma=0.99,
    checkpoint_freq=10,
    log_params_full=False
)
rl_train_kwargs = dict(
    batch_size=1024,
    n_expl_steps_per_epoch=1999,
    n_train_batches_per_epoch=1999,
    n_epochs_q_target_update=1
)
m_rl_circular_kwargs = dict(
    s_shape=(graph_nn_policy_kwargs['n_nodes'], graph_nn_policy_kwargs['in_features']),
    max_len=4e5,
    starting_s_a_r_d_s=None
    #starting_s_a_r_d_s=replay_s_a_r_d_s
)
m_sl_reservoir_kwargs = dict(
    s_shape=(graph_nn_policy_kwargs['n_nodes'], graph_nn_policy_kwargs['in_features']),
    max_len=2e6,
)

validation_env_kwargs_base = dict(
    n_envs=1000,
    env_device=DEVICE,
    out_device=DEVICE,
    obs_type=env_kwargs['obs_type']
)
validation_opponent_env_kwargs = [
    dict(
        opponent=va.BasicThompsonSampling(OBS_NORM),
        opponent_obs_type=ve.SUMMED_OBS
    ),
    dict(
        opponent=va.PullVegasSlotMachines(OBS_NORM),
        opponent_obs_type=ve.SUMMED_OBS
    ),
    dict(
        opponent=va.SavedRLAgent('a3c_agent_small_8_32-790', device=DEVICE, deterministic_policy=True),
        opponent_obs_type=ve.SUMMED_OBS
    ),
    #dict(
    #    opponent=va.SavedRLAgent('awac_agent_4_20_1_norm_v1-215', device=DEVICE, deterministic_policy=True),
    #    opponent_obs_type=ve.SUMMED_OBS_WITH_TIMESTEP
    #)
]

policy_models = [gnn.GraphNNPolicy(**graph_nn_policy_kwargs), gnn.GraphNNPolicy(**graph_nn_policy_kwargs)]
q_models = [gnn.GraphNNQ(**graph_nn_q_kwargs), gnn.GraphNNQ(**graph_nn_q_kwargs)]
q_target_models = [gnn.GraphNNQ(**graph_nn_q_kwargs), gnn.GraphNNQ(**graph_nn_q_kwargs)]
policy_opts = [torch.optim.Adam(model.parameters(), **policy_opt_kwargs) for model in policy_models]
q_opts = [torch.optim.Adam(model.parameters(), **q_opt_kwargs) for model in q_models]
"""
with open(f'runs/awac/small_8_32_1_norm_v1/585_cp.txt', 'r') as f:
    serialized_string = f.readline()[2:-1].encode()
state_dict_bytes = base64.b64decode(serialized_string)
loaded_state_dicts = pickle.loads(state_dict_bytes)
model.load_state_dict(loaded_state_dicts['model_state_dict'])
"""
for model in policy_models + q_models + q_target_models:
    model.to(device=DEVICE)
# Conserves memory when replay_s_a_r_d_s is not being used
del replay_s_a_r_d_s

validation_env_kwargs_dicts = []
for opponent_kwargs in validation_opponent_env_kwargs:
    validation_env_kwargs_dicts.append(copy.copy(validation_env_kwargs_base))
    validation_env_kwargs_dicts[-1].update(opponent_kwargs)

is_small = 'small_' if graph_nn_policy_kwargs['layer_class'] == gnn.SmallFullyConnectedGNNLayer else ''
#with_preprocessing = 'preprocessing_' if graph_nn_policy_kwargs['preprocessing_layer'] else ''
with_preprocessing = ''
is_normalized = 'norm_' if graph_nn_policy_kwargs['normalize'] else ''
folder_name = f"{with_preprocessing}{is_small}{graph_nn_policy_kwargs['n_hidden_layers']}_" \
              f"{'_'.join(np.flip(np.sort(graph_nn_policy_kwargs['layer_sizes'])).astype(str))}_" \
              f"{graph_nn_policy_kwargs['skip_connection_n']}_{is_normalized}v1"
awac_alg = NFSPVectorized(policy_models, q_models, q_target_models, m_rl_circular_kwargs, m_sl_reservoir_kwargs,
                          policy_opts=policy_opts, q_opts=q_opts,
                          validation_env_kwargs_dicts=validation_env_kwargs_dicts,
                          device=DEVICE,
                          #exp_folder=Path(f'runs/nfsp/{folder_name}'),
                          **rl_alg_kwargs)
this_script = Path(__file__).absolute()
shutil.copy(this_script, awac_alg.exp_folder / f'_{this_script.name}')

env_kwargs['n_envs'] = 16
try:
    awac_alg.train(
        ve.KaggleMABEnvTorchVectorized(**env_kwargs),
        n_epochs=1000,
        **rl_train_kwargs
    )
except KeyboardInterrupt:
    if awac_alg.episode_counter > awac_alg.checkpoint_freq:
        print('KeyboardInterrupt: saving model')
        awac_alg.save(finished=True)
