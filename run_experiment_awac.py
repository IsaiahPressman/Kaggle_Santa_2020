import base64
import copy
import numpy as np
from pathlib import Path
import pickle
import shutil
import torch

# Custom imports
from awac import AWACVectorized, ReplayBuffer
import graph_nns as gnn
from replays_to_database import load_s_a_r_d_s
import vectorized_env as ve
import vectorized_agents as va

DEVICE = torch.device('cuda')
OBS_NORM = 100. / 1999.

graph_nn_kwargs = dict(
    in_features=4,
    n_nodes=100,
    n_hidden_layers=8,
    layer_sizes=32,
    layer_class=gnn.SmallFullyConnectedGNNLayer,
    preprocessing_layer=False,
    skip_connection_n=1,
    normalize=True
)
model = gnn.GraphNNActorCritic(**graph_nn_kwargs)

with open(f'runs/awac/small_8_32_1_norm_v2/105_cp.txt', 'r') as f:
    serialized_string = f.readline()[2:-1].encode()
state_dict_bytes = base64.b64decode(serialized_string)
loaded_state_dicts = pickle.loads(state_dict_bytes)
model.load_state_dict(loaded_state_dicts['model_state_dict'])

model.to(device=DEVICE)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-6)

env_kwargs = dict(
    env_device=DEVICE,
    out_device=DEVICE,
    normalize_reward=False,
    reward_type=ve.EVERY_STEP_EV_ZEROSUM,
    obs_type=ve.SUMMED_OBS_WITH_TIMESTEP,
)
rl_train_kwargs = dict(
    batch_size=1024,
    #n_pretrain_batches=10000,
    n_pretrain_batches=0,
    n_steps_per_epoch=1999,
    n_train_batches_per_epoch=None,
    gamma=0.99,
    lagrange_multiplier=1.
)
replay_s_a_r_d_s = load_s_a_r_d_s(
    '/home/isaiah/GitHub/Kaggle/Santa_2020/episode_scraping/latest_250_replays_database_SUMMED_OBS_WITH_TIMESTEP/'
)
replay_buffer = ReplayBuffer(
    s_shape=(graph_nn_kwargs['n_nodes'], graph_nn_kwargs['in_features']),
    max_len=1e6,
    starting_s_a_r_d_s=None,
    #starting_s_a_r_d_s=replay_s_a_r_d_s
)
# Conserve memory
del replay_s_a_r_d_s
validation_env_kwargs_base = dict(
    n_envs=500,
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
        opponent=va.PullVegasSlotMachinesImproved(),
        opponent_obs_type=ve.LAST_STEP_OBS
    ),
    dict(
        opponent=va.SavedRLAgent('a3c_agent_small_8_32-790', device=DEVICE, deterministic_policy=True),
        opponent_obs_type=ve.SUMMED_OBS
    ),
    dict(
        opponent=va.SavedRLAgent('awac_agent_4_20_1_norm_v1-215', device=DEVICE, deterministic_policy=True),
        opponent_obs_type=ve.SUMMED_OBS_WITH_TIMESTEP
    )
]
validation_env_kwargs_dicts = []
for opponent_kwargs in validation_opponent_env_kwargs:
    validation_env_kwargs_dicts.append(copy.copy(validation_env_kwargs_base))
    validation_env_kwargs_dicts[-1].update(opponent_kwargs)

is_small = 'small_' if graph_nn_kwargs['layer_class'] == gnn.SmallFullyConnectedGNNLayer else ''
with_preprocessing = 'preprocessing_' if graph_nn_kwargs['preprocessing_layer'] else ''
is_normalized = 'norm_' if graph_nn_kwargs['normalize'] else ''
folder_name = f"{with_preprocessing}{is_small}{graph_nn_kwargs['n_hidden_layers']}_" \
              f"{'_'.join(np.flip(np.sort(np.unique(graph_nn_kwargs['layer_sizes']))).astype(str))}_" \
              f"{graph_nn_kwargs['skip_connection_n']}_{is_normalized}v3"
awac_alg = AWACVectorized(model, optimizer, replay_buffer,
                          validation_env_kwargs_dicts=validation_env_kwargs_dicts,
                          deterministic_validation_policy=True,
                          device=DEVICE,
                          exp_folder=Path(f'runs/awac/{folder_name}'),
                          clip_grads=10.,
                          checkpoint_freq=5)
this_script = Path(__file__).absolute()
shutil.copy(this_script, awac_alg.exp_folder / f'_{this_script.name}')

#env_kwargs['n_envs'] = 64
env_kwargs['n_envs'] = 128
env_kwargs['opponent'] = va.PullVegasSlotMachinesImproved()
env_kwargs['opponent_obs_type'] = ve.LAST_STEP_OBS
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
