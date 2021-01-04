import base64
from pathlib import Path
import pickle
import torch

# Custom imports
from awac import AWACVectorized, ReplayBuffer
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
    layer_class=gnn.FullyConnectedGNNLayer,
    skip_connection_n=1
)
model = gnn.GraphNNA3C(**graph_nn_kwargs)

"""
with open(f'runs/small_16_32_v1/570_cp.txt', 'r') as f:
    serialized_string = f.readline()[2:-1].encode()
state_dict_bytes = base64.b64decode(serialized_string)
loaded_state_dicts = pickle.loads(state_dict_bytes)
model.load_state_dict(loaded_state_dicts['model_state_dict'])
"""

model.to(device=DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=3e-4)

env_kwargs = dict(
    env_device=DEVICE,
    out_device=DEVICE,
    normalize_reward=False,
    reward_type=ve.EVERY_STEP_EV_ZEROSUM,
    obs_type=ve.SUMMED_OBS,
)
rl_alg_kwargs = dict(
    batch_size=1024,
    n_pretrain_batches=0,
    train_batches_per_timestep=1,
    gamma=0.99,
    lagrange_multiplier=1.
)
replay_buffer = ReplayBuffer(
    s_shape=(100, 3),
    max_len=1e6,
    starting_s_a_r_d_s=None,
)


folder_name = f"small_{graph_nn_kwargs['n_hidden_layers']}_{graph_nn_kwargs['layer_sizes']}"
awac_alg = AWACVectorized(model, optimizer, replay_buffer,
                          device=DEVICE,
                          #exp_folder=Path(f'runs/awac/{folder_name}'),
                          clip_grads=10.,
                          checkpoint_freq=10)

env_kwargs['n_envs'] = 512
env_kwargs['opponent'] = va.BasicThompsonSampling(OBS_NORM)
try:
    awac_alg.train(n_steps=int(1e6), **rl_alg_kwargs, **env_kwargs)
except KeyboardInterrupt:
    if awac_alg.true_ep_num > awac_alg.checkpoint_freq:
        print('KeyboardInterrupt: saving model')
        awac_alg.save(finished=True)
