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

all_ensemble_names = ['a3c_agent_small_8_32', 'awac_agent_small_8_64_32_1_norm', 'a3c_agent_small_8_64_32_2']
prebuilt_ensemble = va.SavedRLAgentMultiObsEnsemble(all_ensemble_names)
obs_types_to_models_dict = {}
n_models = 0
for obs_type, em in zip(
        prebuilt_ensemble.ensemble_model.obs_types,
        prebuilt_ensemble.ensemble_model.ensemble_model
):
    obs_types_to_models_dict[obs_type] = em.models
    n_models += len(em.models)

ensemble_name = 'a3c_agent_small_8_32__awac_agent_small_8_64_32_1_norm__a3c_agent_small_8_64_32_2-weight_probs'
learnable_weights_model = gnn.LearnableWeightsModel(100, n_models, 16)
model = gnn.GraphNNActorCriticMultiObsEnsemble(
    obs_types_to_models_dict,
    learnable_weights_model=learnable_weights_model,
    weight_logits=False
)
"""
with open(f'runs/a3c/smallMean_8_32_1_v1/2150_cp.txt', 'r') as f:
    serialized_string = f.readline()[2:-1].encode()
state_dict_bytes = base64.b64decode(serialized_string)
loaded_state_dicts = pickle.loads(state_dict_bytes)
model.load_state_dict(loaded_state_dicts['model_state_dict'])
"""
model.to(device=DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=2e-3)
"""
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
                      print_change_log=False)"""

env_kwargs = dict(
    env_device=DEVICE,
    out_device=DEVICE,
    normalize_reward=False,
    reward_type=ve.EVERY_STEP_EV_ZEROSUM,
    obs_type=model.obs_types
)
validation_env_kwargs_base = dict(
    n_envs=200,
    env_device=DEVICE,
    out_device=DEVICE,
    obs_type=env_kwargs['obs_type']
)
validation_opponent_env_kwargs = [
    #dict(
    #    opponent=va.BasicThompsonSampling(),
    #    opp_obs_type=ve.SUMMED_OBS
    #),
    dict(
        opponent=va.PullVegasSlotMachinesImproved(),
        opp_obs_type=ve.SUMMED_OBS
    ),
    dict(
        opponent=va.SavedRLAgent('a3c_agent_small_8_32-790', device=DEVICE, deterministic_policy=True),
        opp_obs_type=ve.SUMMED_OBS
    ),
    dict(
        opponent=va.SavedRLAgent('awac_agent_small_8_64_32_1_norm_v1-230', device=DEVICE, deterministic_policy=True),
        opp_obs_type=ve.SUMMED_OBS_WITH_TIMESTEP
    ),
    dict(
        opponent=va.SavedRLAgent('a3c_agent_small_8_64_32_2_v2-30', device=DEVICE, deterministic_policy=False),
        opp_obs_type=ve.LAST_60_EVENTS_AND_SUMMED_OBS_RAVELLED
    )
]
validation_env_kwargs_dicts = []
for opponent_kwargs in validation_opponent_env_kwargs:
    validation_env_kwargs_dicts.append(copy(validation_env_kwargs_base))
    validation_env_kwargs_dicts[-1].update(opponent_kwargs)
rl_alg_kwargs = dict(
    clip_grads=10. if type(optimizer) != AdaBelief else None,
    play_against_past_selves=False,
    n_past_selves=4,
    checkpoint_freq=20,
    log_params_full=True,
    opp_posterior_decay=0.99,
    run_separate_validation=True,
    validation_env_kwargs_dicts=validation_env_kwargs_dicts,
    deterministic_validation_policy=True
)
rl_train_kwargs = dict(
    batch_size=100,
    gamma=0.999
)

def model_constructor():
    assert False, 'Not implemented'
    # return gnn.GraphNNActorCritic(**graph_nn_kwargs)

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
    va.SavedRLAgent('awac_agent_small_8_64_32_1_norm_v1-230', **rl_agent_opp_kwargs)
]
if type(optimizer) == torch.optim.SGD:
    optimizer_name = 'sgd_'
elif type(optimizer) == torch.optim.Adam:
    optimizer_name = 'adam_'
elif type(optimizer) == torch.optim.Adadelta:
    optimizer_name = 'adadelta_'
elif type(optimizer) == AdaBelief:
    optimizer_name = 'adabelief_'
else:
    raise ValueError(f'Unrecognized optimizer: {optimizer}')

folder_name = f'ensemble_{ensemble_name}_{optimizer_name}v1'
a3c_alg = A3CVectorized(model_constructor, optimizer, env_kwargs['obs_type'],
                        model=model, device=DEVICE,
                        exp_folder=Path(f'runs/a3c/{folder_name}'),
                        **rl_alg_kwargs)
this_script = Path(__file__).absolute()
shutil.copy(this_script, a3c_alg.exp_folder / f'_{this_script.name}')

env_kwargs['n_envs'] = 100
#env_kwargs['opponent'] = va.MultiAgent(initial_opponent_pool)
#env_kwargs['opponent'] = va.BasicThompsonSampling()
#env_kwargs['opp_obs_type'] = ve.SUMMED_OBS
try:
    a3c_alg.train(n_episodes=101, **rl_train_kwargs, **env_kwargs)
except KeyboardInterrupt:
    if a3c_alg.true_ep_num > a3c_alg.checkpoint_freq:
        print('KeyboardInterrupt: saving model')
        a3c_alg.save(finished=True)
