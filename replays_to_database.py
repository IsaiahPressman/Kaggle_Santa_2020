import argparse
from copy import copy
import contextlib
import io
import json
import numpy as np
from pathlib import Path
import torch
import tqdm

# Local imports
import vectorized_env as ve

with contextlib.redirect_stdout(io.StringIO()):
    # Silence gfootball import error
    import kaggle_environments


def convert_replay_for_visualization(replay, my_team_idx):
    try:
        env = kaggle_environments.make(
            'mab',
            configuration=replay['configuration'],
            steps=replay['steps'],
            info=replay['info']
        )
    except kaggle_environments.InvalidArgument:
        return None

    actual_rewards = np.zeros((2, 2000), dtype=np.float)
    expected_rewards = np.zeros((2, 2000), dtype=np.float)

    for step_idx, step in enumerate(env.steps):
        if step_idx == 0:
            continue
        for agent_idx, agent in enumerate(step):
            action = agent['action']
            thresholds = env.steps[step_idx - 1][0]['observation']['thresholds']
            actual_rewards[agent_idx, step_idx] = agent['reward']
            expected_rewards[agent_idx, step_idx] = np.ceil(thresholds[action]) / 100.
    actual_advantage = actual_rewards - actual_rewards[[1, 0], :]
    expected_advantage = expected_rewards - expected_rewards[[1, 0], :]

    if my_team_idx == 1:
        actual_rewards = actual_rewards[[1, 0], :]
        expected_rewards = expected_rewards[[1, 0], :]
        actual_advantage = actual_advantage[[1, 0], :]
        expected_advantage = expected_advantage[[1, 0], :]
    return {
        'actual_rewards': np.diff(actual_rewards, axis=1),
        'expected_rewards': expected_rewards[:, 1:],
        'actual_advantage': np.diff(actual_advantage, axis=1),
        'expected_advantage': expected_advantage[:, 1:]
    }


def convert_replays_to_s_a_r_d_s(replays, reward_type, obs_type, normalize_reward):
    assert len(replays) > 0
    kaggle_envs = [kaggle_environments.make(
        'mab',
        configuration=replay['configuration'],
        steps=replay['steps'],
        info=replay['info']
    ) for replay in replays]
    sim_env = ve.KaggleMABEnvTorchVectorized(
        n_envs=len(replays),
        reward_type=reward_type,
        obs_type=obs_type,
        normalize_reward=normalize_reward,
        env_device=torch.device('cpu'),
        out_device=torch.device('cpu')
    )
    kaggle_n_steps = len(kaggle_envs[0].steps)
    assert np.all(np.array([len(ke.steps) for ke in kaggle_envs]) == kaggle_n_steps)

    actions = []
    thresholds = []
    for step in range(kaggle_n_steps):
        actions.append(torch.stack([
            torch.tensor([s['action'] for s in ke.steps[step]]) for ke in kaggle_envs
        ]))
        thresholds.append(torch.stack([
            torch.tensor(ke.steps[step][0]['observation']['thresholds']) for ke in kaggle_envs
        ]).float())

    s_batch = []
    a_batch = []
    r_batch = []
    d_batch = []
    next_s_batch = []
    sim_env.reset()
    sim_env.orig_thresholds = thresholds[0].clone()
    actions.pop(0)
    thresholds.pop(0)
    s = sim_env.obs
    for i, a in enumerate(actions):
        a = a
        next_s, r, done, _ = sim_env.step(a)
        s_batch.append(s.clone())
        a_batch.append(a.clone())
        r_batch.append(r.clone())
        d_batch.append(torch.zeros(r.shape) if not done else torch.ones(r.shape))
        next_s_batch.append(next_s.clone())
        s = next_s
        assert torch.allclose(sim_env.thresholds.view(-1), thresholds[i].view(-1)), f'ERROR: {i}'

    return (torch.cat(s_batch).view(-1, *s.shape[-2:]),
            torch.cat(a_batch).view(-1),
            torch.cat(r_batch).view(-1),
            torch.cat(d_batch).view(-1),
            torch.cat(next_s_batch).view(-1, *s.shape[-2:]))


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def batch_convert_replay_files_to_pt_files(replay_paths, save_dir, batch_size=50, **kwargs):
    if save_dir.exists():
        assert save_dir.is_dir()
        assert not (save_dir / 'replay_names.txt').exists()
    save_dir.mkdir(exist_ok=True)

    replay_paths = list(replay_paths)
    print(f'Processing {len(replay_paths)} replays and saving output to {save_dir.absolute()}')
    remaining_paths = copy(replay_paths)
    s_batches = []
    a_batches = []
    r_batches = []
    d_batches = []
    next_s_batches = []
    for _ in tqdm.trange(np.ceil(float(len(remaining_paths)) / batch_size).astype(np.int)):
        if len(remaining_paths) > batch_size:
            s, a, r, d, next_s = convert_replays_to_s_a_r_d_s(
                [read_json(p) for p in remaining_paths[:batch_size]],
                **kwargs
            )
            remaining_paths = remaining_paths[batch_size:]
        else:
            s, a, r, d, next_s = convert_replays_to_s_a_r_d_s(
                [read_json(p) for p in remaining_paths],
                **kwargs
            )
            remaining_paths = []
        s_batches.append(s)
        a_batches.append(a)
        r_batches.append(r)
        d_batches.append(d)
        next_s_batches.append(next_s)
    assert len(remaining_paths) == 0

    with open(save_dir / 'replay_s.pt', 'wb') as f:
        torch.save(torch.cat(s_batches), f)
    with open(save_dir / 'replay_a.pt', 'wb') as f:
        torch.save(torch.cat(a_batches), f)
    with open(save_dir / 'replay_r.pt', 'wb') as f:
        torch.save(torch.cat(r_batches), f)
    with open(save_dir / 'replay_d.pt', 'wb') as f:
        torch.save(torch.cat(d_batches), f)
    with open(save_dir / 'replay_next_s.pt', 'wb') as f:
        torch.save(torch.cat(next_s_batches), f)
    with open(save_dir / 'replay_names.txt', 'w') as f:
        f.writelines([f'{rp.name}\n' for rp in replay_paths])
    print(f'Saved {torch.cat(s_batches).size(0):,} (s, a, r, d, next_s) transitions from {len(replay_paths)} replays')


def load_s_a_r_d_s(replay_database_dir):
    replay_database_dir = Path(replay_database_dir)
    with open(replay_database_dir / 'replay_s.pt', 'rb') as f:
        replay_s = torch.load(f)
    with open(replay_database_dir / 'replay_a.pt', 'rb') as f:
        replay_a = torch.load(f)
    with open(replay_database_dir / 'replay_r.pt', 'rb') as f:
        replay_r = torch.load(f)
    with open(replay_database_dir / 'replay_d.pt', 'rb') as f:
        replay_d = torch.load(f)
    with open(replay_database_dir / 'replay_next_s.pt', 'rb') as f:
        replay_next_s = torch.load(f)
    return replay_s, replay_a, replay_r, replay_d, replay_next_s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process a list of JSON replay files and create .pt files to easily load into '
                    'awac.BasicReplayBuffer. The output is less disk-efficient, but much faster to load.'
    )
    parser.add_argument(
        'save_dir',
        type=Path,
        help='Where to save the .pt output files'
    )
    parser.add_argument(
        'replay_paths',
        nargs='+',
        type=Path,
        help='A list of JSON replay file paths'
    )
    parser.add_argument(
        '-b',
        '--sample_size',
        type=int,
        default=50,
        help='The batch size to use for batched replay file processing. Default: 50'
    )
    parser.add_argument(
        '-r',
        '--reward_type',
        type=str,
        choices=ve.REWARD_TYPES,
        default=ve.EVERY_STEP_EV_ZEROSUM,
        help=f'The desired reward_type. See vectorized_env.py'
    )
    parser.add_argument(
        '-o',
        '--obs_type',
        type=str,
        choices=ve.OBS_TYPES,
        default=ve.SUMMED_OBS,
        help=f'The desired obs_type. See vectorized_env.py'
    )
    parser.add_argument(
        '-n',
        '--normalize_reward',
        action='store_true',
        help='Including this normalizes the reward. Default: False'
    )
    args = parser.parse_args()
    batch_convert_replay_files_to_pt_files(**vars(args))
