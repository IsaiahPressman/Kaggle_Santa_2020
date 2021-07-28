import multiprocessing
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

import replays_to_database


def process_replay_folder(replay_folder):
    replay_paths = list(replay_folder.glob('*[0-9].json'))
    sub_id = int(replay_folder.name)
    print(f'Converting {len(replay_paths)} replays to visualization format for submission {sub_id} in {replay_folder}')

    n_replays_processed = 0
    for rp in replay_paths:
        # if not Path(f'{replay_folder}/{rp.stem}_visualization_dict.pkl').exists() or FORCE:
        replay = replays_to_database.read_json(rp)
        replay_info = replays_to_database.read_json(rp.parent / (rp.stem + '_info.json'))
        if replay_info['agents'][0]['submissionId'] == sub_id:
            agent_idx = 0
        elif replay_info['agents'][1]['submissionId'] == sub_id:
            agent_idx = 1
        else:
            raise RuntimeError(f'{rp}: Neither agent has the correct submissionId')
        replay_viz_dict = replays_to_database.convert_replay_for_visualization(replay, agent_idx)

        if replay_viz_dict is not None:
            with open(f'{replay_folder}/{rp.stem}_visualization_dict.pkl', 'wb') as f:
                pickle.dump(replay_viz_dict, f, pickle.HIGHEST_PROTOCOL)
            n_replays_processed += 1

    return n_replays_processed


if __name__ == '__main__':
    DESTINATION = 'episode_scraping/replays_by_sub_id'
    sub_id_to_model_name_df = pd.read_csv(f'{DESTINATION}/sub_id_to_model_name.csv', index_col='sub_id')
    nan_mask = sub_id_to_model_name_df.index.isna() | np.any(sub_id_to_model_name_df.isna(), axis=1)
    if nan_mask.any():
        print('Ignoring the following nan-valued rows:')
        sub_id_to_model_name_df = sub_id_to_model_name_df[~nan_mask]
        sub_id_to_model_name_df.index = sub_id_to_model_name_df.index.astype(int)
    sub_id_to_model_names_dict = sub_id_to_model_name_df.to_dict()['model_name']
    assert len(np.unique(list(sub_id_to_model_name_df.index))) == len(sub_id_to_model_name_df)

    replay_folders = [Path(f'{DESTINATION}/{sub_id}') for sub_id in sub_id_to_model_names_dict.keys()]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        n_replays = pool.map(process_replay_folder, replay_folders)

    print(f'Finished processing {np.sum(n_replays)} replays from {len(replay_folders)} submissions')
