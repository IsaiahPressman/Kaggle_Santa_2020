import argparse
import contextlib
import io
import multiprocessing
import os
import numpy as np
from pathlib import Path
import time

with contextlib.redirect_stdout(io.StringIO()):
    # Silence gfootball import error
    from kaggle_environments import make

os.environ['OMP_NUM_THREADS'] = '1'

def get_game_result(agents):
    env = make('mab', debug=True)
    env.run(agents)
    
    p1_score = env.steps[-1][0]['reward']
    p2_score = env.steps[-1][1]['reward']
    return (p1_score, p2_score)


if __name__ == '__main__':
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Compare two agents in an asynchronous multi-game match.')
    parser.add_argument(
        'agent1_path',
        type=str,
        help='The path to the first agent'
    )
    parser.add_argument(
        'agent2_path',
        type=str,
        help='The path to the second agent'
    )
    parser.add_argument(
        '-g',
        '--n_games',
        type=int,
        default=(multiprocessing.cpu_count()-1)*2,
        help=f'The number of games to play. Default: {(multiprocessing.cpu_count()-1)*2}'
    )
    parser.add_argument(
        '-w',
        '--n_workers',
        type=int,
        default=multiprocessing.cpu_count()-1,
        help=f'The number of worker processes to use. Default: {multiprocessing.cpu_count()-1}'
    )
    args = parser.parse_args()
    
    print(f'{Path(args.agent1_path).stem} -vs- {Path(args.agent2_path).stem}')
    
    agent_paths_broadcasted = []
    for i in range(args.n_games):
        agent_paths_broadcasted.append((args.agent1_path, args.agent2_path))
    
    if args.n_workers == 1:
        results = map(get_game_result, agent_paths_broadcasted)
    else:        
        with multiprocessing.Pool(processes=args.n_workers) as pool:
            results = pool.map(get_game_result, agent_paths_broadcasted)
    
    p1_scores = []
    p2_scores = []
    for i, result in enumerate(results):
        p1_scores.append(result[0])
        p2_scores.append(result[1])
        print(f'Round {i+1}: {p1_scores[-1]} - {p2_scores[-1]}')
    p1_scores = np.array(p1_scores)
    p2_scores = np.array(p2_scores)
    print(f'Mean scores: {p1_scores.mean():.2f} - {p2_scores.mean():.2f}')
    print(f'Match score: {np.sum(p1_scores > p2_scores)} - {np.sum(p1_scores == p2_scores)} - {np.sum(p1_scores < p2_scores)}')
    print(f'Finished in {int(time.time() - start_time)} seconds')
