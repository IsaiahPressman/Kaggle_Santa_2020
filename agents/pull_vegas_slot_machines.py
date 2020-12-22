import numpy as np
import pandas as pd
import random, os, datetime

total_reward = 0
bandit_dict = {}


def get_next_bandit():
    best_bandit = 0
    best_bandit_expected = 0
    for bnd in bandit_dict:
        expect = (bandit_dict[bnd]['win'] - bandit_dict[bnd]['loss'] + bandit_dict[bnd]['opp'] - (bandit_dict[bnd]['opp']>0)*1.5) \
                 / (bandit_dict[bnd]['win'] + bandit_dict[bnd]['loss'] + bandit_dict[bnd]['opp'])
        if expect > best_bandit_expected:
            best_bandit_expected = expect
            best_bandit = bnd
    return best_bandit

def agent(observation, configuration):
    global total_reward, bandit_dict

    my_pull = random.randrange(configuration.banditCount)
    if 0 == observation.step:
        total_reward = 0
        bandit_dict = {}
        for i in range(configuration.banditCount):
            bandit_dict[i] = {'win': 1, 'loss': 0, 'opp': 0}
    else:
        last_reward = observation.reward - total_reward
        total_reward = observation.reward
        
        my_idx = observation.agentIndex
        if 0 < last_reward:
            bandit_dict[observation.lastActions[my_idx]]['win'] = bandit_dict[observation.lastActions[my_idx]]['win'] +1
        else:
            bandit_dict[observation.lastActions[my_idx]]['loss'] = bandit_dict[observation.lastActions[my_idx]]['loss'] +1
        bandit_dict[observation.lastActions[1-my_idx]]['opp'] = bandit_dict[observation.lastActions[1-my_idx]]['opp'] +1
        my_pull = get_next_bandit()
    
    return my_pull