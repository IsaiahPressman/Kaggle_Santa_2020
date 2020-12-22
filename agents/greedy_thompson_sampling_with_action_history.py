import numpy as np
from scipy.stats import beta

post_a = None
post_b = None
bandit = None
total_reward = 0
n_selections = None

# Tunable hyperparameters:
greed_start = 1000
greed_end = 1900
min_greed = 0.
max_greed = 0.5


def get_greed_coef(timestep):
    if timestep < greed_start:
        return min_greed
    elif timestep > greed_end:
        return max_greed
    else:
        return min((max_greed - min_greed) * (timestep - greed_start) / (greed_end - greed_start), 1.)


def agent(observation, configuration):
    global total_reward, bandit, post_a, post_b, n_selections
    
    n_bandits = configuration.banditCount
    decay_rate = configuration.decayRate

    if observation.step == 0:
        post_a = np.ones(n_bandits)
        post_b = np.ones(n_bandits)
        n_selections = np.zeros(n_bandits)
    else:
        r = observation.reward - total_reward
        total_reward = observation.reward

        # Update beta posterior
        post_a[bandit] += r
        post_b[bandit] += (1 - r)
        
        # Update selections
        for action in observation.lastActions:
            n_selections[action] += 1
    
    greed = get_greed_coef(observation.step)
    samples = beta.rvs(post_a, post_b, scale=decay_rate**n_selections)
    means = beta.mean(post_a, post_b, scale=decay_rate**n_selections)
    bandit = int(np.argmax(
        greed * means + (1-greed) * samples
    ))
    
    return bandit
