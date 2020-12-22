import numpy as np
from scipy.stats import beta

post_a = None
post_b = None
bandit = None
total_reward = 0
c = 2.

def agent(observation, configuration):
    global reward_sums, total_reward, bandit, post_a, post_b, c, n_selections
    
    n_bandits = configuration.banditCount
    decay_rate = configuration.decayRate

    if observation.step == 0:
        post_a = np.ones(n_bandits)
        post_b = np.ones(n_bandits)
        n_selections = np.zeros(n_bandits)
    else:
        r = observation.reward - total_reward
        total_reward = observation.reward

        # Update Gaussian posterior
        post_a[bandit] += r
        post_b[bandit] += (1 - r)
        
        # Update scale
        for action in observation.lastActions:
            n_selections[action] += 1
    
    bound = post_a / (post_a + post_b).astype(float) + beta.std(post_a, post_b, scale=decay_rate**n_selections) * c
    #bound = beta.ppf(ppf_val, post_a, post_b, scale=decay_rate**n_selections)
    # The maximum possible expected reward is 1
    bound = np.minimum(bound, decay_rate**n_selections)
    bandit = int(np.argmax(bound))
    
    return bandit
