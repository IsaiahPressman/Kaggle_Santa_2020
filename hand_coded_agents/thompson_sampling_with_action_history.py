import numpy as np

post_a = None
post_b = None
bandit = None
total_reward = 0
n_selections = None


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
    
    #samples = beta.rvs(post_a, post_b, scale=decay_rate**n_selections)
    samples = np.random.beta(post_a, post_b) * decay_rate**n_selections
    bandit = int(np.argmax(samples))
    
    return bandit
