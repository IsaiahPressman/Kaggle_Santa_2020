import numpy as np

posteriors = None
bandit = None
total_reward = 0


def agent(observation, configuration):
    global total_reward, bandit, posteriors
    
    n_bandits = configuration.banditCount
    decay_rate = configuration.decayRate
    sample_res = configuration.sampleResolution

    if observation.step == 0:
        posteriors = np.ones((n_bandits, sample_res+1)) / (sample_res+1)
    else:
        r = observation.reward - total_reward
        total_reward = observation.reward

        # Update posterior
        if r == 0:
            likelihood = 1 - np.arange(sample_res+1, dtype=np.float64) / sample_res
        elif r == 1:
            likelihood = np.arange(sample_res+1, dtype=np.float64) / sample_res
        posteriors[bandit] = posteriors[bandit] * likelihood
        #posteriors[bandit] = posteriors[bandit] / posteriors[bandit].sum()
        
    samples = [np.random.choice(sample_res+1, p=p/p.sum()) for p in posteriors]
    bandit = int(np.argmax(samples))
    
    return bandit
