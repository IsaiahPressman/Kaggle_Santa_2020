import pickle
import numpy as np

posteriors = None
thresholds = None
bandit = None
total_reward = 0
posteriors_list = []
thresholds_list = []


def agent(observation, configuration):
    global total_reward, bandit, posteriors, thresholds
    global posteriors_list, thresholds_list
    
    n_bandits = configuration.banditCount
    decay_rate = configuration.decayRate
    sample_res = configuration.sampleResolution

    if observation.step == 0:
        posteriors = np.ones((n_bandits, sample_res+1)) / (sample_res+1)
        thresholds = np.broadcast_to(
            np.expand_dims(np.arange(sample_res+1, dtype=np.float64), 0),
            (n_bandits, sample_res+1)
        ).copy()
    else:
        r = observation.reward - total_reward
        total_reward = observation.reward

        # Update posterior
        if r == 0:
            likelihood = 1 - np.ceil(thresholds[bandit]) / sample_res
        elif r == 1:
            likelihood = np.ceil(thresholds[bandit]) / sample_res
        posteriors[bandit] = posteriors[bandit] * likelihood
        #posteriors[bandit] = posteriors[bandit] / posteriors[bandit].sum()
        
        # Update selections
        for action in observation.lastActions:
            thresholds[action] = thresholds[action] * decay_rate
    
    samples = [np.random.choice(thresholds[i], size=1, replace=True, p=posteriors[i]/posteriors[i].sum()).mean() for i in range(n_bandits)]
    bandit = int(np.argmax(samples))
    
    
    posteriors_list.append(posteriors)
    thresholds_list.append(thresholds)
    if observation.step >= 1998:
        with open('log.pkl', 'wb') as f:
            pickle.dump((posteriors_list, thresholds_list), f, pickle.HIGHEST_PROTOCOL)
        
    return bandit
