import numpy as np
from scipy.stats import beta

post_a = None
post_b = None
bandit = None
total_reward = 0


def agent(observation, configuration):
    global total_reward, bandit, post_a, post_b
    
    n_bandits = configuration.banditCount
    decay_rate = configuration.decayRate
    sample_res = configuration.sampleResolution

    if observation.step == 0:
        post_a = np.ones((n_bandits, 1))
        post_b = np.ones((n_bandits, 1))
    else:
        r = observation.reward - total_reward
        total_reward = observation.reward

        # Update beta posterior
        post_a[bandit] += r
        post_b[bandit] += (1 - r)
        
    beta_kwargs = dict(
        a=post_a,
        b=post_b,
    )
    #discretized_posteriors = beta.cdf((np.arange(sample_res).reshape(1,-1) + 1) / sample_res, **beta_kwargs) - beta.cdf(np.arange(sample_res).reshape(1,-1) / sample_res, **beta_kwargs)
    discretized_posteriors = beta.pdf(np.arange(sample_res).reshape(1,-1) / sample_res, post_a, post_b)
    samples = [np.random.choice(sample_res, p=dp/dp.sum()) for dp in discretized_posteriors]
    bandit = int(np.argmax(samples))
    
    return bandit
