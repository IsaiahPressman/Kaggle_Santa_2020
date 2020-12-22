
import numpy as np
from scipy import stats

post_a = None
post_b = None
bandit = None
total_reward = 0

class Agent():
    def __init__(self, configuration, n_mc_sims=500, mc_noising_rate=0.8, min_mc_kept=10):
        self.n_bandits = configuration.banditCount
        self.decay_rate = configuration.decayRate
        self.total_reward = 0
        self.last_act = None
        
        self.n_mc_sims = n_mc_sims
        self.mc_noising_rate = mc_noising_rate
        self.min_mc_kept = min_mc_kept
        
        self.post_a = np.ones(self.n_bandits)
        self.post_b = np.ones(self.n_bandits)
        self.est_opp_post_a = np.ones(self.n_bandits)
        self.est_opp_post_b = np.ones(self.n_bandits)
        self.opp_actions = np.zeros(self.n_bandits)
        self.amount_decayed = np.ones(self.n_bandits, dtype=np.float)
    
    def simulate_opp_posteriors(self, observation):
        opp_last_act = observation.lastActions[not observation.agentIndex]
        # TODO: This simulates a and b from scratch each round - maybe there is a better way that uses the estimations across rounds?
        # self.opp_actions + 1, because np.random.randint requires a minimum value of 1 (and always returns 0 in that case anyways)
        #mc_successes = np.concatenate([np.random.randint(n, size=(1, self.n_mc_sims, 1)) for n in self.opp_actions + 1], axis=0)
        mc_a_kept = np.empty((self.n_bandits, 0))
        mc_b_kept = np.empty((self.n_bandits, 0))
        
        for i in range(5):
            """
            mc_successes = stats.binom.rvs(
                self.opp_actions.astype(np.int)[:,np.newaxis,np.newaxis],
                (self.est_opp_post_a / (self.est_opp_post_a + self.est_opp_post_b))[:,np.newaxis,np.newaxis],
                size=(self.n_bandits, self.n_mc_sims, 1)
            )"""
            #mc_successes = stats.randint.rvs(0, self.opp_actions.astype(np.int)[:,np.newaxis,np.newaxis] + 1, size=(self.n_bandits, self.n_mc_sims, 1))
            mc_successes = stats.betabinom.rvs(
                self.opp_actions.astype(np.int)[:,np.newaxis],
                np.maximum(self.est_opp_post_a[:,np.newaxis] * (self.mc_noising_rate ** i), 1),
                np.maximum(self.est_opp_post_b[:,np.newaxis] * (self.mc_noising_rate ** i), 1),
                size=(self.n_bandits, self.n_mc_sims)
            )
            mc_failures = self.opp_actions[:, np.newaxis] - mc_successes
            # a/b = successes/failures + 1, because a and b start at 1 for a uniform prior
            mc_a = mc_successes + 1
            mc_b = mc_failures + 1

            mc_thompson_samples = np.random.beta(mc_a, mc_b, size=(self.n_bandits, self.n_mc_sims))# * self.amount_decayed[:, np.newaxis]
            mc_acts = np.argmax(mc_thompson_samples, axis=0)
            correct_acts_mask = mc_acts == opp_last_act
            mc_a_kept = np.concatenate([mc_a_kept, mc_a[:, correct_acts_mask]], axis=1)
            mc_b_kept = np.concatenate([mc_b_kept, mc_b[:, correct_acts_mask]], axis=1)
            if mc_a_kept.shape[1] >= self.min_mc_kept:
                break
                
        if mc_a_kept.shape[1] >= self.min_mc_kept:
            self.est_opp_post_a = np.mean(mc_a_kept.squeeze(), axis=-1)
            self.est_opp_post_b = np.mean(mc_b_kept.squeeze(), axis=-1)
                
            # TODO: Does this second iteration of monte-carlo help stabilize results?
            #mc_thompson_samples = np.random.beta(mc_a, mc_b, size=(self.n_bandits, correct_acts_mask.sum(), 100))# * self.amount_decayed[:, np.newaxis, np.newaxis]
            #mc_weights = np.sum(np.argmax(mc_thompson_samples, axis=0) == opp_last_act, axis=-1)
            #assert False
            #self.est_opp_post_a = np.average(mc_a.squeeze(), axis=-1, weights=mc_weights.squeeze())
            #self.est_opp_post_b = np.average(mc_b.squeeze(), axis=-1, weights=mc_weights.squeeze())
        
    def get_action(self, observation):
        if observation.step > 0:
            r = observation.reward - self.total_reward
            self.total_reward = observation.reward

            # Update agent's beta posterior
            self.post_a[self.last_act] += r
            self.post_b[self.last_act] += (1 - r)
            
            # Estimate opponent's posteriors before they selected a bandit
            self.simulate_opp_posteriors(observation)
            
            # Update opp_actions and n_pulls after estimating opponent's posteriors, but before selecting an action for this round
            opp_last_act = observation.lastActions[not observation.agentIndex]
            self.opp_actions[opp_last_act] += 1
            for action in observation.lastActions:
                self.amount_decayed[action] *= self.decay_rate
        
        if observation.step > 2:
            samples = np.random.beta(self.post_a + self.est_opp_post_a - 1, self.post_b + self.est_opp_post_b - 1) * self.amount_decayed
        else:
            samples = np.random.beta(self.post_a, self.post_b)
        self.last_act = int(np.argmax(samples))
        return self.last_act

curr_agent = None
def agent(observation, configuration):
    global curr_agent
    
    if curr_agent is None:
        curr_agent = Agent(configuration)
    
    return curr_agent.get_action(observation)
