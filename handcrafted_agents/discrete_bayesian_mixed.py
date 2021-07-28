from copy import copy
import numpy as np

#def uncertaintify(reward, ev):
#    uncertainty_const = 0.5
#    assert 0. <= uncertainty_const <= 1.
#    return np.average([reward, ev], weights=[1.-uncertainty_const, uncertainty_const])


class Agent:
    def __init__(self, configuration):
        self.n_bandits = int(configuration.banditCount)
        self.decay_rate = configuration.decayRate
        self.sample_res = configuration.sampleResolution

        self.total_reward = 0
        self.my_posteriors = np.ones((self.n_bandits, self.sample_res + 1)) / (self.sample_res + 1)
        self.thresholds = np.repeat(
            np.arange(self.sample_res + 1, dtype=np.float64)[None, :],
            self.n_bandits,
            axis=0
        )
        self.my_belief_opp_posteriors = np.ones((self.n_bandits, self.sample_res + 1)) / (self.sample_res + 1)
        self.opp_belief_my_posteriors = np.ones((self.n_bandits, self.sample_res + 1)) / (self.sample_res + 1)

        self.my_act_history = []
        self.opp_act_history = []
        self.my_r_history = []
        self.est_opp_r_history = []
        self.last_thresholds = None

    def get_action(self, observation):
        if observation.step == 0:
            return np.random.randint(self.n_bandits)
        else:
            r = observation.reward - self.total_reward
            self.total_reward = observation.reward
            self.my_act_history.append(observation.lastActions[observation.agentIndex])
            self.opp_act_history.append(observation.lastActions[1 - observation.agentIndex])
            self.my_r_history.append(r)
            if observation.step >= 2:
                # Update self.my_belief_opp_posteriors and self.opp_belief_my_posteriors
                last_opp_r = self.update_my_belief()
                self.update_opp_belief()
                self.my_posteriors = self.get_updated_posteriors(
                    self.my_posteriors,
                    self.last_thresholds,
                    [self.my_act_history[-2], self.opp_act_history[-2]],
                    [self.my_r_history[-2], last_opp_r]
                )
            self.last_thresholds = copy(self.thresholds)
            self.thresholds = self.get_updated_thresholds(
                self.thresholds,
                [self.my_act_history[-1], self.opp_act_history[-1]]
            )

        evs = self.get_evs(
            self.get_updated_posteriors(
                self.my_posteriors,
                self.last_thresholds,
                [self.my_act_history[-1], self.opp_act_history[-1]],
                [self.my_r_history[-1], None]),
            self.thresholds)
        #return np.random.choice(np.arange(self.n_bandits), p=evs / evs.sum()).item()
        return np.random.choice(np.arange(self.n_bandits)[evs == np.max(evs)]).item()

    def get_updated_belief(self, player_idx):
        assert player_idx in (0, 1)
        if player_idx == 0:
            opp_act_history = self.opp_act_history
            opp_posteriors = self.my_belief_opp_posteriors
        else:
            opp_act_history = self.my_act_history
            opp_posteriors = self.opp_belief_my_posteriors

        # Make counterfactual assumptions about the reward
        possible_rewards = np.linspace(0.2, 0.8, 11)
        possible_posteriors = [self.get_updated_posteriors(
            opp_posteriors,
            self.last_thresholds,
            opp_act_history[-2],
            r
        ) for r in possible_rewards]
        possible_evs = self.get_evs(
            np.stack(possible_posteriors),
            self.last_thresholds[None, :]
        )
        possible_act_freqs = possible_evs[:, opp_act_history[-1]] / possible_evs.sum(axis=-1)
        actual_act_freq = (np.sum(np.array(opp_act_history) == opp_act_history[-1])) / len(opp_act_history)
        #print(possible_evs[:, opp_act_history[-1]], possible_evs.sum(axis=-1))
        #print(actual_act_freq, possible_act_freqs)
        #print('Reward:', possible_rewards[np.argmin(np.abs(possible_act_freqs - actual_act_freq))])
        #print()

        #print(possible_evs.mean(axis=-1))
        #print(possible_act_freqs, actual_act_freq)
        #print(possible_rewards[np.argmin(np.abs(possible_act_freqs - actual_act_freq))],
        #      np.argmin(np.abs(possible_act_freqs - actual_act_freq)))
        return (possible_posteriors[np.argmin(np.abs(possible_act_freqs - actual_act_freq))],
                possible_rewards[np.argmin(np.abs(possible_act_freqs - actual_act_freq))])

    def update_my_belief(self):
        self.my_belief_opp_posteriors, last_r = self.get_updated_belief(0)
        self.est_opp_r_history.append(last_r)
        return last_r

    def update_opp_belief(self):
        self.opp_belief_my_posteriors, last_r = self.get_updated_belief(1)
        return last_r

    def get_updated_posteriors(self, posteriors, thresholds, actions, rewards):
        if not hasattr(actions, '__len__'):
            actions = (actions,)
        if not hasattr(rewards, '__len__'):
            rewards = (rewards,)
        assert len(actions) == len(rewards)
        posteriors = copy(posteriors)
        thresholds = copy(thresholds)

        for act, r in zip(actions, rewards):
            if r is not None:
                likelihood = ((np.ceil(thresholds[act]) / float(self.sample_res)) ** r) * (
                        (1 - np.ceil(thresholds[act]) / float(self.sample_res)) ** (1 - r))
                posteriors[act] = posteriors[act] * likelihood
                posteriors[act] = posteriors[act] / posteriors[act].sum()
        return posteriors

    def get_updated_thresholds(self, thresholds, actions):
        if type(actions) == int:
            actions = (actions,)
        thresholds = copy(thresholds)
        for act in actions:
            thresholds[act] = thresholds[act] * self.decay_rate
        return thresholds

    def get_evs(self, posteriors, thresholds):
        return np.sum(thresholds * (posteriors / posteriors.sum(axis=-1, keepdims=True)),
                      axis=-1) / float(self.sample_res)


curr_agent = None


def agent(observation, configuration):
    global curr_agent

    if curr_agent is None:
        curr_agent = Agent(configuration)

    return curr_agent.get_action(observation)
