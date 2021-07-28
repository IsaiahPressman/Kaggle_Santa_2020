from copy import copy
import numpy as np


def get_evs(posteriors, thresholds):
    return np.sum(thresholds * (posteriors / posteriors.sum(axis=1, keepdims=True)),
                  axis=1) / 100.


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
        self.last_thresholds = None

    def get_action(self, observation):
        if observation.step == 0:
            return np.random.randint(self.n_bandits)
        else:
            r = observation.reward - self.total_reward
            self.total_reward = observation.reward
            self.my_act_history.append(observation.lastActions[observation.agentIndex])
            self.opp_act_history.append(observation.lastActions[1 - observation.agentIndex])
            if observation.step >= 2:
                # Update self.my_belief_opp_posteriors and self.opp_belief_my_posteriors
                self.update_my_belief()
                self.update_opp_belief()
            self.last_thresholds = copy(self.thresholds)
            self.my_posteriors, self.thresholds = self.get_updated_posteriors_thresholds(
                self.my_posteriors,
                self.thresholds,
                [self.my_act_history[-1], self.opp_act_history[-1]],
                [r, None]
            )

        assert False, "This EV calculation is broken and doesn't account for beliefs"
        evs = get_evs(self.my_posteriors, self.thresholds)
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
        loss_posteriors, _ = self.get_updated_posteriors_thresholds(
            opp_posteriors, self.last_thresholds, opp_act_history[-2], 0.
        )
        loss_evs = get_evs(loss_posteriors, self.last_thresholds)
        win_posteriors, _ = self.get_updated_posteriors_thresholds(
            opp_posteriors, self.last_thresholds, opp_act_history[-2], 1.
        )
        win_evs = get_evs(win_posteriors, self.last_thresholds)
        loss = loss_evs[opp_act_history[-1]] == np.max(loss_evs)
        win = win_evs[opp_act_history[-1]] == np.max(win_evs)
        if loss and win or (not loss and not win):
            return opp_posteriors
        elif loss:
            return loss_posteriors
            #return self.get_updated_posteriors_thresholds(
            #    opp_posteriors, self.last_thresholds, opp_act_history[-2],
            #    uncertaintify(0., get_evs(opp_posteriors, self.last_thresholds)[opp_act_history[-1]]).item()
            #)[0]
        else:
            return win_posteriors
            #return self.get_updated_posteriors_thresholds(
            #    opp_posteriors, self.last_thresholds, opp_act_history[-2],
            #    uncertaintify(1., get_evs(opp_posteriors, self.last_thresholds)[opp_act_history[-1]]).item()
            #)[0]

    def update_my_belief(self):
        self.my_belief_opp_posteriors = self.get_updated_belief(0)

    def update_opp_belief(self):
        self.opp_belief_my_posteriors = self.get_updated_belief(1)

    def get_updated_posteriors_thresholds(self, posteriors, thresholds, actions, rewards):
        if type(actions) == int and (rewards is None or type(rewards) == float):
            actions = (actions,)
            rewards = (rewards,)
        else:
            assert len(actions) == len(rewards)
        posteriors = copy(posteriors)
        thresholds = copy(thresholds)

        for act, r in zip(actions, rewards):
            if r is not None:
                likelihood = ((np.ceil(thresholds[act]) / self.sample_res) ** r) * (
                        (1 - np.ceil(thresholds[act]) / self.sample_res) ** (1 - r))
                posteriors[act] = posteriors[act] * likelihood
                posteriors[act] = posteriors[act] / posteriors[act].sum()
            thresholds[act] = thresholds[act] * self.decay_rate

        return posteriors, thresholds


curr_agent = None


def agent(observation, configuration):
    global curr_agent

    if curr_agent is None:
        curr_agent = Agent(configuration)

    return curr_agent.get_action(observation)
