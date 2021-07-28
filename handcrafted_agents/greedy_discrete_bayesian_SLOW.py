import numpy as np


class Agent:
    def __init__(self, configuration):
        self.n_bandits = int(configuration.banditCount)
        self.decay_rate = configuration.decayRate
        self.sample_res = configuration.sampleResolution

        self.total_reward = 0
        self.my_rewards = []
        self.my_act_history = []
        self.opp_act_history = []
        self.my_belief_opp_rewards = []
        self.opp_belief_my_rewards = []

    def get_action(self, observation):
        if observation.step == 0:
            return np.random.randint(self.n_bandits)
        else:
            r = observation.reward - self.total_reward
            self.total_reward = observation.reward
            self.my_rewards.append(float(r))
            self.my_act_history.append(observation.lastActions[observation.agentIndex])
            self.opp_act_history.append(observation.lastActions[1 - observation.agentIndex])
            self.my_belief_opp_rewards.append(None)
            self.opp_belief_my_rewards.append(None)
            if observation.step >= 2:
                # Update my_belief_opp_rewards and opp_belief_my_rewards
                self.update_my_belief()
                self.update_opp_belief()

        evs = self.get_my_evs(-1)
        return np.random.choice(np.arange(self.n_bandits)[evs == np.max(evs)]).item()

    def get_evs(self, step, player_idx):
        current_step = len(self.my_rewards)
        assert len(self.my_belief_opp_rewards) == current_step
        assert len(self.opp_belief_my_rewards) == current_step
        assert len(self.my_act_history) == current_step
        assert len(self.opp_act_history) == current_step
        # Allows for negative indexing
        step = np.arange(current_step)[step] + 1
        assert player_idx in (0, 1)
        posteriors = np.ones((self.n_bandits, self.sample_res + 1)) / (self.sample_res + 1)
        thresholds = np.repeat(
            np.arange(self.sample_res + 1, dtype=np.float64)[None, :],
            self.n_bandits,
            axis=0
        )
        if player_idx == 0:
            zipped = zip(
                self.my_rewards[:step],
                self.my_belief_opp_rewards[:step],
                self.my_act_history[:step],
                self.opp_act_history[:step]
            )
        else:
            zipped = zip(
                self.my_belief_opp_rewards[:step],
                self.opp_belief_my_rewards[:step],
                self.opp_act_history[:step],
                self.my_act_history[:step]
            )

        for my_r, opp_r, my_act, opp_act in zipped:
            if my_r is not None:
                my_likelihood = ((np.ceil(thresholds[my_act]) / self.sample_res) ** my_r) * (
                        (1 - np.ceil(thresholds[my_act]) / self.sample_res) ** (1 - my_r))
                posteriors[my_act] = posteriors[my_act] * my_likelihood
                posteriors[my_act] = posteriors[my_act] / posteriors[my_act].sum()

            if opp_r is not None:
                opp_likelihood = ((np.ceil(thresholds[opp_act]) / self.sample_res) ** opp_r) * (
                        (1 - np.ceil(thresholds[opp_act]) / self.sample_res) ** (1 - opp_r))
                posteriors[opp_act] = posteriors[opp_act] * opp_likelihood
                posteriors[opp_act] = posteriors[opp_act] / posteriors[opp_act].sum()

            for act in my_act, opp_act:
                thresholds[act] = thresholds[act] * self.decay_rate

        return np.sum(thresholds * (posteriors / posteriors.sum(axis=1, keepdims=True)), axis=1)

    def get_my_evs(self, step):
        return self.get_evs(step, 0)

    def get_opp_evs(self, step):
        return self.get_evs(step, 1)

    def update_belief(self, player_idx):
        assert player_idx in (0, 1)
        if player_idx == 0:
            opp_act_history = self.opp_act_history
            opp_reward_history = self.my_belief_opp_rewards
        else:
            opp_act_history = self.my_act_history
            opp_reward_history = self.opp_belief_my_rewards

        # Make counterfactual assumptions about the reward
        opp_reward_history[-2] = 0.
        loss_evs = self.get_evs(-2, player_idx)
        opp_reward_history[-2] = 1.
        win_evs = self.get_evs(-2, player_idx)
        loss = loss_evs[opp_act_history[-1]] == np.max(loss_evs)
        win = win_evs[opp_act_history[-1]] == np.max(win_evs)
        if loss and win or (not loss and not win):
            opp_reward_history[-2] = None
        elif loss:
            opp_reward_history[-2] = 0.
        else:
            opp_reward_history[-2] = 1.

    def update_my_belief(self):
        self.update_belief(0)

    def update_opp_belief(self):
        self.update_belief(1)


curr_agent = None


def agent(observation, configuration):
    global curr_agent

    if curr_agent is None:
        curr_agent = Agent(configuration)

    return curr_agent.get_action(observation)
