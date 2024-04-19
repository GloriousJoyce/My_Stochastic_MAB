import numpy as np


class Policy:
    def __str__(self):
        return 'generic policy'

    def choose(self, agent):
        return 0


class EpsilonGreedy_Policy(Policy):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return '\u03B5-greedy (\u03B5={})'.format(self.epsilon)

    def choose(self, agent):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(agent.value_estimates))
        else:
            action = np.argmax(agent.value_estimates)
            check = np.where(agent.value_estimates == agent.value_estimates[action])[0]
            if len(check) == 1:
                return action
            else:
                return np.random.choice(check)


class UCB_Policy(Policy):
    def __init__(self, c):
        self.c = c

    def __str__(self):
        return 'UCB1 - (c={})'.format(self.c)

    def choose(self, agent):
        exploration = self.c * np.log(agent.t+1) / agent.action_attempts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1/2)

        q = agent.value_estimates + exploration
        action = np.argmax(q)
        check = np.where(q == q[action])[0]
        if len(check) == 1:
            return action
        else:
            return np.random.choice(check)