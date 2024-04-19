import numpy as np
import pandas as pd
import math
from scipy.stats import bernoulli, norm
from math import pi
from matplotlib import pyplot as plt


class Agent(object):
    """
    Agent decides the action
    """
    def __init__(self, bandit, policy, prior=0):
        self.policy = policy
        self.k = bandit.k
        self.prior = prior
        self._value_estimates = prior*np.ones(self.k)
        self.action_attempts = np.zeros(self.k)
        self.t = 0
        self.last_action = None

    def __str__(self):
        return '{}'.format(str(self.policy))

    def reset(self):
        self._value_estimates[:] = self.prior
        self.action_attempts[:] = 0
        self.last_action = None
        self.t = 0

    def choose(self):
        action = self.policy.choose(self)
        self.last_action = action
        return action

    def observe(self, reward):
        self.action_attempts[self.last_action] += 1

        g = 1 / self.action_attempts[self.last_action]
        q = self._value_estimates[self.last_action]

        self._value_estimates[self.last_action] += g * (reward - q)
        self.t += 1

    @property
    def value_estimates(self):
        return self._value_estimates