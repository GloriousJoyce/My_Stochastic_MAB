import numpy as np
import pandas as pd
import math
from scipy.stats import bernoulli, norm
from math import pi
from matplotlib import pyplot as plt


class Bandit:
    """
    A Multi-armed Bandit
    """
    def __init__(self, k):
        self.k = k
        self.action_values = np.zeros(k)
        self.optimal = 0

    def reset(self):
        self.action_values = np.zeros(self.k)
        self.optimal = 0

    def pull(self, action):
        return 0, True


class Gaussian_Bandit(Bandit):
    """
    Gaussian bandits - N(mu, sigma^2)
    """
    def __init__(self, k, mu, sigma=1):

        assert len(mu) == k, "Wrong size!"

        super(Gaussian_Bandit, self).__init__(k)
        self.mu = mu
        self.sigma = sigma
        self.reset()

    def __str__(self):
        return 'Gaussian Bandits'

    def reset(self):
        self.action_values = np.random.normal(self.mu, self.sigma)
        self.optimal = np.argmax(self.mu)
        indices = np.where(self.mu == self.optimal)[0]
        self.optimal = indices

    def pull(self, action):
        return (np.random.normal(self.mu[action], self.sigma),
                action in self.optimal)


class Bernoulli_Bandit(Bandit):
    """
    Bernoulli bandits - p
    """
    def __init__(self, k, mu):

        assert len(mu) == k, "Wrong size!"

        super(Bernoulli_Bandit, self).__init__(k)
        self.mu = mu
        self.reset()

    def __str__(self):
        return 'Bernoulli Bandits'

    def reset(self):
        self.action_values = np.random.binomial(1, self.mu)
        self.optimal = np.argmax(self.mu)
        indices = np.where(self.mu == self.optimal)[0]
        self.optimal = indices

    def pull(self, action):
        return (np.random.binomial(1, self.mu[action]),
                action in self.optimal)