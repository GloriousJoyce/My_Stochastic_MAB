import numpy as np
import pandas as pd
import math
from scipy.stats import bernoulli, norm
from math import pi
from matplotlib import pyplot as plt


class Environment:
    """
    the entire gamestate
    """
    def __init__(self, bandit, agents):
        self.bandit = bandit
        self.agents = agents

    def reset(self):
        self.bandit.reset()
        for agent in self.agents:
            agent.reset()

    def run(self, trials=10000, experiments=1):
        scores = np.zeros((trials, len(self.agents)))
        regret = np.zeros_like(scores)

        for _ in range(experiments):
            self.reset()
            for t in range(trials):
                for i, agent in enumerate(self.agents):
                    action = agent.choose()
                    reward, is_optimal = self.bandit.pull(action)
                    agent.observe(reward)

                    scores[t, i] += reward
                    if not is_optimal:
                        regret[t, i] += max(self.bandit.mu) - self.bandit.mu[action]

        for i in range(len(self.agents)):
            for j in range(1, trials):
                regret[j][i] += regret[j-1][i]

        label = [str(x) for x in self.agents]

        return scores / experiments, regret / experiments, label

    def plot_regret(self, regret, label):

        trials = regret.shape[0]

        plt.figure()
        plt.grid(True)
        plt.title(str(self.bandit))
        x = np.linspace(0, trials, trials)
        regret = regret.transpose()
        for i in range(regret.shape[0]):
            plt.plot(x, regret[i], label=label[i], linewidth="2")
        plt.xlabel("Round", fontsize=20)
        plt.ylabel("Regret", fontsize=20)
        plt.legend(fontsize=12)
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(str(self.bandit) + " Regret.png", dpi=600, bbox_inches='tight')
        plt.show()
