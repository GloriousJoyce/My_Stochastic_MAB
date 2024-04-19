import matplotlib

from Agent import Agent
from Bandit import Bernoulli_Bandit, Gaussian_Bandit
from Policy import EpsilonGreedy_Policy, UCB_Policy
from Environment import Environment


Experiments = 10
Trials = 100000

# Bernoulli

bandit = Bernoulli_Bandit(5, [0.9, 0.7, 0.5, 0.3, 0.1])
agents = [Agent(bandit, EpsilonGreedy_Policy(0.005)),
          Agent(bandit, UCB_Policy(2))]

env = Environment(bandit, agents)
_, Regret, Label = env.run(Trials, Experiments)
env.plot_regret(Regret, Label)


# Gaussian

bandit = Gaussian_Bandit(5, [0.9, 0.7, 0.5, 0.3, 0.1], 0.1)
agents = [Agent(bandit, EpsilonGreedy_Policy(0.005)),
          Agent(bandit, UCB_Policy(2))]

env = Environment(bandit, agents)
_, Regret, Label = env.run(Trials, Experiments)
env.plot_regret(Regret, Label)