"""P3O (Penalized Proximal Policy Optimization) training algorithm.

Reference: Zhang et al., "Penalized Proximal Policy Optimization for Safe
Reinforcement Learning", IJCAI 2022.
https://arxiv.org/abs/2205.11814
"""

from crax.training.agents.p3o.train import train
from crax.training.agents.p3o import networks
from crax.training.agents.p3o import losses
