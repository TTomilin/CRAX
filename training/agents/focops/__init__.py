"""FOCOPS (First Order Constrained Optimization in Policy Space) training algorithm.

Reference: Zhang et al., "First Order Constrained Optimization in Policy Space",
NeurIPS 2020.
https://arxiv.org/abs/2002.06506
"""

from training.agents.focops.train import train
from training.agents.focops import networks
from training.agents.focops import losses
