"""PPO-Lagrange training algorithm.

This module extends PPO with Lagrangian-based constrained optimization
for safe reinforcement learning.

Provides PPO-Lagrange specific components:
  - networks: PPONetworks with cost_value_network
  - losses: PPO-Lagrange loss with Lagrange multiplier
  - train: Training loop with Lagrange multiplier updates
"""

from crax.training.agents.ppo_lag.train import train
from crax.training.agents.ppo_lag import networks
from crax.training.agents.ppo_lag import losses
from crax.training.agents.ppo import checkpoint

__all__ = ['train', 'networks', 'losses', 'checkpoint']
