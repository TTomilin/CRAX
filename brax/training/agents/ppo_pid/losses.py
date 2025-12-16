# Copyright 2024 The Brax Authors.
# Re-export from base PPO losses for backward compatibility.
from brax.training.agents.ppo.losses import (
    PPONetworkParams,
    compute_gae,
    compute_ppo_loss,
    compute_ppo_lagrange_loss,
)

__all__ = ['PPONetworkParams', 'compute_gae', 'compute_ppo_loss', 'compute_ppo_lagrange_loss']
