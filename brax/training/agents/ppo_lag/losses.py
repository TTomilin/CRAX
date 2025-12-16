# Copyright 2024 The Brax Authors.
# Licensed under the Apache License, Version 2.0
"""PPO-Lagrange losses - re-exports from ppo.losses for backward compatibility."""
from brax.training.agents.ppo.losses import PPONetworkParams, compute_gae, compute_ppo_loss, compute_ppo_lagrange_loss
__all__ = ["PPONetworkParams", "compute_gae", "compute_ppo_loss", "compute_ppo_lagrange_loss"]
