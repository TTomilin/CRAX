# Copyright 2024 The Brax Authors.
# Re-export from base PPO networks for backward compatibility.
from brax.training.agents.ppo.networks import (
    PPONetworks,
    make_inference_fn,
    make_ppo_networks,
)

__all__ = ['PPONetworks', 'make_inference_fn', 'make_ppo_networks']
