"""PPO-PID Lagrange losses - re-exports from ppo_lag.losses for backward compatibility."""
from brax.training.agents.ppo_lag.losses import (
    PPONetworkParams,
    compute_gae,
    compute_ppo_lagrange_loss,
    compute_ppo_loss,
)

__all__ = ['PPONetworkParams', 'compute_gae', 'compute_ppo_loss', 'compute_ppo_lagrange_loss']
