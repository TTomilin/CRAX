"""FOCOPS networks - re-exports from ppo.networks."""
from brax.training.agents.ppo.networks import PPONetworks, make_inference_fn, make_ppo_networks

__all__ = ['PPONetworks', 'make_inference_fn', 'make_ppo_networks']
