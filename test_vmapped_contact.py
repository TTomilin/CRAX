#!/usr/bin/env python3
"""Test contact detection in VMAPPED (batched) humanoid_hop environment."""

import jax
import jax.numpy as jnp
from brax import envs

print("Testing VMAPPED (batched) humanoid_hop environment...")
print("This mimics the training setup with 2048 parallel environments.\n")

# Create environment with batching (like in training)
batch_size = 4  # Use small batch for testing
env = envs.create(
    'humanoid_hop',
    episode_length=100,
    backend='mjx',
    batch_size=batch_size,  # This triggers VmapWrapper
    auto_reset=False
)

print(f"Batched environment created:")
print(f"  - Batch size: {batch_size}")
print(f"  - Observation dim: {env.observation_size}")
print(f"  - Action dim: {env.action_size}")

# Reset environment
rng = jax.random.PRNGKey(0)
state = jax.jit(env.reset)(rng)

print(f"\nInitial batched state shapes:")
print(f"  - Obs shape: {state.obs.shape}")
print(f"  - Reward shape: {state.reward.shape}")
print(f"  - Info keys: {list(state.info.keys())}")
print(f"  - Cost shape: {state.info['cost'].shape if 'cost' in state.info else 'N/A'}")
print(f"  - Left foot force shape: {state.info['left_foot_contact_force'].shape if 'left_foot_contact_force' in state.info else 'N/A'}")

print(f"\nInitial costs per environment:")
for i in range(batch_size):
    cost = float(state.info.get('cost', jnp.zeros(batch_size))[i])
    left = float(state.info.get('left_foot_contact_force', jnp.zeros(batch_size))[i])
    right = float(state.info.get('right_foot_contact_force', jnp.zeros(batch_size))[i])
    print(f"  Env {i}: Cost={cost:.2f}, L={left:.1f}N, R={right:.1f}N")

# Take a few steps with zero actions
print(f"\nTaking 5 steps with zero actions (batched)...")
for step in range(5):
    action = jnp.zeros((batch_size, env.action_size))
    state = jax.jit(env.step)(state, action)
    
    print(f"\n  Step {step+1}:")
    for i in range(batch_size):
        cost = float(state.info.get('cost', jnp.zeros(batch_size))[i])
        left = float(state.info.get('left_foot_contact_force', jnp.zeros(batch_size))[i])
        right = float(state.info.get('right_foot_contact_force', jnp.zeros(batch_size))[i])
        violation = float(state.info.get('contact_violation', jnp.zeros(batch_size))[i])
        print(f"    Env {i}: Cost={cost:.2f}, L={left:.1f}N, R={right:.1f}N, Violation={violation:.0f}")

print(f"\nDone!")
print("\nIf all costs are 0.0, contact detection is broken when vmapped.")
print("If costs are > 0, contact detection works but something else is wrong in training.")


