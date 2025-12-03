#!/usr/bin/env python3
"""Test contact detection in humanoid_hop environment."""

import jax
import jax.numpy as jnp
from brax import envs

print("Creating humanoid_hop environment...")
env = envs.create('humanoid_hop', episode_length=100, backend='mjx')

print(f"Environment created:")
print(f"  - Observation dim: {env.observation_size}")
print(f"  - Action dim: {env.action_size}")

# Reset environment
rng = jax.random.PRNGKey(0)
state = jax.jit(env.reset)(rng)

print(f"\nInitial state:")
print(f"  - Torso height: {state.pipeline_state.x.pos[0, 2]:.3f}m")
print(f"  - Cost: {state.info.get('cost', 'N/A')}")
print(f"  - Left foot force: {state.info.get('left_foot_contact_force', 'N/A')}")
print(f"  - Right foot force: {state.info.get('right_foot_contact_force', 'N/A')}")

# Check body indices
print(f"\nBody indices:")
print(f"  - Left foot body idx: {env._left_foot_body_idx}")
print(f"  - Right foot body idx: {env._right_foot_body_idx}")
print(f"  - Link names: {env.sys.link_names if hasattr(env.sys, 'link_names') else 'N/A'}")

# Check if pipeline_state has cfrc_ext
print(f"\nPipeline state attributes:")
print(f"  - Has cfrc_ext: {hasattr(state.pipeline_state, 'cfrc_ext')}")
if hasattr(state.pipeline_state, 'cfrc_ext'):
    print(f"  - cfrc_ext shape: {state.pipeline_state.cfrc_ext.shape}")
    print(f"  - cfrc_ext[0] (torso): {state.pipeline_state.cfrc_ext[0]}")
    print(f"  - cfrc_ext nonzero entries:")
    for i, forces in enumerate(state.pipeline_state.cfrc_ext):
        if jnp.any(jnp.abs(forces) > 0.01):
            print(f"      Body {i}: {forces}")

# Check foot positions
print(f"\nFoot positions:")
left_idx = env._left_foot_body_idx
right_idx = env._right_foot_body_idx
print(f"  - Left shin (idx {left_idx}) pos: {state.pipeline_state.x.pos[left_idx]}")
print(f"  - Right shin (idx {right_idx}) pos: {state.pipeline_state.x.pos[right_idx]}")

# Check contact field
print(f"\nContact info:")
if hasattr(state.pipeline_state, 'contact') and state.pipeline_state.contact is not None:
    print(f"  - Has contact: True")
    print(f"  - Contact type: {type(state.pipeline_state.contact)}")
    if hasattr(state.pipeline_state.contact, 'link_idx'):
        print(f"  - Contact link indices: {state.pipeline_state.contact.link_idx}")
    if hasattr(state.pipeline_state.contact, 'vel'):
        print(f"  - Contact velocities shape: {state.pipeline_state.contact.vel.shape if hasattr(state.pipeline_state.contact.vel, 'shape') else 'scalar'}")
else:
    print(f"  - Has contact: False or None")

# Take a few steps
print(f"\nTaking 10 steps with zero actions...")
for i in range(10):
    action = jnp.zeros(env.action_size)
    state = jax.jit(env.step)(state, action)
    
    cost = state.info.get('cost', 0.0)
    left_force = state.info.get('left_foot_contact_force', 0.0)
    right_force = state.info.get('right_foot_contact_force', 0.0)
    violation = state.info.get('contact_violation', 0.0)
    
    print(f"  Step {i+1}: Cost={float(cost):.2f}, L={float(left_force):.1f}N, R={float(right_force):.1f}N, Violation={float(violation):.0f}")

print(f"\nFinal torso height: {state.pipeline_state.x.pos[0, 2]:.3f}m")
print("\nDone!")

