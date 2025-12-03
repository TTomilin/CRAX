#!/usr/bin/env python3
"""Diagnostic test to verify contact force detection in HumanoidHop."""

import jax
import jax.numpy as jnp
from brax import envs

def test_contact_detection():
    """Test that contact forces are actually detected when humanoid touches ground."""
    print("=" * 60)
    print("Contact Force Detection Diagnostic")
    print("=" * 60)
    
    # Create environment
    env = envs.create('humanoid_hop', episode_length=2000, backend='mjx')
    rng = jax.random.PRNGKey(42)
    
    # Reset and run for more steps to let humanoid settle
    print("\n1. Letting humanoid settle on ground...")
    state = jax.jit(env.reset)(rng)
    
    # Run with zero actions to let it settle
    for _ in range(20):
        action = jnp.zeros(env.action_size)
        state = jax.jit(env.step)(state, action)
    
    print(f"   After settling:")
    print(f"   - Left foot force: {float(state.info['left_foot_contact_force']):.3f} N")
    print(f"   - Right foot force: {float(state.info['right_foot_contact_force']):.3f} N")
    print(f"   - Contact violation: {float(state.info['contact_violation']):.0f}")
    print(f"   - Cost: {float(state.info['cost']):.3f}")
    
    # Now test with various actions
    print("\n2. Testing with different actions...")
    rng = jax.random.PRNGKey(123)
    state = jax.jit(env.reset)(rng)
    
    max_left = 0.0
    max_right = 0.0
    total_violations = 0
    
    for step in range(100):
        rng, action_rng = jax.random.split(rng)
        action = jax.random.uniform(action_rng, (env.action_size,), minval=-1.0, maxval=1.0)
        state = jax.jit(env.step)(state, action)
        
        left_force = float(state.info['left_foot_contact_force'])
        right_force = float(state.info['right_foot_contact_force'])
        violation = float(state.info['contact_violation'])
        
        max_left = max(max_left, left_force)
        max_right = max(max_right, right_force)
        total_violations += int(violation > 0.5)
        
        if step < 5:
            print(f"   Step {step}: L={left_force:.2f}N, R={right_force:.2f}N, Viol={violation:.0f}")
    
    print(f"\n   Statistics over 100 steps:")
    print(f"   - Max left foot force: {max_left:.2f} N")
    print(f"   - Max right foot force: {max_right:.2f} N")
    print(f"   - Total violations: {total_violations}/100")
    
    # Test left vs right hopping
    print("\n3. Comparing left vs right leg hopping...")
    for leg in ['left', 'right']:
        env_leg = envs.create('humanoid_hop', episode_length=1000, 
                              hopping_leg=leg, contact_threshold=0.5, backend='mjx')
        rng = jax.random.PRNGKey(999)
        state_leg = jax.jit(env_leg.reset)(rng)
        
        violations = 0
        for _ in range(50):
            rng, action_rng = jax.random.split(rng)
            action = jax.random.uniform(action_rng, (env_leg.action_size,), minval=-0.5, maxval=0.5)
            state_leg = jax.jit(env_leg.step)(state_leg, action)
            violations += int(float(state_leg.info['contact_violation']) > 0.5)
        
        print(f"   Hopping on {leg}: {violations}/50 violations with random policy")
    
    print("\n" + "=" * 60)
    
    if max_left > 0 or max_right > 0:
        print("✓ Contact forces detected!")
    else:
        print("⚠ Warning: No contact forces detected. This may be expected if")
        print("  humanoid is airborne or cfrc_ext needs different indexing.")
    print("=" * 60)

if __name__ == '__main__':
    test_contact_detection()



