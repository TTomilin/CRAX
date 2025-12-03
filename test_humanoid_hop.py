#!/usr/bin/env python3
"""Test script for HumanoidHop environment."""

import jax
import jax.numpy as jnp
from brax import envs

def test_humanoid_hop_basic():
    """Test basic environment creation and stepping."""
    print("=" * 60)
    print("Testing HumanoidHop Environment")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating environment...")
    env = envs.create('humanoid_hop', episode_length=1000, backend='mjx')
    print(f"   ✓ Environment created successfully")
    print(f"   - Observation shape: {env.observation_size}")
    print(f"   - Action shape: {env.action_size}")
    
    # Initialize random key
    rng = jax.random.PRNGKey(0)
    
    # Reset environment
    print("\n2. Resetting environment...")
    state = jax.jit(env.reset)(rng)
    print(f"   ✓ Environment reset successfully")
    print(f"   - Observation shape: {state.obs.shape}")
    print(f"   - Initial reward: {state.reward}")
    print(f"   - Initial cost (from info): {state.info.get('cost', 'N/A')}")
    
    # Test random actions for multiple steps
    print("\n3. Testing environment with random actions...")
    n_steps = 10
    total_cost = 0.0
    violations = 0
    
    for step in range(n_steps):
        rng, action_rng = jax.random.split(rng)
        action = jax.random.uniform(action_rng, (env.action_size,), minval=-1.0, maxval=1.0)
        
        state = jax.jit(env.step)(state, action)
        
        cost = state.info.get('cost', 0.0)
        left_force = state.info.get('left_foot_contact_force', 0.0)
        right_force = state.info.get('right_foot_contact_force', 0.0)
        violation = state.info.get('contact_violation', 0.0)
        
        total_cost += float(cost)
        violations += int(float(violation) > 0.5)
        
        if step < 3:  # Print first 3 steps
            print(f"   Step {step+1}:")
            print(f"     - Reward: {float(state.reward):.3f}")
            print(f"     - Cost: {float(cost):.3f}")
            print(f"     - Left foot force: {float(left_force):.3f} N")
            print(f"     - Right foot force: {float(right_force):.3f} N")
            print(f"     - Violation: {float(violation):.0f}")
    
    print(f"\n   Summary after {n_steps} steps:")
    print(f"   - Total cost accumulated: {total_cost:.2f}")
    print(f"   - Steps with violations: {violations}/{n_steps}")
    print(f"   - Violation rate: {100*violations/n_steps:.1f}%")
    
    # Test both hopping legs
    print("\n4. Testing different hopping legs...")
    for leg in ['left', 'right']:
        env_test = envs.create('humanoid_hop', episode_length=1000, 
                               hopping_leg=leg, backend='mjx')
        state_test = jax.jit(env_test.reset)(jax.random.PRNGKey(42))
        print(f"   ✓ Hopping on {leg} leg configuration created successfully")
    
    # Test cost threshold sensitivity
    print("\n5. Testing different contact thresholds...")
    for threshold in [0.1, 0.5, 1.0, 5.0]:
        env_test = envs.create('humanoid_hop', episode_length=1000, 
                               contact_threshold=threshold, backend='mjx')
        state_test = jax.jit(env_test.reset)(jax.random.PRNGKey(42))
        
        # Take one step
        action = jnp.zeros(env_test.action_size)
        state_test = jax.jit(env_test.step)(state_test, action)
        
        print(f"   Threshold {threshold:>4.1f} N: cost = {float(state_test.info['cost']):.3f}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    
    return env, state

def test_vectorized_env():
    """Test batched/vectorized environment."""
    print("\n" + "=" * 60)
    print("Testing Vectorized HumanoidHop Environment")
    print("=" * 60)
    
    batch_size = 4
    print(f"\n1. Creating vectorized environment (batch_size={batch_size})...")
    env = envs.create('humanoid_hop', episode_length=1000, 
                     batch_size=batch_size, backend='mjx')
    print(f"   ✓ Vectorized environment created")
    
    rng = jax.random.PRNGKey(0)
    state = jax.jit(env.reset)(rng)
    
    print(f"\n2. Checking shapes...")
    print(f"   - Observation shape: {state.obs.shape} (expected: ({batch_size}, obs_dim))")
    print(f"   - Reward shape: {state.reward.shape} (expected: ({batch_size},))")
    print(f"   - Cost shape: {state.info['cost'].shape if hasattr(state.info['cost'], 'shape') else 'scalar'}")
    
    # Step with random actions
    print(f"\n3. Stepping vectorized environment...")
    action = jax.random.uniform(rng, (batch_size, env.action_size), minval=-1.0, maxval=1.0)
    state = jax.jit(env.step)(state, action)
    
    print(f"   ✓ Step successful")
    print(f"   - Rewards: {state.reward}")
    print(f"   - Costs: {[float(x) for x in state.metrics['cost']]}")
    
    print("\n" + "=" * 60)
    print("✓ Vectorized tests passed!")
    print("=" * 60)

if __name__ == '__main__':
    # Run tests
    try:
        env, state = test_humanoid_hop_basic()
        test_vectorized_env()
        print("\n🎉 All tests completed successfully!\n")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

