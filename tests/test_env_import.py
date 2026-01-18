#!/usr/bin/env python3
"""Test script to verify humanoid_hop environment can be loaded."""

print("Testing humanoid_hop environment import...")

try:
    from brax import envs
    print(f"✓ Imported brax.envs")
    
    print(f"\nAvailable environments: {list(envs._envs.keys())}")
    
    if 'humanoid_hop' in envs._envs:
        print(f"✓ humanoid_hop is in registry")
    else:
        print(f"✗ humanoid_hop NOT in registry!")
        exit(1)
    
    # Try to get the environment class
    env_class = envs._envs['humanoid_hop']
    print(f"✓ Environment class: {env_class}")
    
    # Try to create an instance
    print(f"\nCreating humanoid_hop environment...")
    env = envs.get_environment('humanoid_hop', hopping_leg='left', contact_threshold=0.5, cost_weight=1.0)
    print(f"✓ Environment created successfully!")
    print(f"  - Observation size: {env.observation_size}")
    print(f"  - Action size: {env.action_size}")
    
    print(f"\n✓ All tests passed!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)


