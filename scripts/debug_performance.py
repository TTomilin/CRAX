#!/usr/bin/env python3
"""Debug script to identify performance issues in CRoMaX."""

import sys
import time

from brax.envs import get_environment

sys.path.insert(0, '.')

print("=" * 60)
print("CRoMaX Debug Performance Test")
print("=" * 60)

# Step 1: Import timing
print("\n[1] Importing modules...")
t0 = time.time()

import jax
import jax.numpy as jnp

# Uncomment to see compilation logs (very verbose):
# jax.config.update("jax_log_compiles", True)

print(f"    JAX imported in {time.time() - t0:.2f}s")
print(f"    JAX devices: {jax.devices()}")
t0 = time.time()

print(f"    CRoMaX imported in {time.time() - t0:.2f}s")

# Step 2: Create environment
print("\n[3] Creating environment (reach-v2)...")
print("    Using primitive shapes (fast) - set use_meshes=True for detailed STL meshes")
t0 = time.time()
try:
    env = get_environment("reach-v2", backend="mjx", auto_reset=False)
    print(f"    Environment created in {time.time() - t0:.2f}s")
    print(f"    obs_size: {env.observation_size}, action_size: {env.action_size}")
    print(f"    System has {env.sys.nq} DOFs, {env.sys.nv} velocities, {env.sys.nu} actuators")
except Exception as e:
    print(f"    ERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Step 3: Reset environment
print("\n[4] Resetting environment...")
t0 = time.time()
rng = jax.random.PRNGKey(0)
rng, reset_rng = jax.random.split(rng)

try:
    state = env.reset(reset_rng)
    # Force blocking to measure actual time
    state.obs.block_until_ready()
    print(f"    Reset completed in {time.time() - t0:.2f}s")
    print(f"    obs shape: {state.obs.shape}")
    print(f"    goal_pos in info: {'goal_pos' in state.info}")
except Exception as e:
    print(f"    ERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Step 4: JIT compile step function
print("\n[5] JIT compiling step function...")
print("    (This includes MJX compilation - expect 30-120s for complex models)")
t0 = time.time()


@jax.jit
def jit_step(state, action):
    return env.step(state, action)


# Dummy action
action = jnp.zeros(4)

# First call triggers compilation
try:
    state = jit_step(state, action)
    state.obs.block_until_ready()  # Wait for computation
    elapsed = time.time() - t0
    print(f"    JIT compilation + first step in {elapsed:.2f}s")
    if elapsed > 60:
        print("    (Note: Long compilation is normal for MJX with complex meshes)")
except Exception as e:
    print(f"    ERROR during JIT step: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Step 5: Timing subsequent steps
print("\n[6] Timing 100 steps (post-JIT)...")
t0 = time.time()
for i in range(100):
    rng, action_rng = jax.random.split(rng)
    action = jax.random.uniform(action_rng, shape=(4,), minval=-1, maxval=1)
    state = jit_step(state, action)

state.obs.block_until_ready()  # Block at end only
elapsed = time.time() - t0
print(f"    100 steps completed in {elapsed:.3f}s ({elapsed / 100 * 1000:.2f}ms per step)")

if elapsed < 1.0:
    print("    SUCCESS: Post-JIT performance is good!")
else:
    print("    WARNING: Post-JIT steps are still slow")

# Step 6: Test rendering (separate from step performance)
print("\n[7] Testing rendering...")
t0 = time.time()
try:
    # Just get one frame to test
    trajectory = [state.pipeline_state]
    frames = env.render(trajectory, height=240, width=320)
    print(f"    Rendered 1 frame in {time.time() - t0:.2f}s")
    print(f"    Frame shape: {frames[0].shape if frames else 'N/A'}")
except Exception as e:
    print(f"    ERROR during render: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("Debug test completed!")
print("=" * 60)
print("\nSummary:")
print("- If JIT compilation takes 30-120s, that's expected for MJX+meshes")
print("- Post-JIT steps should be <10ms each")
print("- If post-JIT steps are slow, there's a tracing issue")
print("=" * 60)
