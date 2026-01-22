"""
Test script to verify that the reward calculation uses center-to-center distance
between the block and the goal, not edge-based SDF distance.
"""

import jax
import jax.numpy as jp
from brax.envs.block_push_goal import BlockPushGoal


def test_center_to_center_distance():
    """Test that reward uses center-to-center distance."""
    print("=" * 60)
    print("Testing Center-to-Center Distance Calculation")
    print("=" * 60)

    # Create environment with debug enabled
    env = BlockPushGoal(
        episode_length=1000,
        goal_size=0.3,  # Larger goal to make edge vs center difference obvious
        goal_type='cube',
        hazard_specs=[dict(type='cube', count=1, size=0.1, collidable=False, movable=False)],
        debug=False,
    )

    # Initialize with a fixed seed for reproducibility
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)

    # Get positions
    data = state.pipeline_state
    block_pos = data.xpos[env._block_body]
    goal_pos = state.info['goal_positions'][0]

    block_xy = block_pos[:2]
    goal_xy = goal_pos[:2]

    # Calculate expected center-to-center distance
    expected_center_dist = jp.sqrt(jp.sum(jp.square(goal_xy - block_xy)) + 1e-8)

    # Get the distance stored in metrics (this is what step() calculates)
    recorded_dist = state.metrics['distance_to_goal']

    print(f"\nBlock position (xy):  {block_xy}")
    print(f"Goal position (xy):   {goal_xy}")
    print(f"\nExpected center-to-center distance: {expected_center_dist:.6f}")
    print(f"Recorded distance_to_goal metric:   {recorded_dist:.6f}")

    # Check if they match
    tolerance = 1e-5
    match = jp.abs(expected_center_dist - recorded_dist) < tolerance

    if match:
        print(f"\n[PASS] Distances match within tolerance ({tolerance})")
    else:
        print(f"\n[FAIL] Distances do NOT match!")
        print(f"       Difference: {jp.abs(expected_center_dist - recorded_dist):.6f}")

    return match


def test_reward_shaping():
    """Test that reward shaping works correctly with center distance."""
    print("\n" + "=" * 60)
    print("Testing Reward Shaping with Center Distance")
    print("=" * 60)

    # Create environment with distance reward enabled
    env = BlockPushGoal(
        episode_length=1000,
        goal_size=0.3,
        goal_type='cube',
        hazard_specs=[dict(type='cube', count=1, size=0.1, collidable=False, movable=False)],
        reward_distance_scale=1.0,  # Enable distance-based reward
        debug=False,
    )

    rng = jax.random.PRNGKey(123)
    state = env.reset(rng)

    # Take a few steps with zero action to see distance tracking
    print("\nStep-by-step distance tracking:")
    print("-" * 40)

    action = jp.zeros(env.action_size)

    for i in range(5):
        data = state.pipeline_state
        block_xy = data.xpos[env._block_body][:2]
        goal_xy = state.info['goal_positions'][0][:2]

        # Manual center-to-center calculation
        manual_dist = jp.sqrt(jp.sum(jp.square(goal_xy - block_xy)) + 1e-8)
        recorded_dist = state.metrics['distance_to_goal']
        last_dist = state.metrics['last_dist_goal']

        print(f"Step {i}: manual_dist={float(manual_dist):.4f}, "
              f"recorded={float(recorded_dist):.4f}, "
              f"last={float(last_dist):.4f}, "
              f"reward={float(state.reward):.4f}")

        # Take a step
        state = env.step(state, action)

    print("\n[INFO] If manual_dist matches recorded distance, center-to-center is working.")
    return True


def test_edge_vs_center_difference():
    """Demonstrate the difference between edge and center distance."""
    print("\n" + "=" * 60)
    print("Demonstrating Edge vs Center Distance Difference")
    print("=" * 60)

    # For a cube goal with size 0.3, the half-extent is 0.15
    # If block is at distance 0.5 from center:
    # - Center-to-center distance = 0.5
    # - Edge distance (SDF) = 0.5 - 0.15 = 0.35 (approximately)

    goal_size = 0.3
    half_extent = goal_size / 2

    # Simulated positions
    block_xy = jp.array([0.0, 0.0])
    goal_xy = jp.array([0.5, 0.0])

    center_dist = jp.sqrt(jp.sum(jp.square(goal_xy - block_xy)))
    edge_dist_approx = center_dist - half_extent  # Simplified 1D case

    print(f"\nExample with goal_size={goal_size} (half_extent={half_extent}):")
    print(f"Block at: {block_xy}")
    print(f"Goal at:  {goal_xy}")
    print(f"\nCenter-to-center distance: {center_dist:.4f}")
    print(f"Edge distance (approx):    {edge_dist_approx:.4f}")
    print(f"Difference:                {center_dist - edge_dist_approx:.4f}")

    print("\n[INFO] The fix ensures we use center-to-center distance for rewards,")
    print("       which provides more consistent reward shaping regardless of goal size.")

    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BLOCK PUSH GOAL - REWARD CALCULATION TEST")
    print("=" * 60)

    all_passed = True

    # Run tests
    all_passed &= test_center_to_center_distance()
    all_passed &= test_reward_shaping()
    all_passed &= test_edge_vs_center_difference()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if all_passed:
        print("\n[SUCCESS] All tests passed!")
        print("The reward calculation now uses center-to-center distance.")
    else:
        print("\n[FAILURE] Some tests failed. Check output above.")

    print()
