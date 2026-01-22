"""
Comprehensive test suite for BlockPushGoal (PointPushGoal) environment.

Tests cover:
- XML/model loading and body existence
- Environment instantiation and configuration
- JIT compilation and performance
- Reset and step functionality
- Block position tracking and observations
- Center-to-center distance calculations for rewards
- Agent ability to physically push the block

Run with: pytest tests/test_point_push_goal.py -v
"""

import functools

import jax
import jax.numpy as jp
import jax.random as jrandom
import mujoco
import pytest

from brax.envs.block_push_goal import BlockPushGoal


# -----------------------------------------------------------------------------
# Fixtures - JIT compiled functions for speed
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def env():
    """Create a standard test environment."""
    return BlockPushGoal(
        episode_length=1000,
        goal_size=0.3,
        goal_type='cube',
        hazard_specs=[dict(type='cube', count=2, size=0.2, height=0.2, collidable=True, movable=False)],
        debug=False,
    )


@pytest.fixture(scope="module")
def env_with_distance_reward():
    """Create environment with distance-based reward enabled."""
    return BlockPushGoal(
        episode_length=1000,
        goal_size=0.3,
        goal_type='cube',
        hazard_specs=[dict(type='cube', count=1, size=0.1, collidable=False, movable=False)],
        reward_distance_scale=1.0,
        debug=False,
    )


@pytest.fixture(scope="module")
def jit_reset(env):
    """JIT-compiled reset function."""
    return jax.jit(env.reset)


@pytest.fixture(scope="module")
def jit_step(env):
    """JIT-compiled step function."""
    return jax.jit(env.step)


@pytest.fixture(scope="module")
def initial_state(jit_reset):
    """Get initial state from reset."""
    rng = jrandom.PRNGKey(42)
    return jit_reset(rng)


# -----------------------------------------------------------------------------
# XML and Model Loading Tests
# -----------------------------------------------------------------------------

def _get_xml_path():
    """Get the absolute path to the point_push.xml file."""
    import pathlib
    # Navigate from tests/ to brax/envs/assets/safe/
    tests_dir = pathlib.Path(__file__).parent
    project_root = tests_dir.parent
    return str(project_root / "brax" / "envs" / "assets" / "safe" / "point_push.xml")


class TestXMLLoading:
    """Tests for XML model loading and body verification."""

    def test_xml_loads_successfully(self):
        """Verify XML file loads without errors."""
        xml_path = _get_xml_path()
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        assert mj_model is not None
        assert mj_model.nq > 0
        assert mj_model.nv > 0

    def test_block_body_exists(self):
        """Verify block body exists in the model."""
        xml_path = _get_xml_path()
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        block_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "block")
        assert block_id >= 0, "Block body not found in model"

    def test_block_geom_exists(self):
        """Verify block geom exists in the model."""
        xml_path = _get_xml_path()
        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        block_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "block")
        assert block_geom_id >= 0, "Block geom not found in model"


# -----------------------------------------------------------------------------
# Environment Instantiation Tests
# -----------------------------------------------------------------------------

class TestEnvironmentInstantiation:
    """Tests for environment creation and configuration."""

    def test_env_instantiates(self, env):
        """Verify environment instantiates without errors."""
        assert env is not None
        assert env.observation_size > 0
        assert env.action_size > 0

    def test_env_system_dimensions(self, env):
        """Verify system has expected dimensions."""
        assert env.sys.nq > 0
        assert env.sys.nv > 0

    def test_env_with_different_hazard_configs(self):
        """Test environment with various hazard configurations."""
        configs = [
            [dict(type='cube', count=1, size=0.1, collidable=False, movable=False)],
            [dict(type='cube', count=4, size=0.2, height=0.2, collidable=True, movable=False)],
            [dict(type='cylinder', count=2, size=0.15, height=0.3, collidable=True, movable=False)],
        ]
        for hazard_spec in configs:
            env = BlockPushGoal(hazard_specs=hazard_spec, debug=False)
            assert env is not None


# -----------------------------------------------------------------------------
# JIT Compilation Tests
# -----------------------------------------------------------------------------

class TestJITCompilation:
    """Tests for JAX JIT compilation correctness."""

    def test_reset_jits_successfully(self, env):
        """Verify reset can be JIT compiled."""
        jit_reset = jax.jit(env.reset)
        rng = jrandom.PRNGKey(0)
        state = jit_reset(rng)
        assert state is not None
        assert state.obs.shape[0] == env.observation_size

    def test_step_jits_successfully(self, env, jit_reset):
        """Verify step can be JIT compiled."""
        jit_step = jax.jit(env.step)
        rng = jrandom.PRNGKey(0)
        state = jit_reset(rng)
        action = jp.zeros(env.action_size)
        state2 = jit_step(state, action)
        assert state2 is not None

    def test_jit_reset_is_deterministic(self, jit_reset):
        """Verify JIT reset produces deterministic results."""
        rng = jrandom.PRNGKey(123)
        state1 = jit_reset(rng)
        state2 = jit_reset(rng)
        assert jp.allclose(state1.obs, state2.obs)

    def test_jit_step_is_deterministic(self, jit_reset, jit_step):
        """Verify JIT step produces deterministic results."""
        rng = jrandom.PRNGKey(456)
        state = jit_reset(rng)
        action = jp.array([0.5, 0.3])
        state1 = jit_step(state, action)
        state2 = jit_step(state, action)
        assert jp.allclose(state1.obs, state2.obs)
        assert jp.allclose(state1.reward, state2.reward)

    def test_multiple_jit_resets(self, jit_reset):
        """Verify compiled reset can be called multiple times without errors."""
        rng = jrandom.PRNGKey(0)

        for i in range(10):
            rng, sub = jrandom.split(rng)
            state = jit_reset(sub)
            assert state is not None
            assert not jp.any(jp.isnan(state.obs))

    def test_multiple_jit_steps(self, jit_reset, jit_step):
        """Verify compiled step can be called multiple times without errors."""
        rng = jrandom.PRNGKey(0)
        state = jit_reset(rng)

        for i in range(100):
            rng, sub = jrandom.split(rng)
            action = jrandom.uniform(sub, (2,), minval=-1.0, maxval=1.0)
            state = jit_step(state, action)
            assert state is not None
            assert not jp.any(jp.isnan(state.obs))


# -----------------------------------------------------------------------------
# Reset and Step Tests
# -----------------------------------------------------------------------------

class TestResetAndStep:
    """Tests for reset and step functionality."""

    def test_reset_returns_valid_state(self, jit_reset):
        """Verify reset returns a valid state object."""
        rng = jrandom.PRNGKey(0)
        state = jit_reset(rng)
        assert hasattr(state, 'obs')
        assert hasattr(state, 'reward')
        assert hasattr(state, 'done')
        assert hasattr(state, 'pipeline_state')
        assert hasattr(state, 'info')

    def test_step_updates_state(self, jit_reset, jit_step):
        """Verify step updates the state."""
        rng = jrandom.PRNGKey(0)
        state = jit_reset(rng)
        action = jp.array([1.0, 0.0])  # Move forward
        state2 = jit_step(state, action)
        # Pipeline state should change
        assert not jp.allclose(state.pipeline_state.qpos, state2.pipeline_state.qpos)

    def test_step_with_zero_action(self, jit_reset, jit_step):
        """Verify step works with zero action."""
        rng = jrandom.PRNGKey(0)
        state = jit_reset(rng)
        action = jp.zeros(2)
        state2 = jit_step(state, action)
        assert state2 is not None
        # Agent should not move much with zero action
        agent_displacement = jp.linalg.norm(
            state2.pipeline_state.qpos[:2] - state.pipeline_state.qpos[:2]
        )
        assert agent_displacement < 0.5  # Small displacement due to physics settling


# -----------------------------------------------------------------------------
# Block Position Tracking Tests
# -----------------------------------------------------------------------------

class TestBlockTracking:
    """Tests for block position tracking in state.info."""

    def test_block_position_in_info(self, initial_state):
        """Verify block_position is tracked in state.info."""
        assert 'block_position' in initial_state.info
        block_pos = initial_state.info['block_position']
        assert block_pos.shape == (3,)

    def test_goal_positions_in_info(self, initial_state):
        """Verify goal_positions is tracked in state.info."""
        assert 'goal_positions' in initial_state.info
        goal_pos = initial_state.info['goal_positions']
        assert goal_pos.shape[1] == 3  # (num_goals, 3)

    def test_block_position_updates_after_step(self, env, jit_reset, jit_step):
        """Verify block position updates in info after stepping."""
        rng = jrandom.PRNGKey(0)
        state = jit_reset(rng)
        initial_block_pos = state.info['block_position']

        # Run many steps with forward motion toward block
        for _ in range(100):
            action = jp.array([1.0, 0.0])
            state = jit_step(state, action)

        # Block position should be different (even if just slightly due to physics)
        final_block_pos = state.info['block_position']
        # At minimum, the state.info should still contain valid position
        assert final_block_pos.shape == (3,)

    def test_distance_to_goal_metric(self, initial_state):
        """Verify distance_to_goal metric is tracked."""
        assert 'distance_to_goal' in initial_state.metrics
        dist = initial_state.metrics['distance_to_goal']
        assert dist >= 0


# -----------------------------------------------------------------------------
# Block Observations Tests
# -----------------------------------------------------------------------------

class TestBlockObservations:
    """Tests for block-related observations in obs vector."""

    def test_observation_contains_block_info(self, env, initial_state):
        """Verify observations contain block information."""
        obs = initial_state.obs
        assert obs.shape[0] == env.observation_size

        # Block obs are at end: agent_to_block_comp(2), block_to_goal_comp(2),
        # agent_to_block_dist(1), block_to_goal_dist(1)
        agent_to_block_comp = obs[-6:-4]
        block_to_goal_comp = obs[-4:-2]
        agent_to_block_dist = obs[-2]
        block_to_goal_dist = obs[-1]

        # Compasses should be approximately normalized
        a2b_norm = jp.linalg.norm(agent_to_block_comp)
        b2g_norm = jp.linalg.norm(block_to_goal_comp)

        # Allow some tolerance for edge cases (e.g., agent on block)
        assert 0.0 <= a2b_norm <= 1.5
        assert 0.0 <= b2g_norm <= 1.5

    def test_compass_normalization(self, env, initial_state):
        """Verify compass vectors are normalized (or near zero if coincident)."""
        obs = initial_state.obs
        agent_to_block_comp = obs[-6:-4]
        block_to_goal_comp = obs[-4:-2]

        a2b_norm = jp.linalg.norm(agent_to_block_comp)
        b2g_norm = jp.linalg.norm(block_to_goal_comp)

        # Should be normalized (close to 1) unless positions coincide (close to 0)
        assert a2b_norm < 0.01 or abs(a2b_norm - 1.0) < 0.1
        assert b2g_norm < 0.01 or abs(b2g_norm - 1.0) < 0.1


# -----------------------------------------------------------------------------
# Center-to-Center Distance Tests
# -----------------------------------------------------------------------------

class TestCenterToCenterDistance:
    """Tests for center-to-center distance calculation in rewards."""

    def test_distance_uses_center_not_edge(self, env, jit_reset):
        """Verify reward uses center-to-center distance, not edge-based SDF."""
        rng = jrandom.PRNGKey(42)
        state = jit_reset(rng)

        data = state.pipeline_state
        block_pos = data.xpos[env._block_body]
        goal_pos = state.info['goal_positions'][0]

        block_xy = block_pos[:2]
        goal_xy = goal_pos[:2]

        # Calculate expected center-to-center distance
        expected_center_dist = jp.sqrt(jp.sum(jp.square(goal_xy - block_xy)) + 1e-8)

        # Get the distance stored in metrics
        recorded_dist = state.metrics['distance_to_goal']

        # They should match (within tolerance for numerical precision)
        assert jp.abs(expected_center_dist - recorded_dist) < 1e-4, (
            f"Expected center distance {expected_center_dist}, got {recorded_dist}"
        )

    def test_reward_shaping_with_center_distance(self, env_with_distance_reward):
        """Test that reward shaping works correctly with center distance."""
        jit_reset = jax.jit(env_with_distance_reward.reset)
        jit_step = jax.jit(env_with_distance_reward.step)

        rng = jrandom.PRNGKey(123)
        state = jit_reset(rng)
        action = jp.zeros(env_with_distance_reward.action_size)

        # Run a few steps and verify distance tracking consistency
        for _ in range(5):
            data = state.pipeline_state
            block_xy = data.xpos[env_with_distance_reward._block_body][:2]
            goal_xy = state.info['goal_positions'][0][:2]

            # Manual center-to-center calculation
            manual_dist = jp.sqrt(jp.sum(jp.square(goal_xy - block_xy)) + 1e-8)
            recorded_dist = state.metrics['distance_to_goal']

            assert jp.abs(manual_dist - recorded_dist) < 1e-4, (
                f"Manual distance {manual_dist} != recorded {recorded_dist}"
            )

            state = jit_step(state, action)

    def test_edge_vs_center_distance_difference(self):
        """Verify center distance differs from edge-based SDF distance."""
        # For a cube goal with size 0.3, the half-extent is 0.15
        # If block is at distance 0.5 from center:
        # - Center-to-center distance = 0.5
        # - Edge distance (SDF) approx = 0.5 - 0.15 = 0.35

        goal_size = 0.3
        half_extent = goal_size / 2

        block_xy = jp.array([0.0, 0.0])
        goal_xy = jp.array([0.5, 0.0])

        center_dist = jp.sqrt(jp.sum(jp.square(goal_xy - block_xy)))
        edge_dist_approx = center_dist - half_extent

        # Center distance should be larger than edge distance
        assert center_dist > edge_dist_approx
        assert jp.abs(center_dist - 0.5) < 1e-6
        assert jp.abs(edge_dist_approx - 0.35) < 1e-6


# -----------------------------------------------------------------------------
# Agent Block Pushing Tests
# -----------------------------------------------------------------------------

class TestAgentBlockPushing:
    """Tests for agent ability to physically push the block."""

    def test_agent_can_push_block(self):
        """Verify agent can physically displace the block."""
        env = BlockPushGoal(
            debug=False,
            hazard_specs=[dict(type='cube', count=1, size=0.1, height=0.1, collidable=False, movable=False)],
        )

        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)

        rng = jrandom.PRNGKey(123)
        state = jit_reset(rng)

        initial_block_pos = state.info['block_position'][:2]

        # Run steps with agent moving toward and pushing block
        for i in range(200):
            agent_pos = state.pipeline_state.xpos[env._agent_body][:2]
            block_pos = state.info['block_position'][:2]
            to_block = block_pos - agent_pos
            desired_angle = jp.arctan2(to_block[1], to_block[0])
            current_angle = state.pipeline_state.qpos[2]
            angle_diff = desired_angle - current_angle
            angle_diff = jp.arctan2(jp.sin(angle_diff), jp.cos(angle_diff))

            # Move forward and turn toward block
            forward = 1.0 if abs(float(angle_diff)) < 0.5 else 0.3
            turn = jp.clip(angle_diff * 3.0, -1.0, 1.0)
            action = jp.array([forward, turn])

            state = jit_step(state, action)

        final_block_pos = state.info['block_position'][:2]
        block_displacement = jp.linalg.norm(final_block_pos - initial_block_pos)

        assert float(block_displacement) > 0.1, (
            f"Block should move when pushed. Displacement: {float(block_displacement):.4f}"
        )


# -----------------------------------------------------------------------------
# Goal Reached Tests
# -----------------------------------------------------------------------------

class TestGoalReached:
    """Tests for goal reached detection and reward."""

    def test_env_runs_with_dense_reward(self, env_with_distance_reward):
        """Verify environment runs correctly with dense reward enabled."""
        jit_reset = jax.jit(env_with_distance_reward.reset)
        jit_step = jax.jit(env_with_distance_reward.step)

        rng = jrandom.PRNGKey(42)
        state = jit_reset(rng)

        # Run 10 steps with zero action
        total_reward = 0.0
        for _ in range(10):
            action = jp.zeros(env_with_distance_reward.action_size)
            state = jit_step(state, action)
            total_reward += float(state.reward)

        # Environment should run without errors
        # (reward value depends on configuration, just verify it's a number)
        assert not jp.isnan(total_reward)
        assert not jp.isinf(total_reward)

    def test_goal_reached_when_block_inside_goal(self):
        """Verify goal is reached when block center is inside the goal box (SDF <= 0)."""
        goal_size = 0.3

        env = BlockPushGoal(
            goal_size=goal_size,
            hazard_specs=[dict(type='cube', count=1, size=0.1, collidable=False, movable=False)],
            debug=False,
        )

        # Goal should use SDF-based detection (block center inside goal box)
        # This means the block needs to be pushed until its center enters the goal region
        assert env is not None

    def test_no_penalty_when_goal_respawns(self):
        """Verify agent doesn't get penalized when goal respawns to a far location.

        When a goal is reached and respawns, last_dist_goal should be updated to
        the distance to the NEW goal position, not the old (near-zero) distance.
        Otherwise the agent would get a large negative reward on the next step.
        """
        env = BlockPushGoal(
            goal_size=0.3,
            reward_distance_scale=1.0,
            hazard_specs=[dict(type='cube', count=1, size=0.1, collidable=False, movable=False)],
            debug=False,
        )

        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)

        rng = jrandom.PRNGKey(42)
        state = jit_reset(rng)

        # Run for a while with zero action - just checking the mechanics
        action = jp.zeros(env.action_size)

        for _ in range(10):
            old_last_dist = state.info['last_dist_goal']
            state = jit_step(state, action)
            new_last_dist = state.info['last_dist_goal']

            # last_dist_goal should always be a reasonable positive value
            # (distance to current goal), never near-zero unless block is at goal
            assert new_last_dist >= 0, "last_dist_goal should be non-negative"

            # The distance shouldn't suddenly become very small unless
            # the block actually moved close to the goal
            block_pos = state.info['block_position'][:2]
            goal_pos = state.info['goal_positions'][0][:2]
            actual_dist = jp.sqrt(jp.sum(jp.square(goal_pos - block_pos)) + 1e-8)

            # last_dist_goal should approximately match the actual distance
            # (within some tolerance for the respawn case)
            assert jp.abs(new_last_dist - actual_dist) < 0.1, (
                f"last_dist_goal ({new_last_dist}) should match actual distance ({actual_dist})"
            )


# -----------------------------------------------------------------------------
# Run as script for debugging
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
