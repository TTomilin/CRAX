"""Tests for the SafePointButton environment."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent))

from brax import envs
from brax.envs.safe_point_button import SafePointButton


def test_safe_point_button_creates_env():
    """Basic construction sanity check."""
    env = SafePointButton()
    assert env is not None
    assert hasattr(env, "reset")
    assert hasattr(env, "step")


def test_safe_point_button_obs_shape():
    """Observation size matches reported size."""
    env = SafePointButton()
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    assert state.obs is not None
    assert jnp.shape(state.obs)[-1] == env.observation_size


def test_safe_point_button_cost_fields():
    """Cost is available in info and metrics."""
    env = SafePointButton()
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    assert "cost" in state.info
    assert "cost" in state.metrics

    action = jnp.zeros(env.action_size)
    next_state = env.step(state, action)

    reward_shape = jnp.shape(next_state.reward)
    assert jnp.shape(next_state.metrics["cost"]) == reward_shape
    assert jnp.shape(next_state.info["cost"]) == reward_shape


def test_safe_point_button_via_get_environment():
    """Test registry and UnifiedEnvAdapter wiring."""
    env = envs.get_environment("safe_point_button")
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    assert "cost" in state.info


def test_safe_point_button_reward_closer_positive():
    """Moving closer to active button yields positive reward."""
    env = SafePointButton(button_count=4, reward_distance_scale=1.0)
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    # Get active button position
    active_idx = state.info["active_button_idx"]
    button_positions = state.info["button_positions"]
    active_button_xy = button_positions[active_idx, :2]

    # Move agent closer to active button
    agent_pos = state.pipeline_state.xpos[env._agent_body]
    direction_to_button = active_button_xy - agent_pos[:2]
    direction_to_button = direction_to_button / (jnp.linalg.norm(direction_to_button) + 1e-8)

    # Small action toward button
    action = jnp.array([direction_to_button[0] * 0.1, direction_to_button[1] * 0.1])
    next_state = env.step(state, action)

    # Reward should be positive (or at least distance should decrease)
    assert next_state.reward >= 0.0 or state.info["last_dist_goal"] > next_state.info["last_dist_goal"]


def test_safe_point_button_button_info():
    """Button-related info fields are present."""
    env = SafePointButton()
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    assert "button_positions" in state.info
    assert "active_button_idx" in state.info
    assert "button_timer" in state.info
    assert jnp.shape(state.info["button_positions"]) == (env._button_count, 3)
    assert 0 <= state.info["active_button_idx"] < env._button_count


def test_safe_point_button_difficulty_overrides():
    """Difficulty levels apply correct overrides."""
    # Level 1 should have hazards and constrained buttons
    env = envs.get_environment("safe_point_button", level=1)
    base_env = env.unwrapped
    assert base_env._buttons_constrained == True
    assert base_env._num_hazards > 0

    # Level 2 should have more hazards
    env = envs.get_environment("safe_point_button", level=2)
    base_env = env.unwrapped
    assert base_env._buttons_constrained == True
    assert base_env._num_hazards >= 8  # At least 8 hazards in level 2
