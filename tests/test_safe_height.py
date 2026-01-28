"""Tests for the safe_height (height-constrained humanoid) environment."""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from brax import envs
from brax.envs.safe_height import (
    SafeHeightHumanoid,
    _LEVEL_MIN_HEIGHTS,
    get_min_height_for_level,
)

LEVELS = [1, 2, 3]


# =============================================================================
# Pytest tests
# =============================================================================

def test_safe_height_creates_env():
    """Test that SafeHeightHumanoid creates successfully."""
    env = SafeHeightHumanoid()
    assert env is not None
    assert hasattr(env, "step")
    assert hasattr(env, "reset")


def test_safe_height_reset_and_step():
    """Test basic reset and step functionality."""
    env = SafeHeightHumanoid()
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    assert state.obs is not None
    assert state.reward is not None

    action = jnp.zeros(env.action_size)
    next_state = env.step(state, action)

    assert next_state.obs is not None
    assert next_state.reward is not None


def test_safe_height_has_cost_fields():
    """Test that height cost fields are present in state."""
    env = SafeHeightHumanoid()
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    required_keys = ["cost", "height", "height_violation"]
    for key_name in required_keys:
        assert key_name in state.metrics, f"Missing metrics key: {key_name}"
        assert key_name in state.info, f"Missing info key: {key_name}"

    action = jnp.zeros(env.action_size)
    next_state = env.step(state, action)

    reward_shape = jnp.shape(next_state.reward)
    assert jnp.shape(next_state.metrics["cost"]) == reward_shape
    assert jnp.shape(next_state.info["cost"]) == reward_shape


@pytest.mark.parametrize("level", LEVELS)
def test_safe_height_level_thresholds(level):
    """Test that level-specific min_height thresholds are applied correctly."""
    env = SafeHeightHumanoid(level=level)
    expected_threshold = _LEVEL_MIN_HEIGHTS[level]
    assert abs(env._min_height - expected_threshold) < 1e-4, \
        f"Level {level}: Expected min_height {expected_threshold}, got {env._min_height}"


def test_safe_height_levels_get_harder():
    """Test that higher levels have stricter (higher) min_height thresholds."""
    thresholds = [get_min_height_for_level(level) for level in LEVELS]

    # Level 1 should be easiest (lowest min_height threshold)
    # Level 3 should be hardest (highest min_height threshold)
    assert thresholds[0] < thresholds[1] < thresholds[2], \
        f"Thresholds should increase with level: {thresholds}"


def test_safe_height_custom_threshold():
    """Test that custom min_height overrides level-based thresholds."""
    custom_min_height = 1.35
    env = SafeHeightHumanoid(level=2, min_height=custom_min_height)

    assert abs(env._min_height - custom_min_height) < 1e-4, \
        f"Custom min_height should override level: expected {custom_min_height}, got {env._min_height}"


def test_safe_height_via_envs_create():
    """Test creating safe_height via brax.envs.create."""
    env = envs.create("safe_height", episode_length=100)
    assert env is not None

    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    assert state.obs is not None


def test_safe_height_via_envs_create_with_level():
    """Test creating safe_height via brax.envs.create with difficulty level."""
    for level in LEVELS:
        env = envs.create("safe_height", level=level, episode_length=100)
        assert env is not None

        key = jax.random.PRNGKey(0)
        state = env.reset(key)
        assert "cost" in state.info


def test_safe_height_via_get_environment():
    """Test creating safe_height via brax.envs.get_environment."""
    env = envs.get_environment("safe_height")
    assert env is not None

    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    assert "cost" in state.info


def test_safe_height_invalid_level():
    """Test that invalid level raises ValueError."""
    with pytest.raises(ValueError, match="Unknown level"):
        get_min_height_for_level(level=99)


def test_safe_height_visual_boundary_enabled():
    """Test that height boundary visual is added when enabled."""
    env = SafeHeightHumanoid(show_height_boundary=True)
    # Check that the geom exists in the system
    assert "height_boundary" in env.sys.geom_names


def test_safe_height_visual_boundary_disabled():
    """Test that height boundary visual is not added when disabled."""
    env = SafeHeightHumanoid(show_height_boundary=False)
    # Check that the geom does not exist in the system
    assert "height_boundary" not in env.sys.geom_names


def test_safe_height_cost_increases_when_low():
    """Test that cost increases when humanoid is below min_height threshold."""
    # Use a high min_height to ensure cost is triggered (humanoid starts low)
    env = SafeHeightHumanoid(min_height=2.0)
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    action = jnp.zeros(env.action_size)
    state = env.step(state, action)

    # Since humanoid starts lying down (~0.15m), cost should be positive
    cost = float(np.asarray(state.info["cost"]))
    assert cost > 0, f"Expected positive cost when below threshold, got {cost}"


def test_safe_height_cost_zero_when_above():
    """Test that cost is zero when humanoid is above min_height threshold."""
    # Use a very low min_height so even lying down humanoid is "above"
    env = SafeHeightHumanoid(min_height=0.01)
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    action = jnp.zeros(env.action_size)
    state = env.step(state, action)

    # With such a low threshold, there should be no violation
    cost = float(np.asarray(state.info["cost"]))
    assert cost == 0.0, f"Expected zero cost when above threshold, got {cost}"


# =============================================================================
# Main: CLI for manual testing
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test safe_height environment")
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3],
                        help="Difficulty level (1=easy, 3=hard). If not set, tests all levels.")
    parser.add_argument("--steps", type=int, default=100, help="Steps per episode")
    args = parser.parse_args()

    levels_to_test = LEVELS if args.level is None else [args.level]

    for level in levels_to_test:
        print(f"\n{'=' * 60}")
        print(f"Testing safe_height: level={level}")
        print(f"{'=' * 60}")

        env = SafeHeightHumanoid(level=level)
        min_height = get_min_height_for_level(level)

        print(f"Action size: {env.action_size}")
        print(f"Observation size: {env.observation_size}")
        print(f"Min height threshold: {min_height:.4f}")
        print(f"  (Level 1: {_LEVEL_MIN_HEIGHTS[1]:.4f}, "
              f"Level 2: {_LEVEL_MIN_HEIGHTS[2]:.4f}, "
              f"Level 3: {_LEVEL_MIN_HEIGHTS[3]:.4f})")

        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        print(f"Initial state obs shape: {state.obs.shape}")

        # Run a few steps
        print(f"\nRunning {args.steps} steps...")
        action = jnp.zeros(env.action_size)
        total_cost = 0.0
        for i in range(args.steps):
            state = env.step(state, action)
            step_cost = float(np.asarray(state.info["cost"]))
            total_cost += step_cost
            if i < 5 or i == args.steps - 1:
                height = float(np.asarray(state.info["height"]))
                print(f"  Step {i+1}: height={height:.4f}, cost={step_cost:.4f}")

        print(f"\nTotal cost over {args.steps} steps: {total_cost:.4f}")
        print(f"Average cost per step: {total_cost / args.steps:.4f}")
