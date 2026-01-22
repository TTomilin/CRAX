"""Tests for the unified safe_velocity environment."""

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
from brax.envs.safe_velocity import (
    SafeVelocity,
    DEFAULT_THRESHOLDS,
    LEVEL_THRESHOLDS,
    get_threshold_for_level,
)
from run_utils import record_episode_video_simple

AGENTS = ["ant", "halfcheetah", "hopper", "humanoid", "swimmer", "walker2d"]
LEVELS = [1, 2, 3]


# =============================================================================
# Pytest tests
# =============================================================================

@pytest.mark.parametrize("agent", AGENTS)
def test_safe_velocity_creates_env(agent):
    """Test that SafeVelocity creates environments for all supported agents."""
    env = SafeVelocity(agent=agent)
    assert env is not None
    assert hasattr(env, "step")
    assert hasattr(env, "reset")


@pytest.mark.parametrize("agent", AGENTS)
def test_safe_velocity_reset_and_step(agent):
    """Test basic reset and step functionality."""
    env = SafeVelocity(agent=agent)
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    assert state.obs is not None
    assert state.reward is not None

    action = jnp.zeros(env.action_size)
    next_state = env.step(state, action)

    assert next_state.obs is not None
    assert next_state.reward is not None


@pytest.mark.parametrize("agent", AGENTS)
def test_safe_velocity_has_cost_fields(agent):
    """Test that velocity cost fields are present in state."""
    env = SafeVelocity(agent=agent)
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    required_keys = ["cost", "velocity_threshold", "velocity_violation", "velocity_value"]
    for key_name in required_keys:
        assert key_name in state.metrics, f"Missing metrics key: {key_name}"
        assert key_name in state.info, f"Missing info key: {key_name}"

    action = jnp.zeros(env.action_size)
    next_state = env.step(state, action)

    reward_shape = jnp.shape(next_state.reward)
    assert jnp.shape(next_state.metrics["cost"]) == reward_shape
    assert jnp.shape(next_state.info["cost"]) == reward_shape


@pytest.mark.parametrize("agent", AGENTS)
def test_safe_velocity_default_thresholds(agent):
    """Test that default thresholds (level 1) are applied correctly."""
    env = SafeVelocity(agent=agent)
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    expected_threshold = DEFAULT_THRESHOLDS[agent]
    actual_threshold = float(np.asarray(state.info["velocity_threshold"]))
    assert abs(actual_threshold - expected_threshold) < 1e-4, \
        f"Expected threshold {expected_threshold}, got {actual_threshold}"


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("level", LEVELS)
def test_safe_velocity_level_thresholds(agent, level):
    """Test that level-specific thresholds are applied correctly."""
    env = SafeVelocity(agent=agent, level=level)
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    expected_threshold = LEVEL_THRESHOLDS[level][agent]
    actual_threshold = float(np.asarray(state.info["velocity_threshold"]))
    assert abs(actual_threshold - expected_threshold) < 1e-4, \
        f"Level {level}: Expected threshold {expected_threshold}, got {actual_threshold}"


@pytest.mark.parametrize("agent", AGENTS)
def test_safe_velocity_levels_get_harder(agent):
    """Test that higher levels have stricter (lower) thresholds."""
    thresholds = [get_threshold_for_level(agent, level) for level in LEVELS]

    # Level 1 should be easiest (highest threshold)
    # Level 3 should be hardest (lowest threshold)
    assert thresholds[0] > thresholds[1] > thresholds[2], \
        f"Thresholds should decrease with level: {thresholds}"


def test_level_threshold_multipliers():
    """Test that level multipliers are correct: L1=1.0, L2=0.75, L3=0.5."""
    for agent in AGENTS:
        baseline = DEFAULT_THRESHOLDS[agent]

        assert abs(LEVEL_THRESHOLDS[1][agent] - baseline * 1.0) < 1e-6
        assert abs(LEVEL_THRESHOLDS[2][agent] - baseline * 0.75) < 1e-6
        assert abs(LEVEL_THRESHOLDS[3][agent] - baseline * 0.5) < 1e-6


@pytest.mark.parametrize("agent", AGENTS)
def test_safe_velocity_custom_threshold(agent):
    """Test that custom thresholds override level-based thresholds."""
    custom_threshold = 1.234
    env = SafeVelocity(agent=agent, level=2, velocity_threshold=custom_threshold)
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    actual_threshold = float(np.asarray(state.info["velocity_threshold"]))
    assert abs(actual_threshold - custom_threshold) < 1e-4, \
        f"Custom threshold should override level: expected {custom_threshold}, got {actual_threshold}"


@pytest.mark.parametrize("cost_mode", ["binary", "hinge"])
def test_safe_velocity_cost_modes(cost_mode):
    """Test both binary and hinge cost modes."""
    env = SafeVelocity(agent="ant", cost_mode=cost_mode)
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    action = jnp.ones(env.action_size) * 0.5
    for _ in range(10):
        state = env.step(state, action)

    cost = state.info["cost"]
    assert cost is not None


def test_safe_velocity_via_envs_create():
    """Test creating safe_velocity via brax.envs.create."""
    env = envs.create("safe_velocity", agent="ant", episode_length=100)
    assert env is not None

    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    assert state.obs is not None


def test_safe_velocity_via_envs_create_with_level():
    """Test creating safe_velocity via brax.envs.create with difficulty level."""
    for level in LEVELS:
        env = envs.create("safe_velocity", agent="halfcheetah", level=level, episode_length=100)
        assert env is not None

        key = jax.random.PRNGKey(0)
        state = env.reset(key)

        expected_threshold = LEVEL_THRESHOLDS[level]["halfcheetah"]
        actual_threshold = float(np.asarray(state.info["velocity_threshold"]))
        assert abs(actual_threshold - expected_threshold) < 1e-4


def test_safe_velocity_via_get_environment():
    """Test creating safe_velocity via brax.envs.get_environment."""
    env = envs.get_environment("safe_velocity", agent="halfcheetah")
    assert env is not None

    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    assert "cost" in state.info


def test_safe_velocity_invalid_agent():
    """Test that invalid agent raises ValueError."""
    with pytest.raises(ValueError, match="Unknown agent"):
        SafeVelocity(agent="invalid_agent")


def test_safe_velocity_invalid_level():
    """Test that invalid level raises ValueError."""
    with pytest.raises(ValueError, match="Unknown level"):
        get_threshold_for_level("ant", level=99)


def test_safe_velocity_invalid_cost_mode():
    """Test that invalid cost_mode raises ValueError."""
    env = SafeVelocity(agent="ant", cost_mode="invalid")
    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    action = jnp.zeros(env.action_size)
    with pytest.raises(ValueError, match="Unknown cost_mode"):
        env.step(state, action)


# =============================================================================
# Video recording helpers
# =============================================================================

def _run_video_for_agent(agent: str, level: int = 1, steps: int = 300, num_episodes: int = 1):
    """Helper to record video for a specific agent and level."""
    env = SafeVelocity(agent=agent, level=level)
    return record_episode_video_simple(
        env,
        steps=steps,
        action_mode="periodic",
        out_name=f"safe_velocity_{agent}_level{level}",
        show_metrics=True,
        extra_metrics=["velocity_value", "velocity_violation"],
        num_episodes=num_episodes,
        fps=50,
    )


# =============================================================================
# Main: CLI for manual testing and video generation
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test safe_velocity environments")
    parser.add_argument("--agent", type=str, default="ant",
                        choices=AGENTS + ["all"],
                        help="Agent to test (or 'all' for all agents)")
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3],
                        help="Difficulty level (1=easy, 3=hard). If not set, tests all levels.")
    parser.add_argument("--steps", type=int, default=300, help="Steps per episode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--video", action="store_true", help="Record video")
    args = parser.parse_args()

    agents_to_test = AGENTS if args.agent == "all" else [args.agent]
    levels_to_test = LEVELS if args.level is None else [args.level]

    for agent in agents_to_test:
        for level in levels_to_test:
            print(f"\n{'=' * 60}")
            print(f"Testing safe_velocity: agent={agent}, level={level}")
            print(f"{'=' * 60}")

            env = SafeVelocity(agent=agent, level=level)
            threshold = get_threshold_for_level(agent, level)

            print(f"Action size: {env.action_size}")
            print(f"Observation size: {env.observation_size}")
            print(f"Velocity threshold: {threshold:.4f}")
            print(f"  (Level 1: {LEVEL_THRESHOLDS[1][agent]:.4f}, "
                  f"Level 2: {LEVEL_THRESHOLDS[2][agent]:.4f}, "
                  f"Level 3: {LEVEL_THRESHOLDS[3][agent]:.4f})")

            key = jax.random.PRNGKey(42)
            state = env.reset(key)
            print(f"Initial state obs shape: {state.obs.shape}")

            if args.video:
                print(f"\nRecording video for {agent} level {level}...")
                video_path = _run_video_for_agent(
                    agent, level=level, steps=args.steps, num_episodes=args.episodes
                )
                print(f"Video saved to: {video_path}")
