"""Tests for the SafeSpider environment."""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from brax import envs
from brax.envs.safe_spider import SafeSpider
from run_utils import record_episode_video_simple

LEVELS = [1, 2, 3]


# =============================================================================
# Pytest tests
# =============================================================================

def test_safe_spider_creates_env():
    """Test that SafeSpider creates environments."""
    env = SafeSpider()
    assert env is not None
    assert hasattr(env, "step")
    assert hasattr(env, "reset")


@pytest.mark.parametrize("level", LEVELS)
def test_safe_spider_difficulty_levels(level):
    """Test all difficulty levels."""
    env = SafeSpider(difficulty=level)
    assert env is not None
    assert env.difficulty == level

    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    assert state.obs is not None


def test_safe_spider_reset_and_step():
    """Test basic reset and step functionality."""
    env = SafeSpider()
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    assert state.obs is not None
    assert state.reward is not None

    action = jnp.zeros(env.action_size)
    next_state = env.step(state, action)

    assert next_state.obs is not None
    assert next_state.reward is not None


def test_safe_spider_has_cost_fields():
    """Test that cost fields are present in state."""
    env = SafeSpider()
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    assert "cost" in state.metrics, "Missing metrics key: cost"
    assert "cost" in state.info, "Missing info key: cost"
    assert "feet_on_ground" in state.metrics, "Missing metrics key: feet_on_ground"

    action = jnp.zeros(env.action_size)
    next_state = env.step(state, action)

    reward_shape = jnp.shape(next_state.reward)
    assert jnp.shape(next_state.metrics["cost"]) == reward_shape
    assert jnp.shape(next_state.info["cost"]) == reward_shape


@pytest.mark.parametrize("level", LEVELS)
def test_safe_spider_restricted_feet_by_level(level):
    """Test that the correct number of feet are restricted per level."""
    env = SafeSpider(difficulty=level)

    expected_counts = {1: 2, 2: 3, 3: 4}
    assert len(env.restricted_feet) == expected_counts[level]


def test_safe_spider_level1_restrictions():
    """Test Level 1 restricts diagonal opposite legs."""
    env = SafeSpider(difficulty=1)
    assert set(env.restricted_feet) == {'front_left', 'back_right'}


def test_safe_spider_level2_restrictions():
    """Test Level 2 restricts alternating tripod."""
    env = SafeSpider(difficulty=2)
    assert set(env.restricted_feet) == {'front_left', 'mid_right', 'back_left'}


def test_safe_spider_level3_restrictions():
    """Test Level 3 restricts all but mid legs."""
    env = SafeSpider(difficulty=3)
    assert set(env.restricted_feet) == {'front_left', 'front_right', 'back_left', 'back_right'}


def test_safe_spider_has_six_legs():
    """Test that the spider has 6 foot geoms defined."""
    env = SafeSpider()
    assert len(SafeSpider.FOOT_GEOMS) == 6


def test_safe_spider_action_size():
    """Test that action size is 12 (2 joints per leg, 6 legs)."""
    env = SafeSpider()
    assert env.action_size == 12


def test_safe_spider_via_envs_create():
    """Test creating safe_spider via brax.envs.create."""
    env = envs.create("safe_spider", episode_length=100)
    assert env is not None

    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    assert state.obs is not None


def test_safe_spider_via_envs_create_with_level():
    """Test creating safe_spider via brax.envs.create with difficulty level."""
    for level in LEVELS:
        env = envs.create("safe_spider", level=level, episode_length=100)
        assert env is not None

        key = jax.random.PRNGKey(0)
        state = env.reset(key)
        assert "cost" in state.info


def test_safe_spider_via_get_environment():
    """Test creating safe_spider via brax.envs.get_environment."""
    env = envs.get_environment("safe_spider")
    assert env is not None

    key = jax.random.PRNGKey(0)
    state = env.reset(key)
    assert "cost" in state.info


def test_safe_spider_invalid_difficulty():
    """Test that invalid difficulty raises ValueError."""
    with pytest.raises(AssertionError):
        SafeSpider(difficulty=99)


# =============================================================================
# Video recording helpers
# =============================================================================

def _run_video_for_level(level: int, steps: int = 300, num_episodes: int = 1):
    """Helper to record video for a specific difficulty level."""
    env = SafeSpider(difficulty=level)
    return record_episode_video_simple(
        env,
        steps=steps,
        action_mode="periodic",
        out_name=f"safe_spider_level{level}",
        show_metrics=True,
        extra_metrics=["feet_on_ground"],
        num_episodes=num_episodes,
        fps=50,
    )


# =============================================================================
# Main: CLI for manual testing and video generation
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SafeSpider environment")
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3],
                        help="Difficulty level (1=easy, 3=hard). If not set, tests all levels.")
    parser.add_argument("--steps", type=int, default=300, help="Steps per episode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    args = parser.parse_args()

    levels_to_test = LEVELS if args.level is None else [args.level]

    for level in levels_to_test:
        print(f"\n{'=' * 60}")
        print(f"Testing SafeSpider: difficulty={level}")
        print(f"{'=' * 60}")

        env = SafeSpider(difficulty=level)

        print(f"Action size: {env.action_size}")
        print(f"Observation size: {env.observation_size}")
        print(f"Restricted feet: {env.restricted_feet}")
        print(f"Number of restricted feet: {len(env.restricted_feet)}")

        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        print(f"Initial state obs shape: {state.obs.shape}")

        print(f"\nRecording video for level {level}...")
        video_path = _run_video_for_level(
            level, steps=args.steps, num_episodes=args.episodes
        )
        print(f"Video saved to: {video_path}")
