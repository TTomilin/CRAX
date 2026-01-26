"""Tests for the SafePointCircle environment."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).parent.parent))

from brax import envs
from brax.envs.safe_point_circle import SafePointCircle


def test_safe_point_circle_creates_env():
    """Basic construction sanity check."""
    env = SafePointCircle()
    assert env is not None
    assert hasattr(env, "reset")
    assert hasattr(env, "step")


def test_safe_point_circle_obs_shape():
    """Observation size matches reported size."""
    env = SafePointCircle()
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    assert state.obs is not None
    assert jnp.shape(state.obs)[-1] == env.observation_size


def test_safe_point_circle_reward_ccw_positive():
    """Counter-clockwise tangential velocity yields positive reward."""
    env = SafePointCircle(circle_radius=1.0)

    qpos = env.sys.qpos0
    qpos = qpos.at[0].set(1.0)
    qpos = qpos.at[1].set(0.0)
    qpos = qpos.at[2].set(0.0)

    qvel = jnp.zeros(env.sys.nv)
    qvel = qvel.at[0].set(0.0)
    qvel = qvel.at[1].set(1.0)

    data = env.pipeline_init(qpos, qvel)
    reward, _, tangent_vel, _ = env._calculate_orbit_components(data)

    assert tangent_vel > 0.0
    assert reward > 0.0


def test_safe_point_circle_cost_fields():
    """Cost is available in info and metrics."""
    env = SafePointCircle()
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    assert "cost" in state.info
    assert "cost" in state.metrics

    action = jnp.zeros(env.action_size)
    next_state = env.step(state, action)

    reward_shape = jnp.shape(next_state.reward)
    assert jnp.shape(next_state.metrics["cost"]) == reward_shape
    assert jnp.shape(next_state.info["cost"]) == reward_shape


def test_safe_point_circle_via_get_environment():
    """Test registry and UnifiedEnvAdapter wiring."""
    env = envs.get_environment("safe_point_circle")
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    assert "cost" in state.info


def test_safe_point_circle_difficulty_overrides():
    """Difficulty levels map to boundary constraints."""
    env = envs.get_environment("safe_point_circle", level=1)
    base_env = env.unwrapped
    assert base_env._boundary_x is None
    assert base_env._boundary_y is None

    env = envs.get_environment("safe_point_circle", level=2)
    base_env = env.unwrapped
    assert base_env._boundary_x == 1.125
    assert base_env._boundary_y is None

    env = envs.get_environment("safe_point_circle", level=3)
    base_env = env.unwrapped
    assert base_env._boundary_x == 1.125
    assert base_env._boundary_y == 1.125
