"""Tests for velocity-constrained environments using the unified safe_velocity env."""

import jax
import jax.numpy as jnp
import pytest

from brax import envs


# Test the unified safe_velocity environment with different agents
AGENT_SPECS = [
    ("ant", {"cost_mode": "binary"}),
    ("ant", {"cost_mode": "hinge"}),
    ("halfcheetah", {}),
    ("hopper", {}),
    ("walker2d", {}),
    ("swimmer", {}),
    ("humanoid", {}),
]


@pytest.mark.parametrize("agent, env_kwargs", AGENT_SPECS)
def test_velocity_env_cost_fields(agent, env_kwargs):
    """Test that velocity cost fields are present in metrics and info."""
    env = envs.get_environment("safe_velocity", agent=agent, **env_kwargs)
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    for key_name in ("cost", "velocity_threshold", "velocity_violation", "velocity_value"):
        assert key_name in state.metrics, f"Missing metrics key: {key_name}"
        assert key_name in state.info, f"Missing info key: {key_name}"

    action = jnp.zeros(env.action_size)
    next_state = env.step(state, action)

    reward_shape = jnp.shape(next_state.reward)
    assert jnp.shape(next_state.metrics["cost"]) == reward_shape
    assert jnp.shape(next_state.info["cost"]) == reward_shape


@pytest.mark.parametrize("agent", ["ant", "halfcheetah", "hopper", "walker2d", "swimmer", "humanoid"])
def test_velocity_env_step_count(agent):
    """Test that step_count increments correctly."""
    env = envs.get_environment("safe_velocity", agent=agent)
    key = jax.random.PRNGKey(0)
    state = env.reset(key)

    assert state.info.get("step_count", 0) == 0

    action = jnp.zeros(env.action_size)
    for expected_step in range(1, 5):
        state = env.step(state, action)
        assert state.info["step_count"] == expected_step
