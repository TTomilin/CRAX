"""Tests for velocity-constrained environments."""

import jax
import jax.numpy as jnp
import pytest

from brax import envs


ENV_SPECS = [
    ("ant_velocity_constrained", {"cost_mode": "binary", "velocity_threshold": 2.6222}),
    ("halfcheetah_velocity_constrained", {}),
    ("hopper_velocity_constrained", {}),
    ("walker2d_velocity_constrained", {}),
    ("swimmer_velocity_constrained", {}),
    ("humanoid_velocity_constrained", {}),
]


@pytest.mark.parametrize("env_name, env_kwargs", ENV_SPECS)
def test_velocity_env_cost_fields(env_name, env_kwargs):
    env = envs.get_environment(env_name, **env_kwargs)
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
