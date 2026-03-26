"""Tests for the PixelObservationWrapper.

Tests cover:
- Basic functionality (reset, step, observation shapes)
- jax.jit compatibility
- jax.vmap compatibility (as used by VmapWrapper)
- jax.jit(jax.vmap(...)) composition (as used in training)
- Full training wrapper stack (VmapWrapper + EpisodeWrapper + AutoResetWrapper)
- Frame stacking under jit/vmap
- Multi-camera rendering
- Grayscale mode
- Observation modes (pixels, pixels+state, state passthrough)
- Vision network factory with cost_value_network
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brax import envs
from brax.envs.wrappers.pixel_observation import PixelObservationWrapper
from brax.envs.wrappers import training


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(vision_kwargs=None, level=1):
    """Create a safe_goal_point env with vision wrapper."""
    vk = dict(cameras=('vision',), height=64, width=64, obs_mode='pixels+state')
    if vision_kwargs:
        vk.update(vision_kwargs)
    return envs.get_environment(
        'safe_goal_point', level=level, vision=True, vision_kwargs=vk,
    )


def _make_env_no_registry(vision_kwargs=None, level=1):
    """Create the wrapper manually to control the wrapper stack."""
    base = envs.get_environment('safe_goal_point', level=level)
    vk = dict(cameras=('vision',), height=64, width=64, obs_mode='pixels+state')
    if vision_kwargs:
        vk.update(vision_kwargs)
    return PixelObservationWrapper(base, **vk)


# ===========================================================================
# 1. Basic functionality (no jit/vmap)
# ===========================================================================

class TestBasicFunctionality:
    def test_reset_returns_dict_obs(self):
        env = _make_env()
        state = env.reset(jax.random.PRNGKey(0))
        assert isinstance(state.obs, dict)
        assert 'pixels/vision' in state.obs
        assert 'state' in state.obs

    def test_pixel_shape(self):
        env = _make_env()
        state = env.reset(jax.random.PRNGKey(0))
        assert state.obs['pixels/vision'].shape == (64, 64, 3)

    def test_pixel_values_range(self):
        env = _make_env()
        state = env.reset(jax.random.PRNGKey(0))
        pixels = state.obs['pixels/vision']
        assert pixels.dtype == jnp.float32
        assert float(pixels.min()) >= 0.0
        assert float(pixels.max()) <= 1.0

    def test_pixels_are_not_all_zero(self):
        """Ensure the renderer actually produces non-trivial images."""
        env = _make_env()
        state = env.reset(jax.random.PRNGKey(0))
        assert float(state.obs['pixels/vision'].sum()) > 0.0

    def test_state_obs_preserved(self):
        """State vector should match the inner env's observation."""
        env = _make_env()
        state = env.reset(jax.random.PRNGKey(0))
        inner_size = env.env.observation_size  # UnifiedEnvAdapter's obs size
        assert state.obs['state'].shape == (inner_size,)

    def test_step_returns_dict_obs(self):
        env = _make_env()
        state = env.reset(jax.random.PRNGKey(0))
        action = jnp.zeros(env.action_size)
        next_state = env.step(state, action)
        assert isinstance(next_state.obs, dict)
        assert 'pixels/vision' in next_state.obs
        assert next_state.obs['pixels/vision'].shape == (64, 64, 3)

    def test_cost_preserved(self):
        env = _make_env()
        state = env.reset(jax.random.PRNGKey(0))
        action = jnp.zeros(env.action_size)
        next_state = env.step(state, action)
        assert 'cost' in next_state.info

    def test_observation_size_property(self):
        env = _make_env()
        obs_size = env.observation_size
        assert isinstance(obs_size, dict)
        assert obs_size['pixels/vision'] == (64, 64, 3)
        assert 'state' in obs_size


# ===========================================================================
# 2. Observation modes
# ===========================================================================

class TestObservationModes:
    def test_pixels_only_mode(self):
        env = _make_env(dict(obs_mode='pixels'))
        state = env.reset(jax.random.PRNGKey(0))
        assert isinstance(state.obs, dict)
        assert 'pixels/vision' in state.obs
        assert 'state' not in state.obs

    def test_state_passthrough_mode(self):
        env = _make_env(dict(obs_mode='state'))
        state = env.reset(jax.random.PRNGKey(0))
        # Should return the inner env's flat array obs
        assert not isinstance(state.obs, dict)
        assert isinstance(state.obs, jax.Array)

    def test_state_passthrough_observation_size(self):
        env = _make_env(dict(obs_mode='state'))
        assert isinstance(env.observation_size, int)


# ===========================================================================
# 3. Multi-camera and grayscale
# ===========================================================================

class TestCameraOptions:
    def test_multi_camera(self):
        env = _make_env(dict(cameras=('vision', 'vision_back')))
        state = env.reset(jax.random.PRNGKey(0))
        assert 'pixels/vision' in state.obs
        assert 'pixels/vision_back' in state.obs
        assert state.obs['pixels/vision'].shape == (64, 64, 3)
        assert state.obs['pixels/vision_back'].shape == (64, 64, 3)

    def test_grayscale(self):
        env = _make_env(dict(grayscale=True))
        state = env.reset(jax.random.PRNGKey(0))
        assert state.obs['pixels/vision'].shape == (64, 64, 1)
        obs_size = env.observation_size
        assert obs_size['pixels/vision'] == (64, 64, 1)


# ===========================================================================
# 4. Frame stacking
# ===========================================================================

class TestFrameStacking:
    def test_frame_stack_shape(self):
        env = _make_env(dict(frame_stack=3))
        state = env.reset(jax.random.PRNGKey(0))
        assert state.obs['pixels/vision'].shape == (64, 64, 9)  # 3 * 3
        assert env.observation_size['pixels/vision'] == (64, 64, 9)

    def test_frame_stack_updates_on_step(self):
        env = _make_env(dict(frame_stack=2))
        state = env.reset(jax.random.PRNGKey(0))
        action = jnp.zeros(env.action_size)
        # After reset, both frames should be identical
        pixels = state.obs['pixels/vision']
        frame0 = pixels[..., :3]
        frame1 = pixels[..., 3:]
        np.testing.assert_array_equal(np.asarray(frame0), np.asarray(frame1))
        # After a step, frames should differ (new frame appended)
        next_state = env.step(state, action)
        pixels2 = next_state.obs['pixels/vision']
        # frame1 of old should equal frame0 of new (shifted left)
        np.testing.assert_array_equal(
            np.asarray(frame1), np.asarray(pixels2[..., :3])
        )

    def test_frame_stack_grayscale(self):
        env = _make_env(dict(frame_stack=4, grayscale=True))
        state = env.reset(jax.random.PRNGKey(0))
        assert state.obs['pixels/vision'].shape == (64, 64, 4)  # 4 * 1


# ===========================================================================
# 5. jax.jit compatibility
# ===========================================================================

class TestJit:
    def test_jit_reset(self):
        env = _make_env()
        jit_reset = jax.jit(env.reset)
        state = jit_reset(jax.random.PRNGKey(0))
        assert state.obs['pixels/vision'].shape == (64, 64, 3)

    def test_jit_step(self):
        env = _make_env()
        state = env.reset(jax.random.PRNGKey(0))
        jit_step = jax.jit(env.step)
        action = jnp.zeros(env.action_size)
        next_state = jit_step(state, action)
        assert next_state.obs['pixels/vision'].shape == (64, 64, 3)

    def test_jit_multiple_steps(self):
        """Verify jitted step can be called repeatedly."""
        env = _make_env()
        state = env.reset(jax.random.PRNGKey(0))
        jit_step = jax.jit(env.step)
        action = jnp.zeros(env.action_size)
        for _ in range(5):
            state = jit_step(state, action)
        assert state.obs['pixels/vision'].shape == (64, 64, 3)

    def test_jit_with_frame_stack(self):
        env = _make_env(dict(frame_stack=3))
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)
        state = jit_reset(jax.random.PRNGKey(0))
        assert state.obs['pixels/vision'].shape == (64, 64, 9)
        action = jnp.zeros(env.action_size)
        state = jit_step(state, action)
        assert state.obs['pixels/vision'].shape == (64, 64, 9)


# ===========================================================================
# 6. jax.vmap compatibility
# ===========================================================================

class TestVmap:
    def test_vmap_reset(self):
        """vmap over reset, as VmapWrapper does."""
        env = _make_env()
        batch_size = 4
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
        vmap_reset = jax.vmap(env.reset)
        state = vmap_reset(keys)
        assert state.obs['pixels/vision'].shape == (batch_size, 64, 64, 3)
        assert state.obs['state'].shape[0] == batch_size

    def test_vmap_step(self):
        """vmap over step, as VmapWrapper does."""
        env = _make_env()
        batch_size = 4
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
        vmap_reset = jax.vmap(env.reset)
        vmap_step = jax.vmap(env.step)
        state = vmap_reset(keys)
        actions = jnp.zeros((batch_size, env.action_size))
        next_state = vmap_step(state, actions)
        assert next_state.obs['pixels/vision'].shape == (batch_size, 64, 64, 3)


# ===========================================================================
# 7. jax.jit(jax.vmap(...)) — the actual training pattern
# ===========================================================================

class TestJitVmap:
    def test_jit_vmap_reset(self):
        """This is the exact pattern used in PPO training: jax.jit(jax.vmap(env.reset))."""
        env = _make_env()
        batch_size = 4
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
        jit_vmap_reset = jax.jit(jax.vmap(env.reset))
        state = jit_vmap_reset(keys)
        assert state.obs['pixels/vision'].shape == (batch_size, 64, 64, 3)
        assert state.obs['state'].shape[0] == batch_size

    def test_jit_vmap_step(self):
        """jit(vmap(step)) — the core training step pattern."""
        env = _make_env()
        batch_size = 4
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
        jit_vmap_reset = jax.jit(jax.vmap(env.reset))
        jit_vmap_step = jax.jit(jax.vmap(env.step))
        state = jit_vmap_reset(keys)
        actions = jnp.zeros((batch_size, env.action_size))
        next_state = jit_vmap_step(state, actions)
        assert next_state.obs['pixels/vision'].shape == (batch_size, 64, 64, 3)

    def test_jit_vmap_multiple_steps(self):
        """Multiple jit(vmap(step)) calls in sequence."""
        env = _make_env()
        batch_size = 4
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
        jit_vmap_reset = jax.jit(jax.vmap(env.reset))
        jit_vmap_step = jax.jit(jax.vmap(env.step))
        state = jit_vmap_reset(keys)
        actions = jnp.zeros((batch_size, env.action_size))
        for _ in range(5):
            state = jit_vmap_step(state, actions)
        assert state.obs['pixels/vision'].shape == (batch_size, 64, 64, 3)

    def test_jit_vmap_with_frame_stack(self):
        """Frame stacking under jit(vmap(...))."""
        env = _make_env(dict(frame_stack=3))
        batch_size = 4
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
        jit_vmap_reset = jax.jit(jax.vmap(env.reset))
        jit_vmap_step = jax.jit(jax.vmap(env.step))
        state = jit_vmap_reset(keys)
        assert state.obs['pixels/vision'].shape == (batch_size, 64, 64, 9)
        actions = jnp.zeros((batch_size, env.action_size))
        state = jit_vmap_step(state, actions)
        assert state.obs['pixels/vision'].shape == (batch_size, 64, 64, 9)


# ===========================================================================
# 8. Full training wrapper stack
# ===========================================================================

class TestTrainingWrapperStack:
    def test_full_wrapper_stack(self):
        """Test with the full training wrapper stack:
        PixelObsWrapper -> VmapWrapper -> EpisodeWrapper -> AutoResetWrapper.
        This is what the actual training loop creates.
        """
        base_env = _make_env()
        wrapped = training.wrap(base_env, episode_length=100, action_repeat=1)
        # wrapped = AutoResetWrapper(EpisodeWrapper(VmapWrapper(base_env)))

        batch_size = 4
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
        # The training loop does jax.jit(jax.vmap(env.reset)) on the already-
        # VmapWrapper'd env, which gives an extra batch dim for devices.
        # But here we just test with the wrapper's own batching.
        state = wrapped.reset(keys)

        # VmapWrapper adds batch dim
        assert state.obs['pixels/vision'].shape == (batch_size, 64, 64, 3)
        assert 'steps' in state.info  # From EpisodeWrapper
        assert 'first_obs' in state.info  # From AutoResetWrapper

        actions = jnp.zeros((batch_size, wrapped.action_size))
        next_state = wrapped.step(state, actions)
        assert next_state.obs['pixels/vision'].shape == (batch_size, 64, 64, 3)
        assert 'cost' in next_state.info

    def test_full_stack_jitted(self):
        """Full stack under jax.jit — the realistic training scenario."""
        base_env = _make_env()
        wrapped = training.wrap(base_env, episode_length=100, action_repeat=1)

        batch_size = 4
        keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

        @jax.jit
        def do_reset(keys):
            return wrapped.reset(keys)

        @jax.jit
        def do_step(state, actions):
            return wrapped.step(state, actions)

        state = do_reset(keys)
        assert state.obs['pixels/vision'].shape == (batch_size, 64, 64, 3)

        actions = jnp.zeros((batch_size, wrapped.action_size))
        for _ in range(3):
            state = do_step(state, actions)
        assert state.obs['pixels/vision'].shape == (batch_size, 64, 64, 3)


# ===========================================================================
# 9. Vision network factory with cost value
# ===========================================================================

class TestVisionNetworks:
    def test_make_ppo_networks_vision_with_cost_value(self):
        """Verify the vision network factory creates all 3 networks."""
        from brax.training.agents.ppo.networks_vision import make_ppo_networks_vision

        obs_size = {
            'pixels/vision': (64, 64, 3),
            'state': (62,),
        }
        networks = make_ppo_networks_vision(
            observation_size=obs_size,
            action_size=2,
            cost_value_hidden_layer_sizes=(256,) * 5,
            policy_obs_key='state',
            value_obs_key='state',
        )
        assert networks.policy_network is not None
        assert networks.value_network is not None
        assert networks.cost_value_network is not None

    def test_vision_network_forward_pass(self):
        """Verify the networks can do a forward pass with pixel obs."""
        from brax.training.agents.ppo.networks_vision import make_ppo_networks_vision

        obs_size = {
            'pixels/vision': (64, 64, 3),
            'state': (62,),
        }
        # Use empty state_obs_key to avoid normalizer dependency in this unit test.
        # In real training, the normalizer is properly initialized.
        networks = make_ppo_networks_vision(
            observation_size=obs_size,
            action_size=2,
            cost_value_hidden_layer_sizes=(256, 256),
        )

        # Initialize params
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        policy_params = networks.policy_network.init(k1)
        value_params = networks.value_network.init(k2)
        cost_value_params = networks.cost_value_network.init(k3)

        # Create dummy observation
        dummy_obs = {
            'pixels/vision': jnp.zeros((1, 64, 64, 3)),
            'state': jnp.zeros((1, 62)),
        }

        # Forward pass — no state_obs_key means no normalizer needed
        policy_out = networks.policy_network.apply(
            None, policy_params, dummy_obs
        )
        value_out = networks.value_network.apply(
            None, value_params, dummy_obs
        )
        cost_out = networks.cost_value_network.apply(
            None, cost_value_params, dummy_obs
        )

        assert policy_out.shape == (1, 4)  # 2 actions * 2 (mean + log_std)
        assert value_out.shape == (1,)
        assert cost_out.shape == (1,)

    def test_make_vision_network_factory(self):
        """Test the run_utils helper for safe RL algorithms."""
        from run_utils import make_vision_network_factory

        factory = make_vision_network_factory(
            'ppo_lag', policy_obs_key='state', value_obs_key='state',
        )
        obs_size = {'pixels/vision': (64, 64, 3), 'state': (62,)}
        networks = factory(obs_size, action_size=2)
        assert networks.cost_value_network is not None

    def test_make_vision_network_factory_plain_ppo(self):
        """Plain PPO should not get a cost value network."""
        from run_utils import make_vision_network_factory

        factory = make_vision_network_factory(
            'ppo', policy_obs_key='state', value_obs_key='state',
        )
        obs_size = {'pixels/vision': (64, 64, 3), 'state': (62,)}
        networks = factory(obs_size, action_size=2)
        assert networks.cost_value_network is None


# ===========================================================================
# 10. Error handling
# ===========================================================================

class TestErrorHandling:
    def test_invalid_camera_name(self):
        with pytest.raises(ValueError, match="Camera .* not found"):
            _make_env(dict(cameras=('nonexistent_camera',)))

    def test_invalid_obs_mode(self):
        with pytest.raises(ValueError, match="obs_mode must be"):
            _make_env(dict(obs_mode='invalid'))


# ===========================================================================
# Run with pytest
# ===========================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
