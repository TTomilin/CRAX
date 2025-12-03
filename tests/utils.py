import jax
import jax.numpy as jp
import numpy as np


# --- Circular policy maker ---
def make_circular_policy(action_dim: int, thrust: float, yaw_rate: float, thrust_idx: int, yaw_idx: int):
    """
    Returns a make_inference_fn(params)->inference(obs, rng)->(action, info)
    that always outputs constant [thrust, yaw_rate] (others = 0).
    Works with your record_episode_video which JITs the policy.
    """
    base = jp.zeros((action_dim,), dtype=jp.float32)
    base = base.at[thrust_idx].set(thrust)
    base = base.at[yaw_idx].set(yaw_rate)

    def _make_inference_fn(_params):
        # Must be JAX-friendly (no Python counters/state), since it gets jitted.
        def _infer(obs, rng):
            return base, {}  # (action, extras)

        return _infer

    return _make_inference_fn


# --- Multi-directional policy maker ---
def make_multi_directional_policy(action_dim: int, thrust: float, thrust_idx: int, yaw_idx: int, steps: int, num_yaw_values: int = 5, rng_seed: int = 42):
    """
    Returns a make_inference_fn(params)->inference(obs, rng)->(action, info)
    that creates {num_yaw_values} different policies with fixed yaw values
    and periodically switches between those policies.

    Args:
        action_dim: Dimension of action space
        thrust: Constant thrust value (max thrust)
        thrust_idx: Index for thrust in action vector
        yaw_idx: Index for yaw in action vector
        steps: Total number of steps in the episode
        num_yaw_values: Number of different policies with fixed yaw values (default: 5)
        rng_seed: Random seed for sampling yaw values (default: 42)

    Creates {num_yaw_values} different policies, each with a fixed yaw value
    sampled from the range [-1, 1], and switches between them periodically.
    """
    # Sample yaw values from [-1, 1] range
    key = jax.random.PRNGKey(rng_seed)
    k_mag, k_sign = jax.random.split(key)
    mags  = jax.random.uniform(k_mag, (num_yaw_values,), minval=0.3, maxval=0.7)
    signs = jax.random.choice(k_sign, jp.array([-1.0, 1.0], dtype=jp.float32), shape=(num_yaw_values,))
    yaw_values = (mags * signs).astype(jp.float32)

    acts = []
    for y in yaw_values:
        a = jp.zeros((action_dim,), dtype=jp.float32)
        a = a.at[thrust_idx].set(thrust)
        a = a.at[yaw_idx].set(y)
        acts.append(a)
    actions = jp.stack(acts)  # (num_yaw_values, action_dim)

    change_frequency = max(1, steps // max(1, num_yaw_values))

    # Host-side step counter (Python), used only via host_callback
    host_counter = {"t": 0}

    def _make_inference_fn(_params):
        def _infer(obs, rng):
            # get current step index from host, then increment
            def _next_idx_py(_):
                t = host_counter["t"]
                host_counter["t"] = t + 1
                idx = (t // change_frequency) % actions.shape[0]
                return np.int32(idx)

            idx = jax.experimental.io_callback(
                _next_idx_py,
                jax.ShapeDtypeStruct((), jp.int32),
                np.array(0, dtype=np.int32),
            )

            return actions[idx], {}  # fixed yaw for this segment
        return _infer
    return _make_inference_fn