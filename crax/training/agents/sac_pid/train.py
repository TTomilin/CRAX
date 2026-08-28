"""SAC-PID Lagrange training.

Thin wrapper around SAC-Lag that replaces the simple gradient-ascent lambda
update with a PID controller, mirroring the PPO-PID / PPO-Lag relationship.

PID update (per gradient step):
  v          = mean_cost - per_step_safety_bound
  integral   = clip(integral + v, -pid_integral_clip, pid_integral_clip)
  deriv_ema  = beta * deriv_ema + (1 - beta) * (v - prev_v)
  pid_output = kp * v + ki * integral + kd * deriv_ema
  lambda     = clip(relu(lambda + pid_output), 0, pid_lambda_clip)
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from crax import base, envs
from crax.training import types
from crax.training.agents.sac_lag import networks as sac_lag_networks
from crax.training.agents.sac_lag.train import train as sac_lag_train

Metrics = types.Metrics


def train(
    environment: envs.Env,
    num_timesteps: int,
    episode_length: int,
    wrap_env: bool = True,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    discounting: float = 0.99,
    seed: int = 0,
    batch_size: int = 256,
    num_evals: int = 0,
    normalize_observations: bool = False,
    max_devices_per_host: Optional[int] = None,
    reward_scaling: float = 1.0,
    tau: float = 0.005,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    grad_updates_per_step: int = 32,
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[
        sac_lag_networks.SACLagNetworks
    ] = sac_lag_networks.make_sac_lag_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    training_metrics_steps: int = 1_000_000,
    eval_env: Optional[envs.Env] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    # Vision (pixel-obs) training -- see sac_lag.train.train.
    vision_kwargs: Optional[Dict[str, Any]] = None,
    augment_pixels: bool = False,
    # Constraint
    safety_bound: float = 25.0,
    initial_lambda: float = 0.0,
    # PID gains
    pid_kp: float = 10.0,
    pid_ki: float = 0.01,
    pid_kd: float = 0.01,
    pid_integral_clip: float = 1.0,
    pid_lambda_clip: float = 1e6,
    pid_deriv_ema_beta: float = 0.95,
):
    """SAC-PID Lagrange training."""
    per_step_safety_bound = safety_bound / episode_length if episode_length else safety_bound

    def pid_lambda_update(lambda_lagr, aux_state, mean_cost):
        integral, prev_v, deriv_ema = aux_state
        v = mean_cost - per_step_safety_bound
        new_integral = jnp.clip(integral + v, -pid_integral_clip, pid_integral_clip)
        new_deriv_ema = (
            pid_deriv_ema_beta * deriv_ema
            + (1.0 - pid_deriv_ema_beta) * (v - prev_v)
        )
        pid_output = pid_kp * v + pid_ki * new_integral + pid_kd * new_deriv_ema
        new_lambda = jnp.clip(jax.nn.relu(lambda_lagr + pid_output), 0.0, pid_lambda_clip)
        new_aux_state = (new_integral, v, new_deriv_ema)
        metrics = {
            'lambda_lagr': new_lambda,
            'pid/violation': v,
            'pid/integral': new_integral,
            'pid/deriv_ema': new_deriv_ema,
        }
        return new_lambda, new_aux_state, metrics

    def init_aux_state_fn():
        # (integral, prev_v, deriv_ema) — all zeros
        return (
            jnp.zeros((), dtype=jnp.float32),
            jnp.zeros((), dtype=jnp.float32),
            jnp.zeros((), dtype=jnp.float32),
        )

    return sac_lag_train(
        environment=environment,
        num_timesteps=num_timesteps,
        episode_length=episode_length,
        wrap_env=wrap_env,
        wrap_env_fn=wrap_env_fn,
        action_repeat=action_repeat,
        num_envs=num_envs,
        num_eval_envs=num_eval_envs,
        learning_rate=learning_rate,
        discounting=discounting,
        seed=seed,
        batch_size=batch_size,
        num_evals=num_evals,
        normalize_observations=normalize_observations,
        max_devices_per_host=max_devices_per_host,
        reward_scaling=reward_scaling,
        tau=tau,
        min_replay_size=min_replay_size,
        max_replay_size=max_replay_size,
        grad_updates_per_step=grad_updates_per_step,
        deterministic_eval=deterministic_eval,
        network_factory=network_factory,
        progress_fn=progress_fn,
        training_metrics_steps=training_metrics_steps,
        eval_env=eval_env,
        randomization_fn=randomization_fn,
        vision_kwargs=vision_kwargs,
        augment_pixels=augment_pixels,
        safety_bound=safety_bound,
        initial_lambda=initial_lambda,
        lambda_update_fn=pid_lambda_update,
        init_aux_state_fn=init_aux_state_fn,
    )
