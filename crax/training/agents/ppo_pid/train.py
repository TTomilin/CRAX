"""PPO-PID Lagrange training.

Thin wrapper around the base PPO trainer with PID-based Lagrangian constraint handling.
"""

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from crax import base
from crax import envs
from crax.training import types
from crax.training.agents.ppo import train as ppo_train
from crax.training.agents.ppo import networks as ppo_networks
from crax.training.agents.ppo_lag import losses as ppo_lag_losses

Metrics = types.Metrics
TrainingState = ppo_train.TrainingState


def train(
        environment: envs.Env,
        num_timesteps: int,
        episode_length: int,
        max_devices_per_host: Optional[int] = None,
        wrap_env: bool = True,
        augment_pixels: bool = False,
        num_envs: int = 1,
        action_repeat: int = 1,
        wrap_env_fn: Optional[Callable[[Any], Any]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        learning_rate: float = 1e-4,
        entropy_cost: float = 1e-4,
        discounting: float = 0.9,
        unroll_length: int = 10,
        batch_size: int = 32,
        num_minibatches: int = 16,
        num_updates_per_batch: int = 2,
        num_resets_per_eval: int = 0,
        normalize_observations: bool = False,
        reward_scaling: float = 1.0,
        clipping_epsilon: float = 0.3,
        gae_lambda: float = 0.95,
        max_grad_norm: Optional[float] = None,
        normalize_advantage: bool = True,
        network_factory: types.NetworkFactory[ppo_networks.PPONetworks] = None,
        seed: int = 0,
        safety_bound: float = 0.0,
        initial_lambda_lagr: float = 0.0,
        pid_kp: float = 10.0,
        pid_ki: float = 0.01,
        pid_kd: float = 0.01,
        pid_integral_clip: float = 1.0,
        pid_lambda_clip: float = 1e6,
        pid_deriv_ema_beta: float = 0.95,
        num_evals: int = 0,
        eval_env: Optional[envs.Env] = None,
        num_eval_envs: int = 128,
        deterministic_eval: bool = False,
        buffer_size: int = 1000,
        log_training_metrics: bool = True,
        training_metrics_steps: Optional[int] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
        policy_params_fn: Callable[..., None] = lambda *args: None,
        save_checkpoint_path: Optional[str] = None,
        restore_checkpoint_path: Optional[str] = None,
        restore_params: Optional[Any] = None,
        restore_value_fn: bool = True,
        pretrained_params: Optional[Any] = None,
        init_cost_value_from: str = 'value',
):
    """PPO-PID Lagrange training."""
    per_step_safety_bound = safety_bound / episode_length if episode_length else safety_bound

    if network_factory is None:
        def network_factory(obs_size, action_size, **kwargs):
            return ppo_networks.make_ppo_networks(
                obs_size, action_size, cost_value_hidden_layer_sizes=(256,) * 5, **kwargs)

    def post_step_fn(training_state, metrics):
        lambda_lagr, integral, prev_v, deriv_ema = training_state.aux_state
        avg_cost = jnp.mean(metrics["mean_cost"][-1])
        v = (avg_cost - per_step_safety_bound).reshape((1,))
        new_integral = jnp.clip(integral + v, -pid_integral_clip, pid_integral_clip)
        new_deriv_ema = pid_deriv_ema_beta * deriv_ema + (1.0 - pid_deriv_ema_beta) * (v - prev_v)
        pid_output = pid_kp * v + pid_ki * new_integral + pid_kd * new_deriv_ema
        updated_lambda = jnp.clip(jax.nn.relu(lambda_lagr + pid_output), 0.0, pid_lambda_clip)
        new_state = training_state.replace(aux_state=(updated_lambda, new_integral, v, new_deriv_ema))
        return new_state, {"lambda_lagr": updated_lambda, "pid/violation": v}

    def lagrange_loss_fn(params, normalizer_params, data, rng, aux_state=None,
                         ppo_network=None, entropy_cost=1e-4, discounting=0.9,
                         reward_scaling=1.0, gae_lambda=0.95, clipping_epsilon=0.3,
                         normalize_advantage=True):
        lambda_lagr = aux_state[0] if aux_state else jnp.array([0.0])
        return ppo_lag_losses.compute_ppo_lagrange_loss(
            params=params, normalizer_params=normalizer_params, data=data, rng=rng,
            ppo_network=ppo_network, lambda_lagr=lambda_lagr, safety_bound=per_step_safety_bound,
            entropy_cost=entropy_cost, discounting=discounting, reward_scaling=reward_scaling,
            gae_lambda=gae_lambda, clipping_epsilon=clipping_epsilon, normalize_advantage=normalize_advantage)

    def init_aux_state_fn():
        return (jnp.array([initial_lambda_lagr]), jnp.zeros((1,)), jnp.zeros((1,)), jnp.zeros((1,)))

    effective_restore_params = pretrained_params if pretrained_params is not None else restore_params

    return ppo_train.train(
        environment=environment, num_timesteps=num_timesteps, max_devices_per_host=max_devices_per_host,
        wrap_env=wrap_env, augment_pixels=augment_pixels,
        num_envs=num_envs, episode_length=episode_length, action_repeat=action_repeat,
        wrap_env_fn=wrap_env_fn, randomization_fn=randomization_fn, learning_rate=learning_rate,
        entropy_cost=entropy_cost, discounting=discounting, unroll_length=unroll_length,
        batch_size=batch_size, num_minibatches=num_minibatches, num_updates_per_batch=num_updates_per_batch,
        num_resets_per_eval=num_resets_per_eval, normalize_observations=normalize_observations,
        reward_scaling=reward_scaling, clipping_epsilon=clipping_epsilon, gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm, normalize_advantage=normalize_advantage, network_factory=network_factory,
        seed=seed, num_evals=num_evals, eval_env=eval_env, num_eval_envs=num_eval_envs,
        deterministic_eval=deterministic_eval, buffer_size=buffer_size, log_training_metrics=log_training_metrics,
        training_metrics_steps=training_metrics_steps, progress_fn=progress_fn, policy_params_fn=policy_params_fn,
        save_checkpoint_path=save_checkpoint_path, restore_checkpoint_path=restore_checkpoint_path,
        restore_params=effective_restore_params, restore_value_fn=restore_value_fn,
        loss_fn=lagrange_loss_fn, post_step_fn=post_step_fn,
        extra_fields=("truncation", "episode_metrics", "episode_done", "cost"),
        init_aux_state_fn=init_aux_state_fn)
