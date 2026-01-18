"""FOCOPS (First Order Constrained Optimization in Policy Space) training.

Thin wrapper around the base PPO trainer with FOCOPS-specific loss and constraint handling.
Reference: Zhang et al., "First Order Constrained Optimization in Policy Space", NeurIPS 2020.
https://arxiv.org/abs/2002.06506
"""

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from brax import base
from brax import envs
from brax.training import types
from brax.training.agents.ppo import train as ppo_train
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.focops import losses as focops_losses

Metrics = types.Metrics
TrainingState = ppo_train.TrainingState


def train(
        environment: envs.Env,
        num_timesteps: int,
        episode_length: int,
        max_devices_per_host: Optional[int] = None,
        wrap_env: bool = True,
        madrona_backend: bool = False,
        augment_pixels: bool = False,
        num_envs: int = 1,
        action_repeat: int = 1,
        wrap_env_fn: Optional[Callable[[Any], Any]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        learning_rate: float = 1e-4,
        entropy_cost: float = 1e-4,
        discounting: float = 0.99,
        unroll_length: int = 10,
        batch_size: int = 32,
        num_minibatches: int = 16,
        num_updates_per_batch: int = 2,
        num_resets_per_eval: int = 0,
        normalize_observations: bool = False,
        reward_scaling: float = 1.0,
        gae_lambda: float = 0.95,
        max_grad_norm: Optional[float] = None,
        normalize_advantage: bool = True,
        network_factory: types.NetworkFactory[ppo_networks.PPONetworks] = None,
        seed: int = 0,
        # FOCOPS specific params
        safety_bound: float = 0.0,
        initial_nu: float = 1.0,
        nu_lr: float = 0.05,
        nu_max: float = 200.0,
        focops_lam: float = 1.5,
        focops_eta: float = 0.02,
        # Eval params
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
    """FOCOPS training."""
    per_step_safety_bound = safety_bound / episode_length if episode_length else safety_bound

    if network_factory is None:
        def network_factory(obs_size, action_size, **kwargs):
            return ppo_networks.make_ppo_networks(
                obs_size, action_size, cost_value_hidden_layer_sizes=(256,) * 5, **kwargs)

    def post_step_fn(training_state, metrics):
        """Updates nu (Lagrange multiplier) based on constraint violation."""
        nu = training_state.aux_state
        avg_cost = jnp.mean(metrics['mean_cost'][-1])
        cost_violation = avg_cost - per_step_safety_bound
        updated_nu = jnp.clip(jax.nn.relu(nu + nu_lr * cost_violation), 0.0, nu_max)
        new_state = training_state.replace(aux_state=updated_nu)
        return new_state, {'nu': updated_nu, 'cost_violation': cost_violation}

    def focops_loss_fn(params, normalizer_params, data, rng, aux_state=None,
                       ppo_network=None, entropy_cost=1e-4, discounting=0.99,
                       reward_scaling=1.0, gae_lambda=0.95, clipping_epsilon=0.3,
                       normalize_advantage=True):
        """FOCOPS loss function."""
        nu = aux_state if aux_state is not None else jnp.array([initial_nu])
        return focops_losses.compute_focops_loss(
            params=params, normalizer_params=normalizer_params, data=data, rng=rng,
            ppo_network=ppo_network, nu=nu, focops_lam=focops_lam, focops_eta=focops_eta,
            entropy_cost=entropy_cost, discounting=discounting, reward_scaling=reward_scaling,
            gae_lambda=gae_lambda, normalize_advantage=normalize_advantage)

    def init_aux_state_fn():
        return jnp.array([initial_nu], dtype=jnp.float32)

    effective_restore_params = pretrained_params if pretrained_params is not None else restore_params

    return ppo_train.train(
        environment=environment, num_timesteps=num_timesteps, max_devices_per_host=max_devices_per_host,
        wrap_env=wrap_env, madrona_backend=madrona_backend, augment_pixels=augment_pixels,
        num_envs=num_envs, episode_length=episode_length, action_repeat=action_repeat,
        wrap_env_fn=wrap_env_fn, randomization_fn=randomization_fn, learning_rate=learning_rate,
        entropy_cost=entropy_cost, discounting=discounting, unroll_length=unroll_length,
        batch_size=batch_size, num_minibatches=num_minibatches, num_updates_per_batch=num_updates_per_batch,
        num_resets_per_eval=num_resets_per_eval, normalize_observations=normalize_observations,
        reward_scaling=reward_scaling, clipping_epsilon=0.3, gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm, normalize_advantage=normalize_advantage, network_factory=network_factory,
        seed=seed, num_evals=num_evals, eval_env=eval_env, num_eval_envs=num_eval_envs,
        deterministic_eval=deterministic_eval, buffer_size=buffer_size, log_training_metrics=log_training_metrics,
        training_metrics_steps=training_metrics_steps, progress_fn=progress_fn, policy_params_fn=policy_params_fn,
        save_checkpoint_path=save_checkpoint_path, restore_checkpoint_path=restore_checkpoint_path,
        restore_params=effective_restore_params, restore_value_fn=restore_value_fn,
        loss_fn=focops_loss_fn, post_step_fn=post_step_fn,
        extra_fields=('truncation', 'episode_metrics', 'episode_done', 'cost'),
        init_aux_state_fn=init_aux_state_fn)
