"""P3O (Penalized Proximal Policy Optimization) training.

Thin wrapper around the base PPO trainer with P3O-specific loss and constraint handling.
Reference: Zhang et al., "Penalized Proximal Policy Optimization for Safe RL", IJCAI 2022.
https://arxiv.org/abs/2205.11814
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from crax import base
from crax import envs
from training import counterfactual
from training import types
from training.agents.ppo import train as ppo_train
from training.agents.ppo_lag.train import update_observed_episode_length
from training.agents.ppo import networks as ppo_networks
from training.agents.p3o import losses as p3o_losses

Metrics = types.Metrics
TrainingState = ppo_train.TrainingState


def train(
        environment: envs.Env,
        num_timesteps: int,
        episode_length: int,
        max_devices_per_host: Optional[int] = None,
        wrap_env: bool = True,
        augment_pixels: bool = False,
        vision_kwargs: Optional[Dict[str, Any]] = None,
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
        clipping_epsilon: float = 0.2,
        gae_lambda: float = 0.95,
        max_grad_norm: Optional[float] = None,
        normalize_advantage: bool = True,
        network_factory: types.NetworkFactory[ppo_networks.PPONetworks] = None,
        seed: int = 0,
        # P3O specific params
        safety_bound: float = 0.0,
        initial_kappa: float = 0.01,
        kappa_increase_factor: float = 1.1,
        kappa_max: float = 1000.0,
        observed_episode_length_bound: bool = False,
        episode_length_ema: float = 0.9,
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
        # counterfactual hazard relabelling (see training/counterfactual.py)
        counterfactual_mode: str = 'none',
        counterfactual_ratio: float = 0.5,
        counterfactual_candidates: int = 4,
        counterfactual_rho_clip: float = 1.0,
        counterfactual_gate_sharpness: float = 10.0,
):
    """P3O training.

    `observed_episode_length_bound` and the counterfactual arguments are as in ppo_lag.train.
    """
    per_step_safety_bound = safety_bound / episode_length if episode_length else safety_bound

    if network_factory is None:
        def network_factory(obs_size, action_size, **kwargs):
            return ppo_networks.make_ppo_networks(
                obs_size, action_size, cost_value_hidden_layer_sizes=(256,) * 5, **kwargs)

    # P3O stores (kappa, cost_violation) in aux_state, plus the observed episode length if the
    # per-step bound tracks it
    def post_step_fn(training_state, metrics):
        """Updates kappa based on constraint violation."""
        kappa = training_state.aux_state[0]
        avg_cost = jnp.mean(metrics['mean_cost'][-1])
        extra_metrics = {}
        step_bound = per_step_safety_bound
        if observed_episode_length_bound:
            observed_length = update_observed_episode_length(
                metrics, training_state.aux_state[2], episode_length_ema)
            step_bound = safety_bound / observed_length
            extra_metrics = {'observed_episode_length': observed_length, 'per_step_safety_bound': step_bound}
        cost_violation = jnp.array([avg_cost - step_bound])
        constraint_violated = cost_violation[0] > 0.0

        # Increase kappa when violated, decrease slowly when satisfied
        updated_kappa = jnp.where(
            constraint_violated,
            jnp.minimum(kappa * kappa_increase_factor, kappa_max),
            jnp.maximum(kappa * 0.9, initial_kappa),
        )
        aux_state = (updated_kappa, cost_violation)
        if observed_episode_length_bound:
            aux_state = aux_state + (observed_length,)
        new_state = training_state.replace(aux_state=aux_state)
        return new_state, {'kappa': updated_kappa, 'cost_violation': cost_violation[0], **extra_metrics}

    def p3o_loss_fn(params, normalizer_params, data, rng, aux_state=None,
                    ppo_network=None, entropy_cost=1e-4, discounting=0.99,
                    reward_scaling=1.0, gae_lambda=0.95, clipping_epsilon=0.2,
                    normalize_advantage=True):
        """P3O loss function."""
        kappa, cost_violation = aux_state[:2] if aux_state else (jnp.array([initial_kappa]), jnp.array([0.0]))
        return p3o_losses.compute_p3o_loss(
            params=params, normalizer_params=normalizer_params, data=data, rng=rng,
            ppo_network=ppo_network, kappa=kappa, cost_violation=cost_violation,
            safety_bound=safety_bound, episode_length=episode_length,
            entropy_cost=entropy_cost, discounting=discounting, reward_scaling=reward_scaling,
            gae_lambda=gae_lambda, clipping_epsilon=clipping_epsilon, normalize_advantage=normalize_advantage)

    def step_bound_fn(aux_state):
        return safety_bound / aux_state[2] if observed_episode_length_bound else per_step_safety_bound

    def init_aux_state_fn():
        aux_state = (jnp.array([initial_kappa], dtype=jnp.float32), jnp.array([0.0], dtype=jnp.float32))
        if observed_episode_length_bound:
            # Starts at the fixed conversion; tracks completed episodes from the first update
            aux_state = aux_state + (jnp.asarray(episode_length, dtype=jnp.float32),)
        return aux_state

    effective_restore_params = pretrained_params if pretrained_params is not None else restore_params

    return ppo_train.train(
        environment=environment, num_timesteps=num_timesteps, max_devices_per_host=max_devices_per_host,
        wrap_env=wrap_env, augment_pixels=augment_pixels, vision_kwargs=vision_kwargs,
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
        loss_fn=p3o_loss_fn, post_step_fn=post_step_fn,
        init_aux_state_fn=init_aux_state_fn,
        **counterfactual.counterfactual_train_kwargs(
            environment, ('truncation', 'episode_metrics', 'episode_done', 'cost'), action_repeat,
            counterfactual_mode, counterfactual_ratio, counterfactual_candidates, counterfactual_rho_clip,
            counterfactual_gate_sharpness, step_bound_fn))
