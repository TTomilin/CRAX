"""CRPO (Constrained Rectified Policy Optimization) training.

Thin wrapper around the base PPO trainer with CRPO's rectified-objective
constraint handling.
Reference: Xu, Liang, Lan, "CRPO: A New Approach for Safe Reinforcement
Learning with Convergence Guarantee", ICML 2021.
https://arxiv.org/abs/2011.05869
"""

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from crax import base
from crax import envs
from crax.training import types
from crax.training.agents.crpo import losses as crpo_losses
from crax.training.agents.ppo import networks as ppo_networks
from crax.training.agents.ppo import train as ppo_train

Metrics = types.Metrics

# Re-export TrainingState for backward compatibility
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
        # CRPO specific params
        safety_bound: float = 0.0,
        crpo_eta: float = 0.0,
        # eval
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
        # transfer learning / curriculum support
        pretrained_params: Optional[Any] = None,
        init_cost_value_from: str = 'value',
):
    """CRPO training.

    Args:
      environment: the environment to train
      num_timesteps: the total number of environment steps to use during training
      max_devices_per_host: maximum number of chips to use per host process
      wrap_env: If True, wrap the environment for training.
      augment_pixels: whether to add image augmentation to pixel inputs
      num_envs: the number of parallel environments to use for rollouts
      episode_length: the length of an environment episode
      action_repeat: the number of timesteps to repeat an action
      wrap_env_fn: a custom function that wraps the environment for training.
      randomization_fn: a user-defined callback function that generates randomized
        environments
      learning_rate: learning rate for ppo loss
      entropy_cost: entropy reward for ppo loss
      discounting: discounting rate
      unroll_length: the number of timesteps to unroll in each environment.
      batch_size: the batch size for each minibatch SGD step
      num_minibatches: the number of times to run the SGD step
      num_updates_per_batch: the number of times to run the gradient update
      num_resets_per_eval: the number of environment resets to run between evals
      normalize_observations: whether to normalize observations
      reward_scaling: float scaling for reward
      clipping_epsilon: clipping epsilon for PPO loss
      gae_lambda: General advantage estimation lambda
      max_grad_norm: gradient clipping norm value.
      normalize_advantage: whether to normalize advantage estimate
      network_factory: function that generates networks
      seed: random seed
      safety_bound: the safety constraint bound for CRPO (episodic)
      crpo_eta: tolerance added to the safety bound when deciding whether the
        constraint is satisfied. The reward objective is optimized whenever
        avg_cost <= per_step_safety_bound + crpo_eta; otherwise the cost
        objective is optimized instead for that training step.
      num_evals: the number of evals to run during the entire training run.
      eval_env: an optional environment for eval only
      num_eval_envs: the number of envs to use for evaluation.
      deterministic_eval: whether to run the eval with a deterministic policy
      log_training_metrics: whether to log training metrics
      training_metrics_steps: the number of environment steps between logging
      progress_fn: a user-defined callback function for reporting metrics
      policy_params_fn: a user-defined callback function for saving checkpoints
      save_checkpoint_path: the path used to save checkpoints.
      restore_checkpoint_path: the path used to restore previous model params
      restore_params: raw network parameters to restore the TrainingState from.
      restore_value_fn: whether to restore the value function from the checkpoint
        or use a random initialization
      pretrained_params: alias for restore_params, used for curriculum/transfer learning.
        Takes precedence over restore_params if both are specified.
      init_cost_value_from: how to initialize cost_value network when transferring
        from an algorithm without cost_value (e.g., PPO). Options:
        - 'value': copy from value network (often works well for transfer)
        - 'random': use random initialization

    Returns:
      Tuple of (make_policy function, network params, metrics, eval_env)
    """
    # Convert episodic safety bound to per-step bound
    per_step_safety_bound = safety_bound / episode_length if episode_length else safety_bound

    # Create network factory with cost value network if not provided
    if network_factory is None:
        def network_factory(obs_size, action_size, **kwargs):
            return ppo_networks.make_ppo_networks(
                obs_size, action_size, cost_value_hidden_layer_sizes=(256,) * 5, **kwargs)

    # CRPO stores the current regime (1.0 = optimize reward, 0.0 = optimize cost) in aux_state
    def post_step_fn(training_state: TrainingState, metrics: Metrics) -> Tuple[TrainingState, Metrics]:
        """Rectifies the objective based on the constraint's current violation."""
        avg_cost = jnp.mean(metrics['mean_cost'][-1])
        cost_violation = avg_cost - per_step_safety_bound
        updated_regime = jnp.where(cost_violation > crpo_eta, 0.0, 1.0)
        new_training_state = training_state.replace(aux_state=jnp.array([updated_regime]))
        return new_training_state, {'regime': updated_regime, 'cost_violation': cost_violation}

    # Define the custom loss function for CRPO
    def crpo_loss_fn(params, normalizer_params, data, rng, aux_state=None,
                     ppo_network=None, entropy_cost=1e-4, discounting=0.99,
                     reward_scaling=1.0, gae_lambda=0.95, clipping_epsilon=0.2,
                     normalize_advantage=True):
        """CRPO loss function."""
        regime = aux_state if aux_state is not None else jnp.array([1.0])
        return crpo_losses.compute_crpo_loss(
            params=params, normalizer_params=normalizer_params, data=data, rng=rng,
            ppo_network=ppo_network, regime=regime,
            entropy_cost=entropy_cost, discounting=discounting, reward_scaling=reward_scaling,
            gae_lambda=gae_lambda, clipping_epsilon=clipping_epsilon, normalize_advantage=normalize_advantage)

    # Initialize aux_state assuming the constraint starts out satisfied
    def init_aux_state_fn():
        return jnp.array([1.0], dtype=jnp.float32)

    # Extra fields to collect during rollout (including cost)
    extra_fields = ('truncation', 'episode_metrics', 'episode_done', 'cost')

    # Handle pretrained_params taking precedence over restore_params
    effective_restore_params = pretrained_params if pretrained_params is not None else restore_params

    # Call base PPO trainer with hooks
    return ppo_train.train(
        environment=environment,
        num_timesteps=num_timesteps,
        max_devices_per_host=max_devices_per_host,
        wrap_env=wrap_env,
        augment_pixels=augment_pixels,
        num_envs=num_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        wrap_env_fn=wrap_env_fn,
        randomization_fn=randomization_fn,
        learning_rate=learning_rate,
        entropy_cost=entropy_cost,
        discounting=discounting,
        unroll_length=unroll_length,
        batch_size=batch_size,
        num_minibatches=num_minibatches,
        num_updates_per_batch=num_updates_per_batch,
        num_resets_per_eval=num_resets_per_eval,
        normalize_observations=normalize_observations,
        reward_scaling=reward_scaling,
        clipping_epsilon=clipping_epsilon,
        gae_lambda=gae_lambda,
        max_grad_norm=max_grad_norm,
        normalize_advantage=normalize_advantage,
        network_factory=network_factory,
        seed=seed,
        num_evals=num_evals,
        eval_env=eval_env,
        num_eval_envs=num_eval_envs,
        deterministic_eval=deterministic_eval,
        buffer_size=buffer_size,
        log_training_metrics=log_training_metrics,
        training_metrics_steps=training_metrics_steps,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        save_checkpoint_path=save_checkpoint_path,
        restore_checkpoint_path=restore_checkpoint_path,
        restore_params=effective_restore_params,
        restore_value_fn=restore_value_fn,
        # Constrained RL hooks
        loss_fn=crpo_loss_fn,
        post_step_fn=post_step_fn,
        extra_fields=extra_fields,
        init_aux_state_fn=init_aux_state_fn,
    )
