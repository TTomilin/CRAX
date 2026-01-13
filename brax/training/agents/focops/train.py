# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FOCOPS (First Order Constrained Optimization in Policy Space) training.

Reference: Zhang et al., "First Order Constrained Optimization in Policy Space",
NeurIPS 2020.
https://arxiv.org/abs/2002.06506
"""

import functools
import time
from typing import Any, Callable, Optional, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging

from brax import base
from brax import envs
from brax.training import acting
from brax.training import logger as metric_logger
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.focops import checkpoint
from brax.training.agents.focops import losses as focops_losses
from brax.training.agents.focops import networks as focops_networks
from brax.training.types import PRNGKey
from brax.training.types import Params

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the FOCOPS learner."""

    optimizer_state: optax.OptState
    params: focops_losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: types.UInt64
    nu: jnp.ndarray  # Lagrange multiplier for cost constraint


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def _maybe_wrap_env(
        env: envs.Env,
        wrap_env: bool,
        num_envs: int,
        episode_length: Optional[int],
        action_repeat: int,
        device_count: int,
        key_env: PRNGKey,
        wrap_env_fn: Optional[Callable[[Any], Any]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
):
    """Wraps the environment for training/eval if wrap_env is True."""
    if not wrap_env:
        return env
    if episode_length is None:
        raise ValueError('episode_length must be specified in focops.train')
    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // device_count
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(
            randomization_fn, rng=randomization_rng
        )
    if wrap_env_fn is not None:
        wrap_for_training = wrap_env_fn
    else:
        wrap_for_training = envs.training.wrap
    env = wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )
    return env


def train(
        environment: envs.Env,
        num_timesteps: int,
        max_devices_per_host: Optional[int] = None,
        wrap_env: bool = True,
        num_envs: int = 1,
        episode_length: Optional[int] = None,
        action_repeat: int = 1,
        wrap_env_fn: Optional[Callable[[Any], Any]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        # Optimizer params
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
        network_factory: types.NetworkFactory[
            focops_networks.PPONetworks
        ] = focops_networks.make_ppo_networks,
        seed: int = 0,
        # FOCOPS specific params
        safety_bound: float = 0.0,
        initial_nu: float = 1.0,   # Start with some cost awareness
        nu_lr: float = 0.05,       # Faster response to violations
        nu_max: float = 200.0,     # Allow stronger penalties
        focops_lam: float = 1.5,   # Temperature (lower = more aggressive)
        focops_eta: float = 0.02,  # KL threshold
        # Eval params
        num_evals: int = 0,
        eval_env: Optional[envs.Env] = None,
        num_eval_envs: int = 128,
        deterministic_eval: bool = False,
        # Training metrics
        buffer_size: int = 1000,
        log_training_metrics: bool = True,
        training_metrics_steps: Optional[int] = None,
        # Callbacks
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
        policy_params_fn: Callable[..., None] = lambda *args: None,
        # Checkpointing
        save_checkpoint_path: Optional[str] = None,
        restore_checkpoint_path: Optional[str] = None,
        restore_params: Optional[Any] = None,
        restore_value_fn: bool = True,
):
    """FOCOPS training.

    Args:
      environment: the environment to train
      num_timesteps: total number of environment steps
      max_devices_per_host: maximum number of chips to use per host
      wrap_env: If True, wrap the environment for training
      num_envs: number of parallel environments
      episode_length: length of an environment episode
      action_repeat: number of timesteps to repeat an action
      wrap_env_fn: custom function that wraps the environment
      randomization_fn: callback for domain randomization
      learning_rate: learning rate
      entropy_cost: entropy bonus coefficient
      discounting: discount factor
      unroll_length: number of timesteps to unroll
      batch_size: batch size for minibatch SGD
      num_minibatches: number of minibatches per update
      num_updates_per_batch: number of gradient updates per batch
      num_resets_per_eval: environment resets between evals
      normalize_observations: whether to normalize observations
      reward_scaling: reward multiplier
      gae_lambda: GAE lambda
      max_grad_norm: gradient clipping norm
      normalize_advantage: whether to normalize advantages
      network_factory: function that generates networks
      seed: random seed
      safety_bound: episodic cost constraint bound
      initial_nu: initial Lagrange multiplier
      nu_lr: learning rate for nu updates
      nu_max: maximum value for nu
      focops_lam: temperature parameter λ (fixed)
      focops_eta: KL threshold for gradient masking
      num_evals: number of evaluations during training
      eval_env: optional environment for evaluation
      num_eval_envs: number of environments for evaluation
      deterministic_eval: whether to use deterministic policy for eval
      buffer_size: metrics buffer size
      log_training_metrics: whether to log training metrics
      training_metrics_steps: steps between logging
      progress_fn: callback for reporting metrics
      policy_params_fn: callback for saving checkpoints
      save_checkpoint_path: path for saving checkpoints
      restore_checkpoint_path: path for restoring checkpoints
      restore_params: raw network parameters to restore
      restore_value_fn: whether to restore value function

    Returns:
      Tuple of (make_policy function, network params, metrics)
    """
    assert batch_size * num_minibatches % num_envs == 0

    xt = time.time()

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        'Device count: %d, process count: %d (id %d), local device count: %d, '
        'devices to be used count: %d',
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count

    env_step_per_training_step = (
            batch_size * unroll_length * num_minibatches * action_repeat
    )

    if num_evals == 0:
        num_evals_after_init = 1
        num_training_steps_per_epoch = np.ceil(
            num_timesteps / (env_step_per_training_step * max(num_resets_per_eval, 1))
        ).astype(int)
    else:
        num_evals_after_init = max(num_evals - 1, 1)
        num_training_steps_per_epoch = np.ceil(
            num_timesteps
            / (
                    num_evals_after_init
                    * env_step_per_training_step
                    * max(num_resets_per_eval, 1)
            )
        ).astype(int)

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    key_policy, key_value, key_cost_value = jax.random.split(global_key, 3)
    del global_key

    assert num_envs % device_count == 0

    env = _maybe_wrap_env(
        environment,
        wrap_env,
        num_envs,
        episode_length,
        action_repeat,
        device_count,
        key_env,
        wrap_env_fn,
        randomization_fn,
    )
    if local_devices_to_use > 1:
        reset_fn = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
    else:
        reset_fn = jax.jit(jax.vmap(env.reset))
    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(
        key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
    )
    env_state = reset_fn(key_envs)
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)

    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize
    ppo_network = network_factory(
        obs_shape, env.action_size, preprocess_observations_fn=normalize
    )
    make_policy = focops_networks.make_inference_fn(ppo_network)

    optimizer = optax.adam(learning_rate=learning_rate)
    if max_grad_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )

    def loss_fn(params, normalizer_params, data, rng, nu):
        return focops_losses.compute_focops_loss(
            params=params,
            normalizer_params=normalizer_params,
            data=data,
            rng=rng,
            ppo_network=ppo_network,
            nu=nu,
            focops_lam=focops_lam,
            focops_eta=focops_eta,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            normalize_advantage=normalize_advantage,
        )

    def gradient_update_fn(params, normalizer_params, data, rng, nu, optimizer_state):
        """Gradient update function for FOCOPS."""

        def loss_and_pgrad(params, normalizer_params, data, rng, nu):
            total_loss, metrics = loss_fn(params, normalizer_params, data, rng, nu)
            return total_loss, metrics

        grad_fn = jax.value_and_grad(loss_and_pgrad, has_aux=True)
        (loss, metrics), grads = grad_fn(params, normalizer_params, data, rng, nu)

        updates, new_optimizer_state = optimizer.update(grads, optimizer_state, params)
        new_params = optax.apply_updates(params, updates)

        return (loss, metrics), new_params, new_optimizer_state

    metrics_aggregator = metric_logger.MetricsLogger(
        buffer_size=buffer_size,
        steps_between_logging=training_metrics_steps,
        progress_fn=progress_fn,
    )

    def minibatch_step(
            carry,
            data: types.Transition,
            normalizer_params: running_statistics.RunningStatisticsState,
            nu: jnp.ndarray,
    ):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, optimizer_state = gradient_update_fn(
            params,
            normalizer_params,
            data,
            key_loss,
            nu,
            optimizer_state,
        )

        return (optimizer_state, params, key), metrics

    def sgd_step(
            carry,
            unused_t,
            data: types.Transition,
            normalizer_params: running_statistics.RunningStatisticsState,
            nu: jnp.ndarray,
    ):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params, nu=nu),
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, key), metrics

    def training_step(
            carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t
    ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
        training_state, state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        policy = make_policy((
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        ))

        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env,
                current_state,
                policy,
                current_key,
                unroll_length,
                extra_fields=('truncation', 'episode_metrics', 'episode_done', 'cost'),
            )
            return (next_state, next_key), data

        (state, _), data = jax.lax.scan(
            f,
            (state, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_envs,
        )
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )
        assert data.discount.shape[1:] == (unroll_length,)

        state_extras = data.extras['state_extras']
        episode_metrics = state_extras['episode_metrics']
        jax.debug.callback(
            metrics_aggregator.update_env_metrics,
            episode_metrics,
            state_extras['episode_done'],
            training_state.env_steps,
        )

        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                sgd_step, data=data, normalizer_params=normalizer_params, nu=training_state.nu
            ),
            (training_state.optimizer_state, training_state.params, key_sgd),
            (),
            length=num_updates_per_batch,
        )

        # FOCOPS nu (Lagrange multiplier) update using episodic costs
        # ν ← max(0, ν + α(J^C - d))
        ep_cost = episode_metrics['cost']
        ep_length = episode_metrics['length']

        ep_cost_last = ep_cost[:, -1]
        ep_steps_last = jnp.where(
            ep_length is not None,
            ep_length[:, -1],
            jnp.ones_like(ep_cost_last) * float(unroll_length),
        )

        mean_cost_per_step = jnp.mean(ep_cost_last / jnp.maximum(ep_steps_last, 1.0))
        safety_bound_step = safety_bound / float(episode_length)

        # Cost violation: positive when constraint is violated
        cost_violation = mean_cost_per_step - safety_bound_step

        # Update nu: increase when violated, decrease when satisfied
        nu_update_rate = jnp.asarray(nu_lr, dtype=jnp.float32)
        nu_cap = jnp.asarray(nu_max, dtype=jnp.float32)
        updated_nu = jax.nn.relu(training_state.nu + nu_update_rate * cost_violation)
        updated_nu = jnp.minimum(updated_nu, nu_cap)

        metrics = {
            **metrics,
            'nu': updated_nu,
            'cost_violation': cost_violation,
            'mean_cost_per_step': mean_cost_per_step,
            'safety_bound_step': safety_bound_step,
        }

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
            nu=updated_nu,
        )

        if log_training_metrics:
            jax.debug.callback(
                metrics_aggregator.update_train_metrics,
                metrics,
                new_training_state.env_steps,
            )

        return (new_training_state, state, new_key), metrics

    def training_epoch(
            training_state: TrainingState, state: envs.State, key: PRNGKey
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        (training_state, state, _), loss_metrics = jax.lax.scan(
            training_step,
            (training_state, state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        return training_state, state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    def training_epoch_with_timing(
            training_state: TrainingState, env_state: envs.State, key: PRNGKey
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        nonlocal training_walltime
        t = time.time()
        training_state, env_state = _strip_weak_type((training_state, env_state))
        result = training_epoch(training_state, env_state, key)
        training_state, env_state, metrics = _strip_weak_type(result)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        fps = (
                      num_training_steps_per_epoch
                      * env_step_per_training_step
                      * max(num_resets_per_eval, 1)
              ) / epoch_training_time

        metrics = {
            'performance/fps': fps,
            'performance/walltime': training_walltime,
            **{f'training/{name}': value for name, value in metrics.items()},
        }
        return training_state, env_state, metrics

    # Initialize model params and training state
    init_params = focops_losses.PPONetworkParams(
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
        cost_value=ppo_network.cost_value_network.init(key_cost_value),
    )

    obs_spec = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
    )
    training_state = TrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        normalizer_params=running_statistics.init_state(obs_spec),
        env_steps=types.UInt64(hi=0, lo=0),
        nu=jnp.array([initial_nu], dtype=jnp.float32),
    )

    if restore_checkpoint_path is not None:
        params = checkpoint.load(restore_checkpoint_path)
        cost_value_params = params[2] if len(params) > 2 else init_params.cost_value
        value_params = params[1] if restore_value_fn else init_params.value
        nu_value = params[3] if len(params) > 3 else jnp.array([initial_nu], dtype=jnp.float32)
        training_state = training_state.replace(
            normalizer_params=params[0],
            params=training_state.params.replace(
                policy=params[1],
                value=value_params,
                cost_value=cost_value_params
            ),
            nu=nu_value,
        )

    if restore_params is not None:
        logging.info('Restoring TrainingState from `restore_params`.')
        cost_value_params = restore_params[2] if len(restore_params) > 2 else init_params.cost_value
        value_params = restore_params[1] if restore_value_fn else init_params.value
        nu_value = restore_params[3] if len(restore_params) > 3 else jnp.array([initial_nu], dtype=jnp.float32)
        training_state = training_state.replace(
            normalizer_params=restore_params[0],
            params=training_state.params.replace(
                policy=restore_params[1],
                value=value_params,
                cost_value=cost_value_params
            ),
            nu=nu_value,
        )

    if num_timesteps == 0:
        return (
            make_policy,
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
                training_state.params.cost_value,
            ),
            {},
        )

    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

    evaluator = None
    if num_evals > 0:
        eval_env = _maybe_wrap_env(
            eval_env or environment,
            wrap_env,
            num_eval_envs,
            episode_length,
            action_repeat,
            device_count=1,
            key_env=eval_key,
            wrap_env_fn=wrap_env_fn,
            randomization_fn=randomization_fn,
        )
        evaluator = acting.Evaluator(
            eval_env,
            functools.partial(make_policy, deterministic=deterministic_eval),
            num_eval_envs=num_eval_envs,
            episode_length=episode_length,
            action_repeat=action_repeat,
            key=eval_key,
        )

    metrics = {}
    if process_id == 0 and num_evals > 1 and evaluator is not None:
        metrics = evaluator.run_evaluation(
            _unpmap((
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            )),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    training_metrics = {}
    training_walltime = 0
    current_step = 0

    for it in range(num_evals_after_init):
        logging.info('starting iteration %s %s', it, time.time() - xt)

        for _ in range(max(num_resets_per_eval, 1)):
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, training_metrics) = (
                training_epoch_with_timing(training_state, env_state, epoch_keys)
            )
            current_step = int(_unpmap(training_state.env_steps))
            progress_fn(current_step, training_metrics)

            key_envs = jax.vmap(
                lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
            )(key_envs, key_envs.shape[1])
            env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

        if process_id != 0:
            continue

        params = _unpmap((
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
            training_state.params.cost_value,
        ))

        policy_params_fn(current_step, make_policy, params)

        if save_checkpoint_path is not None:
            ckpt_config = checkpoint.network_config(
                observation_size=obs_shape,
                action_size=env.action_size,
                normalize_observations=normalize_observations,
                network_factory=network_factory,
            )
            full_params = params + (_unpmap(training_state.nu),)
            checkpoint.save(
                save_checkpoint_path, current_step, full_params, ckpt_config
            )

        if num_evals > 0 and evaluator is not None:
            metrics = evaluator.run_evaluation(
                params,
                training_metrics,
            )
            logging.info(metrics)
            progress_fn(current_step, metrics)

    total_steps = current_step
    assert total_steps >= num_timesteps

    pmap.assert_is_replicated(training_state)
    params = _unpmap((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
        training_state.params.cost_value,
    ))

    if not metrics:
        metrics = {'training/final_step': total_steps}
        if training_metrics:
            metrics.update(training_metrics)

    logging.info('total steps: %s', total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics, eval_env)
