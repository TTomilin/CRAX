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

"""P3O (Penalized Proximal Policy Optimization) training.

Reference: Zhang et al., "Penalized Proximal Policy Optimization for Safe
Reinforcement Learning", IJCAI 2022.
https://arxiv.org/abs/2205.11814
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

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
from brax.training.agents.p3o import checkpoint
from brax.training.agents.p3o import losses as p3o_losses
from brax.training.agents.p3o import networks as p3o_networks
from brax.training.types import PRNGKey
from brax.training.types import Params

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the P3O learner."""

    optimizer_state: optax.OptState
    params: p3o_losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: types.UInt64
    kappa: jnp.ndarray  # Penalty coefficient


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def _validate_madrona_args(
        madrona_backend: bool,
        num_envs: int,
        num_eval_envs: int,
        action_repeat: int,
        eval_env: Optional[envs.Env] = None,
):
    """Validates arguments for Madrona-MJX."""
    if madrona_backend:
        if eval_env:
            raise ValueError("Madrona-MJX doesn't support multiple env instances")
        if num_eval_envs != num_envs:
            raise ValueError('Madrona-MJX requires a fixed batch size')
        if action_repeat != 1:
            raise ValueError(
                "Implement action_repeat using PipelineEnv's _n_frames to avoid"
                ' unnecessary rendering!'
            )


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
        episode_length = getattr(env, 'episode_length', None)
    if episode_length is None:
        raise ValueError('episode_length must be specified (no env default found) in p3o.train')
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


def _random_translate_pixels(
        obs: Mapping[str, jax.Array], key: PRNGKey
) -> Mapping[str, jax.Array]:
    """Apply random translations to B x T x ... pixel observations."""

    @jax.vmap
    def rt_all_views(
            ub_obs: Mapping[str, jax.Array], key: PRNGKey
    ) -> Mapping[str, jax.Array]:
        def rt_view(
                img: jax.Array, padding: int, key: PRNGKey
        ) -> jax.Array:
            crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
            zero = jnp.zeros((1,), dtype=jnp.int32)
            crop_from = jnp.concatenate([zero, crop_from, zero])
            padded_img = jnp.pad(
                img,
                ((0, 0), (padding, padding), (padding, padding), (0, 0)),
                mode='edge',
            )
            return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

        out = {}
        for k_view, v_view in ub_obs.items():
            if k_view.startswith('pixels/'):
                key, key_shift = jax.random.split(key)
                out[k_view] = rt_view(v_view, 4, key_shift)
        return {**ub_obs, **out}

    bdim = next(iter(obs.items()), None)[1].shape[0]
    keys = jax.random.split(key, bdim)
    obs = rt_all_views(obs, keys)
    return obs


def _remove_pixels(
        obs: Union[jnp.ndarray, Mapping[str, jax.Array]],
) -> Union[jnp.ndarray, Mapping[str, jax.Array]]:
    """Removes pixel observations from the observation dict."""
    if not isinstance(obs, Mapping):
        return obs
    return {k: v for k, v in obs.items() if not k.startswith('pixels/')}


def train(
        environment: envs.Env,
        num_timesteps: int,
        max_devices_per_host: Optional[int] = None,
        # high-level control flow
        wrap_env: bool = True,
        madrona_backend: bool = False,
        augment_pixels: bool = False,
        # environment wrapper
        num_envs: int = 1,
        episode_length: Optional[int] = None,
        action_repeat: int = 1,
        wrap_env_fn: Optional[Callable[[Any], Any]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        # ppo params
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
        network_factory: types.NetworkFactory[
            p3o_networks.PPONetworks
        ] = p3o_networks.make_ppo_networks,
        seed: int = 0,
        # P3O specific params
        safety_bound: float = 0.0,
        initial_kappa: float = 0.01,
        kappa_increase_factor: float = 1.5,
        kappa_max: float = 1000.0,
        # eval
        num_evals: int = 0,
        eval_env: Optional[envs.Env] = None,
        num_eval_envs: int = 128,
        deterministic_eval: bool = False,
        # training metrics
        buffer_size: int = 1000,
        log_training_metrics: bool = True,
        training_metrics_steps: Optional[int] = None,
        # callbacks
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
        policy_params_fn: Callable[..., None] = lambda *args: None,
        # checkpointing
        save_checkpoint_path: Optional[str] = None,
        restore_checkpoint_path: Optional[str] = None,
        restore_params: Optional[Any] = None,
        restore_value_fn: bool = True,
        # transfer learning / curriculum support
        pretrained_params: Optional[Any] = None,
        init_cost_value_from: str = 'value',
):
    """P3O training.

    Args:
      environment: the environment to train
      num_timesteps: the total number of environment steps to use during training
      max_devices_per_host: maximum number of chips to use per host process
      wrap_env: If True, wrap the environment for training.
      madrona_backend: whether to use Madrona backend for training
      augment_pixels: whether to add image augmentation to pixel inputs
      num_envs: the number of parallel environments to use for rollouts
      episode_length: the length of an environment episode
      action_repeat: the number of timesteps to repeat an action
      wrap_env_fn: a custom function that wraps the environment for training
      randomization_fn: a user-defined callback function that generates randomized
        environments
      learning_rate: learning rate for ppo loss
      entropy_cost: entropy reward for ppo loss
      discounting: discounting rate
      unroll_length: the number of timesteps to unroll in each environment
      batch_size: the batch size for each minibatch SGD step
      num_minibatches: the number of times to run the SGD step
      num_updates_per_batch: the number of times to run the gradient update
      num_resets_per_eval: the number of environment resets between each eval
      normalize_observations: whether to normalize observations
      reward_scaling: float scaling for reward
      clipping_epsilon: clipping epsilon for PPO loss
      gae_lambda: General advantage estimation lambda
      max_grad_norm: gradient clipping norm value
      normalize_advantage: whether to normalize advantage estimate
      network_factory: function that generates networks
      seed: random seed
      safety_bound: the safety constraint bound (episodic)
      initial_kappa: initial penalty coefficient
      kappa_increase_factor: factor to increase kappa when constraint violated (rho)
      kappa_max: maximum value for kappa
      num_evals: the number of evals to run during training
      eval_env: an optional environment for eval only
      num_eval_envs: the number of envs to use for evaluation
      deterministic_eval: whether to run the eval with a deterministic policy
      log_training_metrics: whether to log training metrics
      training_metrics_steps: the number of steps between logging
      progress_fn: a user-defined callback function for reporting metrics
      policy_params_fn: a user-defined callback function for saving checkpoints
      save_checkpoint_path: the path used to save checkpoints
      restore_checkpoint_path: the path used to restore previous model params
      restore_params: raw network parameters to restore
      restore_value_fn: whether to restore the value function from checkpoint

    Returns:
      Tuple of (make_policy function, network params, metrics)
    """
    assert batch_size * num_minibatches % num_envs == 0
    _validate_madrona_args(
        madrona_backend, num_envs, num_eval_envs, action_repeat, eval_env
    )

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
    make_policy = p3o_networks.make_inference_fn(ppo_network)

    optimizer = optax.adam(learning_rate=learning_rate)
    if max_grad_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )

    def loss_fn(params, normalizer_params, data, rng, kappa, cost_violation):
        return p3o_losses.compute_p3o_loss(
            params=params,
            normalizer_params=normalizer_params,
            data=data,
            rng=rng,
            ppo_network=ppo_network,
            kappa=kappa,
            cost_violation=cost_violation,
            safety_bound=safety_bound,
            episode_length=episode_length,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            clipping_epsilon=clipping_epsilon,
            normalize_advantage=normalize_advantage,
        )

    def gradient_update_fn(params, normalizer_params, data, rng, kappa, cost_violation, optimizer_state):
        """Custom gradient update function for P3O."""

        def loss_and_pgrad(params, normalizer_params, data, rng, kappa, cost_violation):
            total_loss, metrics = loss_fn(params, normalizer_params, data, rng, kappa, cost_violation)
            return total_loss, metrics

        grad_fn = jax.value_and_grad(loss_and_pgrad, has_aux=True)
        (loss, metrics), grads = grad_fn(params, normalizer_params, data, rng, kappa, cost_violation)

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
            kappa: jnp.ndarray,
            cost_violation: jnp.ndarray,
    ):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, optimizer_state = gradient_update_fn(
            params,
            normalizer_params,
            data,
            key_loss,
            kappa,
            cost_violation,
            optimizer_state,
        )

        return (optimizer_state, params, key), metrics

    def sgd_step(
            carry,
            unused_t,
            data: types.Transition,
            normalizer_params: running_statistics.RunningStatisticsState,
            kappa: jnp.ndarray,
            cost_violation: jnp.ndarray,
    ):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        if augment_pixels:
            key, key_rt = jax.random.split(key)
            r_translate = functools.partial(_random_translate_pixels, key=key_rt)
            data = types.Transition(
                observation=r_translate(data.observation),
                action=data.action,
                reward=data.reward,
                discount=data.discount,
                next_observation=r_translate(data.next_observation),
                extras=data.extras,
            )

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                minibatch_step,
                normalizer_params=normalizer_params,
                kappa=kappa,
                cost_violation=cost_violation,
            ),
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
            _remove_pixels(data.observation),
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        # Compute cost violation BEFORE SGD steps using episodic costs (like ppo_lag)
        ep_cost = episode_metrics['cost']  # [B, T]
        ep_length = episode_metrics['length']  # [B, T]

        # Use the last value in the rollout for each env
        ep_cost_last = ep_cost[:, -1]  # [B]
        ep_steps_last = jnp.where(
            ep_length is not None,
            ep_length[:, -1],
            jnp.ones_like(ep_cost_last) * float(unroll_length),
        )

        # Estimate per-step cost in the current episode
        mean_cost_per_step = jnp.mean(ep_cost_last / jnp.maximum(ep_steps_last, 1.0))

        # Convert the episodic safety_bound to per-step
        safety_bound_step = safety_bound / float(episode_length)

        # Cost violation: positive when constraint is violated
        cost_violation = jnp.array([mean_cost_per_step - safety_bound_step], dtype=jnp.float32)

        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                sgd_step,
                data=data,
                normalizer_params=normalizer_params,
                kappa=training_state.kappa,
                cost_violation=cost_violation,
            ),
            (training_state.optimizer_state, training_state.params, key_sgd),
            (),
            length=num_updates_per_batch,
        )

        # P3O kappa adaptation: bidirectional update
        constraint_violated = cost_violation[0] > 0.0

        # Increase kappa when violated, decrease slowly when satisfied
        rho_increase = jnp.asarray(kappa_increase_factor, dtype=jnp.float32)
        rho_decrease = jnp.asarray(0.9, dtype=jnp.float32)
        kappa_cap = jnp.asarray(kappa_max, dtype=jnp.float32)
        kappa_min = jnp.asarray(initial_kappa, dtype=jnp.float32)

        updated_kappa = jnp.where(
            constraint_violated,
            jnp.minimum(training_state.kappa * rho_increase, kappa_cap),
            jnp.maximum(training_state.kappa * rho_decrease, kappa_min),
        )

        # Add kappa info to metrics
        metrics = {
            **metrics,
            'kappa': updated_kappa,
            'cost_violation': cost_violation[0],
            'constraint_violated': constraint_violated.astype(jnp.float32),
            'mean_cost_per_step': mean_cost_per_step,
            'safety_bound_step': safety_bound_step,
        }

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
            kappa=updated_kappa,
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

    # Initialize model params and training state.
    init_params = p3o_losses.PPONetworkParams(
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
        normalizer_params=running_statistics.init_state(
            _remove_pixels(obs_spec)
        ),
        env_steps=types.UInt64(hi=0, lo=0),
        kappa=jnp.array([initial_kappa], dtype=jnp.float32),
    )

    if restore_checkpoint_path is not None:
        params = checkpoint.load(restore_checkpoint_path)
        cost_value_params = params[2] if len(params) > 2 else init_params.cost_value
        value_params = params[1] if restore_value_fn else init_params.value
        kappa_value = params[3] if len(params) > 3 else jnp.array([initial_kappa], dtype=jnp.float32)
        training_state = training_state.replace(
            normalizer_params=params[0],
            params=training_state.params.replace(
                policy=params[1],
                value=value_params,
                cost_value=cost_value_params
            ),
            kappa=kappa_value,
        )

    # pretrained_params takes precedence over restore_params (for curriculum/transfer)
    effective_restore_params = pretrained_params if pretrained_params is not None else restore_params

    if effective_restore_params is not None:
        logging.info('Restoring TrainingState from pretrained/restore params.')
        # Handle transfer from PPO (no cost_value) or legacy checkpoints
        # Params tuple structure: (normalizer[0], policy[1], value[2], cost_value[3], kappa[4])
        has_cost_value = len(effective_restore_params) > 3

        if has_cost_value:
            cost_value_params = effective_restore_params[3]
        elif init_cost_value_from == 'value':
            logging.info('Initializing cost_value from value network (transfer mode).')
            cost_value_params = effective_restore_params[2]
        else:
            logging.info('Using random initialization for cost_value network.')
            cost_value_params = init_params.cost_value

        value_params = effective_restore_params[2] if restore_value_fn else init_params.value
        kappa_value = effective_restore_params[4] if len(effective_restore_params) > 4 else jnp.array([initial_kappa], dtype=jnp.float32)
        training_state = training_state.replace(
            normalizer_params=effective_restore_params[0],
            params=training_state.params.replace(
                policy=effective_restore_params[1],
                value=value_params,
                cost_value=cost_value_params
            ),
            kappa=kappa_value,
        )

    if num_timesteps == 0:
        return (
            make_policy,
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
                training_state.params.cost_value,
                training_state.kappa,
            ),
            {},
            None,  # eval_env
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
            # Add kappa (penalty coefficient) to params for checkpoint
            ckpt_params = params + (_unpmap(training_state.kappa),)
            checkpoint.save(
                save_checkpoint_path, current_step, ckpt_params, ckpt_config
            )

        if num_evals > 0 and evaluator is not None:
            metrics = evaluator.run_evaluation(
                params,
                training_metrics,
            )
            logging.info(metrics)
            progress_fn(current_step, metrics)

    total_steps = current_step
    if not total_steps >= num_timesteps:
        raise AssertionError(
            f'Total steps {total_steps} is less than `num_timesteps`='
            f' {num_timesteps}.'
        )

    pmap.assert_is_replicated(training_state)
    # Include kappa (penalty coefficient) for curriculum/transfer learning
    params = _unpmap((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
        training_state.params.cost_value,
        training_state.kappa,
    ))

    if not metrics:
        metrics = {'training/final_step': total_steps}
        if training_metrics:
            metrics.update(training_metrics)

    logging.info('total steps: %s', total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics, eval_env)
