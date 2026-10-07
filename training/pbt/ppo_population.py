"""Round-based PPO learner for population-based training.

`training.agents.ppo.train.train` runs one agent end-to-end with its
hyperparameters baked into the compiled graph as Python constants. PBT instead
trains many agents for short rounds and changes their hyperparameters between
rounds, so this module rebuilds the same single-device PPO update (rollout ->
normaliser update -> epochs of minibatch SGD, `jit(vmap(...))` over a leading
device axis of size 1) with every tunable hyperparameter passed in as a traced
scalar. One compilation then serves every agent and every hyperparameter
value: agents share the trainer and only differ in the `TrainingState`, env
state and hyperparameter dict they pass in.

Agents are trained one after another on the process' GPU. Batching agents
with `vmap` (as the Brax MF-PBT reference does) is not an option for pixel
observations: the MJWarp renderer needs a statically-sized, un-vmapped batch
(see crax/envs/wrappers/pixel_observation_gpu.py), and a single vision agent
with a few thousand envs already saturates the GPU.
"""

import functools
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from crax import envs
from training import acting
from training import types
from training.acme import running_statistics
from training.acme import specs
from training.agents.ppo import losses as ppo_losses
from training.agents.ppo import networks as ppo_networks
from training.agents.ppo.train import (
    TrainingState, _maybe_wrap_env, _random_translate_pixels, _remove_pixels, _strip_weak_type, _unpmap,
)
from training.types import PRNGKey

# Hyperparameters that are inputs of the compiled round (so PBT may change them freely).
DYNAMIC_HPARAMS = (
    'learning_rate', 'entropy_cost', 'clipping_epsilon', 'discounting', 'gae_lambda', 'reward_scaling',
)

# Agent state exported to / imported from the population store (numpy, no device axis).
AgentBlob = Dict[str, Any]


class PPOPopulationTrainer:

    def __init__(
            self,
            environment: envs.Env,
            *,
            steps_per_round: int,
            episode_length: int,
            num_envs: int,
            unroll_length: int,
            batch_size: int,
            num_minibatches: int,
            num_updates_per_batch: int,
            network_factory: types.NetworkFactory[ppo_networks.PPONetworks] = ppo_networks.make_ppo_networks,
            eval_env: Optional[envs.Env] = None,
            num_eval_envs: int = 128,
            deterministic_eval: bool = False,
            normalize_observations: bool = True,
            normalize_advantage: bool = True,
            max_grad_norm: Optional[float] = None,
            augment_pixels: bool = False,
            vision_kwargs: Optional[Dict[str, Any]] = None,
            action_repeat: int = 1,
            seed: int = 0,
            extra_fields: Tuple[str, ...] = ('truncation', 'episode_metrics', 'episode_done'),
    ):
        assert batch_size * num_minibatches % num_envs == 0, (
            f"batch_size * num_minibatches ({batch_size * num_minibatches}) must be divisible by "
            f"num_envs ({num_envs})"
        )
        self.num_envs = num_envs
        self.env_step_per_training_step = batch_size * unroll_length * num_minibatches * action_repeat
        self.num_training_steps_per_round = int(np.ceil(steps_per_round / self.env_step_per_training_step))
        # What a round actually costs, after rounding up to whole training steps.
        self.steps_per_round = self.num_training_steps_per_round * self.env_step_per_training_step
        self.normalize_observations = normalize_observations
        self.network_factory = network_factory

        key_env, key_eval = jax.random.split(jax.random.PRNGKey(seed))
        self.env = _maybe_wrap_env(
            environment, True, num_envs, episode_length, action_repeat,
            device_count=1, key_env=key_env, vision_kwargs=vision_kwargs,
        )
        self._reset_fn = jax.jit(jax.vmap(self.env.reset))
        env_state = self.reset(key_env)
        # Discard the device and env batch axes.
        self.obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)
        self._obs_spec = jax.tree_util.tree_map(
            lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
        )
        del env_state

        normalize = running_statistics.normalize if normalize_observations else (lambda x, y: x)
        ppo_network = network_factory(self.obs_shape, self.env.action_size, preprocess_observations_fn=normalize)
        self.ppo_network = ppo_network
        self.make_policy = ppo_networks.make_inference_fn(ppo_network)

        # Adam without its learning-rate scaling; the (traced) learning rate is applied per update.
        self._optimizer = optax.scale_by_adam()
        if max_grad_norm is not None:
            self._optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), self._optimizer)
        optimizer = self._optimizer

        def loss_fn(params, normalizer_params, data, key, hp):
            return ppo_losses.compute_ppo_loss(
                params, normalizer_params, data, key,
                ppo_network=ppo_network,
                entropy_cost=hp['entropy_cost'],
                discounting=hp['discounting'],
                reward_scaling=hp['reward_scaling'],
                gae_lambda=hp['gae_lambda'],
                clipping_epsilon=hp['clipping_epsilon'],
                normalize_advantage=normalize_advantage,
            )

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        def minibatch_step(carry, data: types.Transition, normalizer_params, hp):
            optimizer_state, params, key = carry
            key, key_loss = jax.random.split(key)
            (_, metrics), grads = grad_fn(params, normalizer_params, data, key_loss, hp)
            updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
            updates = jax.tree_util.tree_map(lambda u: -hp['learning_rate'] * u, updates)
            params = optax.apply_updates(params, updates)
            metrics = {**metrics, 'grad_norm': optax.global_norm(grads)}
            return (optimizer_state, params, key), metrics

        def sgd_step(carry, unused_t, data: types.Transition, normalizer_params, hp):
            optimizer_state, params, key = carry
            key, key_perm, key_grad = jax.random.split(key, 3)

            if augment_pixels:
                key, key_rt = jax.random.split(key)
                r_translate = functools.partial(_random_translate_pixels, key=key_rt)
                data = data._replace(
                    observation=r_translate(data.observation),
                    next_observation=r_translate(data.next_observation),
                )

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(key_perm, x)
                return jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])

            shuffled_data = jax.tree_util.tree_map(convert_data, data)
            (optimizer_state, params, _), metrics = jax.lax.scan(
                functools.partial(minibatch_step, normalizer_params=normalizer_params, hp=hp),
                (optimizer_state, params, key_grad),
                shuffled_data,
                length=num_minibatches,
            )
            return (optimizer_state, params, key), metrics

        env = self.env
        env_step_per_training_step = self.env_step_per_training_step
        policy_params_tuple = self._policy_params_tuple

        def training_step(carry, unused_t, hp):
            training_state, state, key = carry
            key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)
            policy = self.make_policy(policy_params_tuple(training_state))

            def f(carry, unused_t):
                current_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                next_state, data = acting.generate_unroll(
                    env, current_state, policy, current_key, unroll_length, extra_fields=extra_fields,
                )
                return (next_state, next_key), data

            (state, _), data = jax.lax.scan(
                f, (state, key_generate_unroll), (), length=batch_size * num_minibatches // num_envs,
            )
            # Have leading dimensions (batch_size * num_minibatches, unroll_length)
            data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
            data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data)
            assert data.discount.shape[1:] == (unroll_length,)

            # Totals of the episodes that finished during this step (the per-round
            # equivalent of MetricsLogger.update_env_metrics, kept on device).
            done = data.extras['state_extras']['episode_done'].astype(jnp.float32)
            episode_sums = {
                name: jnp.sum(value * done)
                for name, value in data.extras['state_extras']['episode_metrics'].items()
            }

            normalizer_params = running_statistics.update(
                training_state.normalizer_params, _remove_pixels(data.observation),
            )
            (optimizer_state, params, _), metrics = jax.lax.scan(
                functools.partial(sgd_step, data=data, normalizer_params=normalizer_params, hp=hp),
                (training_state.optimizer_state, training_state.params, key_sgd),
                (),
                length=num_updates_per_batch,
            )
            new_training_state = training_state.replace(
                optimizer_state=optimizer_state,
                params=params,
                normalizer_params=normalizer_params,
                env_steps=training_state.env_steps + env_step_per_training_step,
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            return (new_training_state, state, new_key), (metrics, episode_sums, jnp.sum(done))

        def training_round(training_state, state, key, hp):
            (training_state, state, _), (metrics, episode_sums, num_episodes) = jax.lax.scan(
                functools.partial(training_step, hp=hp),
                (training_state, state, key),
                (),
                length=self.num_training_steps_per_round,
            )
            round_metrics = {
                'loss_metrics': jax.tree_util.tree_map(jnp.mean, metrics),
                # Last training step only: closest to the policy the round hands on.
                'loss_metrics_last': jax.tree_util.tree_map(lambda x: x[-1], metrics),
                'episode_sums': jax.tree_util.tree_map(jnp.sum, episode_sums),
                'num_episodes': jnp.sum(num_episodes),
            }
            return training_state, state, round_metrics

        # Single device: vmap over the leading device dim (size 1), then jit,
        # exactly like the non-pmap path of ppo.train.
        self._training_round = jax.jit(jax.vmap(training_round))

        self._evaluator = None
        if num_eval_envs > 0:
            wrapped_eval_env = _maybe_wrap_env(
                eval_env or environment, True, num_eval_envs, episode_length, action_repeat,
                device_count=1, key_env=key_eval, vision_kwargs=vision_kwargs,
            )
            self._evaluator = acting.Evaluator(
                wrapped_eval_env,
                functools.partial(self.make_policy, deterministic=deterministic_eval),
                num_eval_envs=num_eval_envs,
                episode_length=episode_length,
                action_repeat=action_repeat,
                key=key_eval,
            )

    # --- Agent state ----------------------------------------------------------

    def _policy_params_tuple(self, state: TrainingState) -> Tuple[Any, ...]:
        """(normalizer, policy, value[, shared encoder]), as ppo.train's make_policy expects."""
        base = (state.normalizer_params, state.params.policy, state.params.value)
        if self.ppo_network.encoder_network is not None:
            return base + (state.params.encoder,)
        return base

    def policy_params(self, training_state: TrainingState) -> Tuple[Any, ...]:
        """Inference params (no device axis) for checkpointing / videos."""
        return _unpmap(self._policy_params_tuple(training_state))

    def reset(self, key: PRNGKey) -> envs.State:
        keys = jax.random.split(key, self.num_envs)
        return self._reset_fn(jnp.reshape(keys, (1, -1) + keys.shape[1:]))

    def init_agent(self, key: PRNGKey) -> TrainingState:
        key_policy, key_value, key_cost_value, key_encoder = jax.random.split(key, 4)
        net = self.ppo_network
        params = ppo_losses.PPONetworkParams(
            policy=net.policy_network.init(key_policy),
            value=net.value_network.init(key_value),
            cost_value=net.cost_value_network.init(key_cost_value) if net.cost_value_network is not None else None,
            encoder=net.encoder_network.init(key_encoder) if net.encoder_network is not None else None,
        )
        training_state = TrainingState(
            optimizer_state=self._optimizer.init(params),
            params=params,
            normalizer_params=running_statistics.init_state(_remove_pixels(self._obs_spec)),
            env_steps=types.UInt64(hi=0, lo=0),
        )
        return self._add_device_axis(training_state)

    @staticmethod
    def _add_device_axis(tree):
        return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (1,) + jnp.shape(x)), tree)

    def export_agent(self, training_state: TrainingState) -> AgentBlob:
        state = jax.device_get(_unpmap(training_state))
        return {
            'params': state.params,
            'normalizer_params': state.normalizer_params,
            'optimizer_state': state.optimizer_state,
        }

    def import_agent(self, blob: AgentBlob, env_steps: int) -> TrainingState:
        """Rebuilds a TrainingState from a stored blob, keeping the importing slot's own step count."""
        training_state = TrainingState(
            optimizer_state=blob['optimizer_state'],
            params=blob['params'],
            normalizer_params=blob['normalizer_params'],
            env_steps=types.UInt64(hi=int(env_steps) >> 32, lo=int(env_steps) & 0xFFFFFFFF),
        )
        return self._add_device_axis(jax.tree_util.tree_map(jnp.asarray, training_state))

    # --- Training / evaluation -----------------------------------------------

    def train_round(
            self,
            training_state: TrainingState,
            env_state: envs.State,
            hparams: Mapping[str, float],
            key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, Dict[str, float]]:
        """Trains one agent for `steps_per_round` env steps with the given hyperparameters."""
        missing = [h for h in DYNAMIC_HPARAMS if h not in hparams]
        if missing:
            raise ValueError(f"Missing hyperparameters {missing}")
        hp = {h: jnp.full((1,), hparams[h], dtype=jnp.float32) for h in DYNAMIC_HPARAMS}

        t = time.time()
        training_state, env_state = _strip_weak_type((training_state, env_state))
        training_state, env_state, raw = self._training_round(
            training_state, env_state, jnp.reshape(key, (1,) + key.shape), hp,
        )
        training_state, env_state = _strip_weak_type((training_state, env_state))
        raw = jax.device_get(_unpmap(raw))
        elapsed = time.time() - t

        metrics = {f'training/{k}': float(v) for k, v in raw['loss_metrics'].items()}
        metrics.update({f'training/last_{k}': float(v) for k, v in raw['loss_metrics_last'].items()})
        num_episodes = float(raw['num_episodes'])
        metrics['training/num_episodes'] = num_episodes
        if num_episodes > 0:
            metrics.update({f'episodic/{k}': float(v) / num_episodes for k, v in raw['episode_sums'].items()})
        metrics['training/round_time'] = elapsed
        metrics['training/sps'] = self.steps_per_round / elapsed
        return training_state, env_state, metrics

    def evaluate(self, training_state: TrainingState, key: Optional[PRNGKey] = None) -> Dict[str, float]:
        """Runs `num_eval_envs` full episodes. Passing the same `key` to every agent of a round
        evaluates them all from the same initial states (common random numbers)."""
        if self._evaluator is None:
            return {}
        if key is not None:
            self._evaluator._key = key
        metrics = self._evaluator.run_evaluation(self.policy_params(training_state), training_metrics={})
        # eval/walltime accumulates over every agent this evaluator scored, so it means nothing per agent.
        return {k: float(np.asarray(v)) for k, v in metrics.items() if k != 'eval/walltime'}

    def network_config(self):
        from training.agents.ppo import checkpoint
        return checkpoint.network_config(
            observation_size=self.obs_shape,
            action_size=self.env.action_size,
            normalize_observations=self.normalize_observations,
            network_factory=self.network_factory,
        )
