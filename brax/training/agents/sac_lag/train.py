"""SAC-Lagrangian training.

Off-policy safe RL via Soft Actor-Critic + Lagrangian dual ascent.

The method maintains:
  - A reward Q-network Q(s,a)  (same as vanilla SAC)
  - A cost Q-network Qc(s,a)   (new)
  - A Lagrange multiplier λ ≥ 0 updated by projected gradient ascent

Policy objective (per gradient step):
  L_pi = E[alpha * log_prob - Q(s,a) + lambda * Qc(s,a)]

Lagrange multiplier update (per gradient step):
  lambda <- clip(max(0, lambda + (lr_lambda/episode_length) * (mean_cost - per_step_safety_bound)), 0, lambda_max)

  Dividing by episode_length normalises the per-step update so lagrangian_lr is
  on the same scale as PPO-Lag's lagrangian_coef_rate (both are effectively
  per-episode rates). Without this, lambda would grow ~episode_length times
  faster than PPO-Lag with the same lr value, killing all reward learning.

References:
  - SAC:      https://arxiv.org/pdf/1812.05905.pdf
  - CMDP/Lag: https://arxiv.org/pdf/2001.06684.pdf  (Ray et al., 2019)
"""

import functools
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import logger as metric_logger
from brax.training import pmap
from brax.training import replay_buffers
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.sac_lag import losses as sac_lag_losses
from brax.training.agents.sac_lag import networks as sac_lag_networks
from brax.training.types import Params, PRNGKey
import flax
import jax
import jax.numpy as jnp
import optax

Metrics = types.Metrics
Transition = types.Transition
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
ReplayBufferState = Any

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the SAC-Lag learner."""
    policy_optimizer_state: optax.OptState
    policy_params: Params
    q_optimizer_state: optax.OptState
    q_params: Params
    target_q_params: Params
    qc_optimizer_state: optax.OptState
    qc_params: Params
    target_qc_params: Params
    gradient_steps: types.UInt64
    env_steps: types.UInt64
    alpha_optimizer_state: optax.OptState
    alpha_params: Params
    lambda_lagr: jnp.ndarray   # Lagrange multiplier (scalar, >= 0)
    normalizer_params: running_statistics.RunningStatisticsState


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    local_devices_to_use: int,
    sac_network: sac_lag_networks.SACLagNetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    qc_optimizer: optax.GradientTransformation,
    initial_lambda: float,
) -> TrainingState:
    """Initialises the training state and replicates it over devices."""
    key_policy, key_q, key_qc = jax.random.split(key, 3)
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    policy_params = sac_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)

    q_params = sac_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)

    qc_params = sac_network.qc_network.init(key_qc)
    qc_optimizer_state = qc_optimizer.init(qc_params)

    normalizer_params = running_statistics.init_state(
        specs.Array((obs_size,), jnp.dtype('float32'))
    )
    lambda_lagr = jnp.asarray(initial_lambda, dtype=jnp.float32)

    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        qc_optimizer_state=qc_optimizer_state,
        qc_params=qc_params,
        target_qc_params=qc_params,
        gradient_steps=types.UInt64(hi=0, lo=0),
        env_steps=types.UInt64(hi=0, lo=0),
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=log_alpha,
        lambda_lagr=lambda_lagr,
        normalizer_params=normalizer_params,
    )
    return jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )


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
    discounting: float = 0.9,
    seed: int = 0,
    batch_size: int = 256,
    num_evals: int = 1,
    normalize_observations: bool = False,
    max_devices_per_host: Optional[int] = None,
    reward_scaling: float = 1.0,
    tau: float = 0.005,
    min_replay_size: int = 0,
    max_replay_size: Optional[int] = None,
    grad_updates_per_step: int = 1,
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
    # SAC-Lag specific
    safety_bound: float = 25.0,
    lagrangian_lr: float = 0.01,
    initial_lambda: float = 0.0,
    lambda_max: float = 2.0,
):
    """SAC-Lagrangian training.

    Args:
      environment: the environment to train.
      num_timesteps: total number of environment steps.
      episode_length: length of each episode (used to convert episodic safety
        bound to per-step bound).
      wrap_env: whether to wrap the environment for training.
      wrap_env_fn: optional custom environment wrapper.
      action_repeat: number of times each action is repeated.
      num_envs: number of parallel environments.
      num_eval_envs: number of environments used for evaluation.
      learning_rate: learning rate for policy, Q, and Qc networks.
      discounting: discount factor gamma.
      seed: random seed.
      batch_size: SGD mini-batch size.
      num_evals: number of evaluation checkpoints.
      normalize_observations: whether to z-score observations.
      max_devices_per_host: maximum number of accelerators to use.
      reward_scaling: scalar multiplier on environment rewards.
      tau: soft target-network update coefficient.
      min_replay_size: minimum replay buffer size before training starts.
      max_replay_size: maximum replay buffer capacity.
      grad_updates_per_step: gradient updates per environment step.
      deterministic_eval: use deterministic policy during evaluation.
      network_factory: factory for SAC-Lag networks.
      progress_fn: callback invoked at each eval checkpoint.
      eval_env: optional separate environment for evaluation.
      randomization_fn: optional domain-randomisation callback.
      safety_bound: episodic cumulative cost limit (constraint threshold d).
      lagrangian_lr: learning rate for the Lagrange multiplier. Normalized
        internally by episode_length so the effective per-episode update rate is
        comparable to PPO-Lag's lagrangian_coef_rate.
      initial_lambda: initial value of the Lagrange multiplier.
      lambda_max: upper bound on the Lagrange multiplier (prevents runaway).

    Returns:
      Tuple of (make_policy, params, metrics).
    """
    process_id = jax.process_index()
    local_devices_to_use = jax.local_device_count()
    if max_devices_per_host is not None:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    device_count = local_devices_to_use * jax.process_count()

    if min_replay_size >= num_timesteps:
        raise ValueError(
            'No training will happen because min_replay_size >= num_timesteps'
        )
    if max_replay_size is None:
        max_replay_size = num_timesteps
    num_timesteps = int(num_timesteps)
    max_replay_size = int(max_replay_size)
    min_replay_size = int(min_replay_size)

    # Convert episodic safety bound to per-step bound
    per_step_safety_bound = safety_bound / episode_length if episode_length else safety_bound

    # Normalize lambda lr by episode_length so lagrangian_lr is on the same
    # per-episode scale as PPO-Lag's lagrangian_coef_rate. Without this,
    # SAC-Lag updates lambda every gradient step (~250K times for 5e8 steps)
    # whereas PPO-Lag updates once per rollout, causing lambda to explode.
    effective_lagrangian_lr = lagrangian_lr / max(episode_length or 1, 1)

    env_steps_per_actor_step = action_repeat * num_envs
    num_prefill_actor_steps = -(-min_replay_size // num_envs)
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    assert num_timesteps - num_prefill_env_steps >= 0
    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = -(
        -(num_timesteps - num_prefill_env_steps)
        // (num_evals_after_init * env_steps_per_actor_step)
    )

    assert num_envs % device_count == 0
    env = environment
    if wrap_env:
        if wrap_env_fn is not None:
            wrap_for_training = wrap_env_fn
        elif isinstance(env, envs.Env):
            wrap_for_training = envs.training.wrap
        else:
            raise ValueError('Unsupported environment type: %s' % type(env))

        rng = jax.random.PRNGKey(seed)
        rng, key = jax.random.split(rng)
        v_randomization_fn = None
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn,
                rng=jax.random.split(
                    key, num_envs // jax.process_count() // local_devices_to_use
                ),
            )
        env = wrap_for_training(
            env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=v_randomization_fn,
        )

    obs_size = env.observation_size
    if isinstance(obs_size, Dict):
        raise NotImplementedError('Dictionary observations not supported in SAC-Lag')
    action_size = env.action_size

    normalize_fn = lambda x, y: x
    if normalize_observations:
        normalize_fn = running_statistics.normalize

    sac_network = network_factory(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
    )
    make_policy = sac_lag_networks.make_inference_fn(sac_network)

    alpha_optimizer = optax.adam(learning_rate=3e-4)
    policy_optimizer = optax.adam(learning_rate=learning_rate)
    q_optimizer = optax.adam(learning_rate=learning_rate)
    qc_optimizer = optax.adam(learning_rate=learning_rate)

    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        next_observation=dummy_obs,
        extras={
            'state_extras': {'truncation': 0.0, 'cost': 0.0},
            'policy_extras': {},
        },
    )
    replay_buffer = replay_buffers.UniformSamplingQueue(
        max_replay_size=max_replay_size // device_count,
        dummy_data_sample=dummy_transition,
        sample_batch_size=batch_size * grad_updates_per_step // device_count,
    )

    metrics_aggregator = metric_logger.MetricsLogger(
        buffer_size=10,
        steps_between_logging=int(training_metrics_steps),
        progress_fn=progress_fn,
    )

    alpha_loss, critic_loss, cost_critic_loss, actor_loss = sac_lag_losses.make_losses(
        sac_network=sac_network,
        reward_scaling=reward_scaling,
        discounting=discounting,
        action_size=action_size,
    )
    alpha_update = gradients.gradient_update_fn(
        alpha_loss, alpha_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    critic_update = gradients.gradient_update_fn(
        critic_loss, q_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    cost_critic_update = gradients.gradient_update_fn(
        cost_critic_loss, qc_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )
    actor_update = gradients.gradient_update_fn(
        actor_loss, policy_optimizer, pmap_axis_name=_PMAP_AXIS_NAME
    )

    def sgd_step(
        carry: Tuple[TrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry
        key, key_alpha, key_critic, key_cost_critic, key_actor = jax.random.split(key, 5)

        # --- Entropy temperature update (unchanged from SAC) ---
        alpha_loss_val, alpha_params, alpha_optimizer_state = alpha_update(
            training_state.alpha_params,
            training_state.policy_params,
            training_state.normalizer_params,
            transitions,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state,
        )
        alpha = jnp.exp(training_state.alpha_params)

        # --- Reward critic update (unchanged from SAC) ---
        critic_loss_val, q_params, q_optimizer_state = critic_update(
            training_state.q_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_q_params,
            alpha,
            transitions,
            key_critic,
            optimizer_state=training_state.q_optimizer_state,
        )

        # --- Cost critic update (new) ---
        cost_critic_loss_val, qc_params, qc_optimizer_state = cost_critic_update(
            training_state.qc_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_qc_params,
            transitions,
            key_cost_critic,
            optimizer_state=training_state.qc_optimizer_state,
        )

        # --- Lagrange-penalised actor update ---
        actor_loss_val, policy_params, policy_optimizer_state = actor_update(
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.q_params,
            training_state.qc_params,
            alpha,
            training_state.lambda_lagr,
            transitions,
            key_actor,
            optimizer_state=training_state.policy_optimizer_state,
        )

        # --- Soft target-network updates ---
        new_target_q_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau,
            training_state.target_q_params, q_params,
        )
        new_target_qc_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau,
            training_state.target_qc_params, qc_params,
        )

        # --- Lagrange multiplier update (projected gradient ascent on dual) ---
        # Use the mean per-step cost from the current batch as the constraint signal.
        # effective_lagrangian_lr = lagrangian_lr / episode_length normalises the
        # per-gradient-step update so that over one episode's worth of gradient
        # steps the total lambda shift matches PPO-Lag's per-rollout update.
        mean_cost = jnp.mean(transitions.extras['state_extras']['cost'])
        cost_violation = mean_cost - per_step_safety_bound
        new_lambda_lagr = jnp.minimum(
            jax.nn.relu(training_state.lambda_lagr + effective_lagrangian_lr * cost_violation),
            jnp.asarray(lambda_max, jnp.float32),
        )

        metrics = {
            'critic_loss': critic_loss_val,
            'cost_critic_loss': cost_critic_loss_val,
            'actor_loss': actor_loss_val,
            'alpha_loss': alpha_loss_val,
            'alpha': jnp.exp(alpha_params),
            'lambda_lagr': new_lambda_lagr,
            'mean_cost': mean_cost,
            'cost_violation': cost_violation,
        }

        new_training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            target_q_params=new_target_q_params,
            qc_optimizer_state=qc_optimizer_state,
            qc_params=qc_params,
            target_qc_params=new_target_qc_params,
            gradient_steps=training_state.gradient_steps + 1,
            env_steps=training_state.env_steps,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            lambda_lagr=new_lambda_lagr,
            normalizer_params=training_state.normalizer_params,
        )
        return (new_training_state, key), metrics

    def get_experience(
        normalizer_params: running_statistics.RunningStatisticsState,
        policy_params: Params,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[running_statistics.RunningStatisticsState, envs.State, ReplayBufferState]:
        policy = make_policy((normalizer_params, policy_params))
        env_state, transitions = acting.actor_step(
            env, env_state, policy, key, extra_fields=('truncation', 'cost')
        )
        normalizer_params = running_statistics.update(
            normalizer_params,
            transitions.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )
        buffer_state = replay_buffer.insert(buffer_state, transitions)
        return normalizer_params, env_state, buffer_state

    def training_step(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        experience_key, training_key = jax.random.split(key)
        normalizer_params, env_state, buffer_state = get_experience(
            training_state.normalizer_params,
            training_state.policy_params,
            env_state,
            buffer_state,
            experience_key,
        )
        new_env_steps = training_state.env_steps + env_steps_per_actor_step
        # Log episodic metrics (sum_reward, length, cost, etc.) for any
        # episodes that completed this step, matching PPO's MetricsLogger pattern.
        jax.debug.callback(
            metrics_aggregator.update_env_metrics,
            env_state.info['episode_metrics'],
            env_state.info['episode_done'],
            new_env_steps,
        )
        training_state = training_state.replace(
            normalizer_params=normalizer_params,
            env_steps=new_env_steps,
        )
        buffer_state, transitions = replay_buffer.sample(buffer_state)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (grad_updates_per_step, -1) + x.shape[1:]),
            transitions,
        )
        (training_state, _), metrics = jax.lax.scan(
            sgd_step, (training_state, training_key), transitions
        )
        metrics['buffer_current_size'] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

    def prefill_replay_buffer(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            new_normalizer_params, env_state, buffer_state = get_experience(
                training_state.normalizer_params,
                training_state.policy_params,
                env_state,
                buffer_state,
                key,
            )
            new_training_state = training_state.replace(
                normalizer_params=new_normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )
            return (new_training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_prefill_actor_steps,
        )[0]

    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=_PMAP_AXIS_NAME)

    def training_epoch(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, new_key = jax.random.split(k)
            ts, es, bs, metrics = training_step(ts, es, bs, k)
            return (ts, es, bs, new_key), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, env_state, buffer_state, metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        nonlocal training_walltime
        t = time.time()
        (training_state, env_state, buffer_state, metrics) = training_epoch(
            training_state, env_state, buffer_state, key
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            'training/sps': sps,
            'training/walltime': training_walltime,
            **{f'training/{name}': value for name, value in metrics.items()},
        }
        return training_state, env_state, buffer_state, metrics

    global_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)

    training_state = _init_training_state(
        key=global_key,
        obs_size=obs_size,
        local_devices_to_use=local_devices_to_use,
        sac_network=sac_network,
        alpha_optimizer=alpha_optimizer,
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
        qc_optimizer=qc_optimizer,
        initial_lambda=initial_lambda,
    )
    del global_key

    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

    env_keys = jax.random.split(env_key, num_envs // jax.process_count())
    env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
    env_state = jax.pmap(env.reset)(env_keys)

    buffer_state = jax.pmap(replay_buffer.init)(
        jax.random.split(rb_key, local_devices_to_use)
    )

    if not eval_env:
        eval_env = environment
    if wrap_env:
        v_randomization_fn = None
        if randomization_fn is not None:
            v_randomization_fn = functools.partial(
                randomization_fn, rng=jax.random.split(eval_key, num_eval_envs)
            )
        eval_env = wrap_for_training(
            eval_env,
            episode_length=episode_length,
            action_repeat=action_repeat,
            randomization_fn=v_randomization_fn,
        )

    evaluator = acting.Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        metrics = evaluator.run_evaluation(
            _unpmap((training_state.normalizer_params, training_state.policy_params)),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    # Prefill replay buffer
    t = time.time()
    prefill_key, local_key = jax.random.split(local_key)
    prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_keys
    )

    replay_size = (
        jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
    )
    logging.info('replay size after prefill %s', replay_size)
    assert replay_size >= min_replay_size
    training_walltime = time.time() - t

    current_step = 0
    for _ in range(num_evals_after_init):
        logging.info('step %s', current_step)

        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
        (training_state, env_state, buffer_state, training_metrics) = (
            training_epoch_with_timing(
                training_state, env_state, buffer_state, epoch_keys
            )
        )
        current_step = int(_unpmap(training_state.env_steps))

        # Always log training metrics (losses, lambda, etc.) after every epoch,
        # matching PPO's pattern of calling progress_fn unconditionally.
        if process_id == 0:
            progress_fn(current_step, training_metrics)

        if process_id == 0 and num_evals > 0:
            metrics = evaluator.run_evaluation(
                _unpmap(
                    (training_state.normalizer_params, training_state.policy_params)
                ),
                training_metrics,
            )
            logging.info(metrics)
            progress_fn(current_step, metrics)

    total_steps = current_step
    if not total_steps >= num_timesteps:
        raise AssertionError(
            f'Total steps {total_steps} is less than `num_timesteps`={num_timesteps}.'
        )

    params = _unpmap(
        (training_state.normalizer_params, training_state.policy_params)
    )
    pmap.assert_is_replicated(training_state)
    logging.info('total steps: %s', total_steps)
    pmap.synchronize_hosts()
    return make_policy, params, metrics, eval_env
