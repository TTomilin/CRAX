# Copyright 2025 Safe-Brax Authors.
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

"""PPO-Saute trainer shim.

Pre-wraps the environment with SauteWrapper and delegates to standard PPO.
"""
from typing import Any, Callable, Optional, Tuple

from brax import envs
from brax.training.agents.ppo import train as ppo_train
from brax.envs.wrappers.saute import SauteWrapper

TrainReturn = Tuple[ppo_train.InferenceParams, ppo_train.Metrics]


def _apply_saute(env: envs.Env, **kwargs) -> envs.Env:
  return SauteWrapper(env, **kwargs)


def train(
    environment: envs.Env,
    num_timesteps: int,
    # Saute parameters:
    initial_budget: float = 1.0,
    gamma_budget: Optional[float] = None,  # defaults to PPO discounting if None
    termination_on_violation: bool = True,
    violation_penalty: float = 0.0,
    normalize_budget_obs: bool = True,
    # Standard PPO args:
    wrap_env: bool = True,
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.97,
    unroll_length: int = 10,
    batch_size: int = 1024,
    num_minibatches: int = 32,
    num_updates_per_batch: int = 2,
    normalize_observations: bool = True,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    seed: int = 0,
    progress_fn: Callable[[int, ppo_train.Metrics], None] = lambda *args: None,
    save_checkpoint_path: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    restore_params: Optional[Any] = None,
    restore_value_fn: bool = True,
    eval_env: Optional[envs.Env] = None,
    training_metrics_steps: Optional[float] = None,
    **kwargs,
) -> TrainReturn:
  # choose gamma for Saute; default to PPO discounting
  g_budget = discounting if gamma_budget is None else gamma_budget

  # Pre-wrap envs so vectorization happens after Saute
  env_for_training = _apply_saute(
      environment,
      initial_budget=initial_budget,
      gamma_budget=g_budget,
      termination_on_violation=termination_on_violation,
      violation_penalty=violation_penalty,
      normalize_budget_obs=normalize_budget_obs,
  ) if wrap_env else environment

  if eval_env is not None and wrap_env:
    eval_env = _apply_saute(
        eval_env,
        initial_budget=initial_budget,
        gamma_budget=g_budget,
        termination_on_violation=termination_on_violation,
        violation_penalty=violation_penalty,
        normalize_budget_obs=normalize_budget_obs,
    )

  return ppo_train.train(
      environment=env_for_training,
      num_timesteps=num_timesteps,
      wrap_env=wrap_env,
      num_envs=num_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      learning_rate=learning_rate,
      entropy_cost=entropy_cost,
      discounting=discounting,
      unroll_length=unroll_length,
      batch_size=batch_size,
      num_minibatches=num_minibatches,
      num_updates_per_batch=num_updates_per_batch,
      normalize_observations=normalize_observations,
      reward_scaling=reward_scaling,
      clipping_epsilon=clipping_epsilon,
      gae_lambda=gae_lambda,
      seed=seed,
      progress_fn=progress_fn,
      save_checkpoint_path=save_checkpoint_path,
      restore_checkpoint_path=restore_checkpoint_path,
      restore_params=restore_params,
      restore_value_fn=restore_value_fn,
      eval_env=eval_env,
      training_metrics_steps=training_metrics_steps,
      **kwargs,
  )
