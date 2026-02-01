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

# pylint:disable=g-multiple-import
"""Environments for training and evaluating policies."""

from typing import Optional, Type

import jax
from jax import numpy as jp

from brax.envs import ant, humanoid_hop
from brax.envs import fast
from brax.envs import half_cheetah
from brax.envs import hopper
from brax.envs import humanoid
from brax.envs import safe_height
from brax.envs import humanoidstandup
from brax.envs import inverted_double_pendulum
from brax.envs import inverted_pendulum
from brax.envs import pusher
from brax.envs import reacher
from brax.envs import safe_reacher
from brax.envs import safe_velocity
from brax.envs import swimmer
from brax.envs import walker2d
from brax.envs import safe_ant
from brax.envs import safe_spider
from brax.envs import safe_walker
from brax.envs.PointResettingGoalRandomHazardLidarSensorObs import PointResettingGoalRandomHazardLidarSensorObs
from brax.envs.PointResettingGoalRandomHazardSensorObs import PointResettingGoalRandomHazardSensorObs
from brax.envs.safe_goal import SafePointGoal
from brax.envs.safe_circle import SafePointCircle
from brax.envs.safe_push import SafePush
from brax.envs.base import Env, PipelineEnv, State, Wrapper
from brax.envs.wrappers import training
from brax.envs.difficulty import apply_difficulty, get_supported_levels, supports_difficulty

_envs = {
    'ant': ant.Ant,
    'fast': fast.Fast,
    'halfcheetah': half_cheetah.Halfcheetah,
    'hopper': hopper.Hopper,
    'humanoid': humanoid.Humanoid,
    'humanoid_hop': humanoid_hop.Humanoid,
    'safe_height': safe_height.SafeHeightHumanoid,
    'humanoidstandup': humanoidstandup.HumanoidStandup,
    'inverted_pendulum': inverted_pendulum.InvertedPendulum,
    'inverted_double_pendulum': inverted_double_pendulum.InvertedDoublePendulum,
    'pusher': pusher.Pusher,
    'reacher': reacher.Reacher,
    'safe_reacher': safe_reacher.SafeReacher,
    'safe_velocity': safe_velocity.SafeVelocity,
    'swimmer': swimmer.Swimmer,
    'walker2d': walker2d.Walker2d,
    'safe_walker': safe_walker.SafeWalker,
    'safe_ant': safe_ant.SafeAnt,
    'safe_spider': safe_spider.SafeSpider,
    'point_resetting_goal_random_hazard_sensor_obs': PointResettingGoalRandomHazardSensorObs,
    'point_resetting_goal_random_hazard_lidar_sensor_obs': PointResettingGoalRandomHazardLidarSensorObs,
    'safe_point_goal': SafePointGoal,
    'safe_point_circle': SafePointCircle,
    'block_push_goal': SafePush,
}


class UnifiedEnvAdapter(Wrapper):
    """A wrapper that provides a unified safety-compatible interface.

    - Ensures a 'cost' signal is present in state.info and state.metrics.
    - Preserves original reward as 'raw_reward' in state.info and metrics.
    - Adds 'shaped_reward' (identical to reward unless inner env sets it).
    - Stores any arbitrary kwargs for future use; does not change inner env.
    """

    def __init__(self, env: Env, **unified_kwargs):
        super().__init__(env)
        # Store unified kwargs for forward compatibility (e.g., physics/cost/reward specs)
        self._unified_kwargs = unified_kwargs or {}

    def _ensure_unified_fields(self, state: State) -> State:
        # Ensure cost exists in info and metrics
        cost = state.info.get('cost', state.metrics.get('cost', None))
        if cost is None:
            cost = jp.zeros_like(state.reward)
        # mutate copies of dicts
        info = dict(state.info)
        metrics = dict(state.metrics)
        info.setdefault('cost', cost)
        metrics.setdefault('cost', cost)

        # Preserve rewards metadata
        info.setdefault('raw_reward', state.reward)
        info.setdefault('shaped_reward', state.reward)

        # Recreate a new State with updated dicts
        return State(
            pipeline_state=state.pipeline_state,
            obs=state.obs,
            reward=state.reward,
            done=state.done,
            metrics=metrics,
            info=info,
        )

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        return self._ensure_unified_fields(state)

    def step(self, state: State, action: jax.Array) -> State:
        next_state = self.env.step(state, action)
        return self._ensure_unified_fields(next_state)


def get_environment(env_name: str, level: Optional[int] = None, **kwargs) -> Env:
    """Returns an environment from the environment registry.

    Args:
      env_name: environment name string
      level: optional difficulty level (1, 2, 3). If provided, applies difficulty
             overrides for supported environments.
      **kwargs: keyword arguments that get passed to the Env class constructor

    Returns:
      env: an environment
    """
    # Apply difficulty overrides if level is specified
    if level is not None:
        kwargs = apply_difficulty(env_name, kwargs, level)

    env_cls = _envs[env_name]
    base_env = env_cls(**kwargs)
    # Always wrap with unified adapter so downstream code can rely on cost/info fields
    return UnifiedEnvAdapter(base_env, **kwargs)


def register_environment(env_name: str, env_class: Type[Env]):
    """Adds an environment to the registry.

    Args:
      env_name: environment name string
      env_class: the Env class to add to the registry
    """
    _envs[env_name] = env_class


def create(
        env_name: str,
        episode_length: Optional[int] = None,
        action_repeat: int = 1,
        auto_reset: bool = True,
        batch_size: Optional[int] = None,
        level: Optional[int] = None,
        **kwargs,
) -> Env:
    """Creates an environment from the registry.

    Args:
      env_name: environment name string
      episode_length: length of episode
      action_repeat: how many repeated actions to take per environment step
      auto_reset: whether to auto reset the environment after an episode is done
      batch_size: the number of environments to batch together
      level: optional difficulty level (1, 2, 3). If provided, applies difficulty
             overrides for supported environments.
      **kwargs: keyword arguments that get passed to the Env class constructor

    Returns:
      env: an environment
    """
    # Apply difficulty overrides if level is specified
    if level is not None:
        kwargs = apply_difficulty(env_name, kwargs, level)

    env_cls = _envs[env_name]
    try:
        base_env = env_cls(**kwargs)
    except TypeError:
        base_env = env_cls()

    env = UnifiedEnvAdapter(base_env, **kwargs)

    # Resolve episode length: prefer explicit, else environment default if available
    if episode_length is None:
        episode_length = getattr(env, 'episode_length', None)
    if episode_length is not None:
        env = training.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = training.VmapWrapper(env, batch_size)
    if auto_reset:
        env = training.AutoResetWrapper(env)

    return env
