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

from typing import Any, Dict, Optional, Type

import jax
from jax import numpy as jp

from brax.envs import ant
from brax.envs import fast
from brax.envs import half_cheetah
from brax.envs import hopper
from brax.envs import humanoid
from brax.envs import humanoidstandup
from brax.envs import inverted_double_pendulum
from brax.envs import inverted_pendulum
from brax.envs import pusher
from brax.envs import reacher
from brax.envs import safe_ant
from brax.envs import safe_height
from brax.envs import safe_lift
from brax.envs import safe_reacher
from brax.envs import safe_spider
from brax.envs import safe_velocity
from brax.envs import safe_pathway
from brax.envs import swimmer
from brax.envs import walker2d
from brax.envs.base import Env, PipelineEnv, State, Wrapper
from brax.envs.difficulty import apply_difficulty, get_supported_levels, get_task_for_env, register_env_task_mapping, \
    register_task_difficulty, supports_difficulty
from brax.envs.safe_ant import SafeLiftAnt
from brax.envs.safe_button import SafeButton, SafeButtonPoint
from brax.envs.safe_circle import SafeCircle, SafeCirclePoint
from brax.envs.safe_goal import SafeGoal, SafeGoalPoint
from brax.envs.safe_height import SafeHeight, SafeHeightHumanoid
from brax.envs.safe_lift import SafeLift, SafeLiftHumanoid
from brax.envs.safe_push import SafePush, SafePushPoint
from brax.envs.safe_reacher import SafeReacher
from brax.envs.safe_spider import SafeLiftSpider
from brax.envs.safe_velocity import (
    SafeVelocity,
    SafeVelocityAnt,
    SafeVelocityHumanoid,
    SafeVelocityHalfcheetah,
    SafeVelocityHopper,
    SafeVelocitySwimmer,
    SafeVelocityWalker2d,
)
from brax.envs.safe_pathway import SafePathway, SafePathwayWalker2D
from brax.envs.wrappers import training

_envs = {
    # Original Brax environments
    'ant': ant.Ant,
    'fast': fast.Fast,
    'halfcheetah': half_cheetah.Halfcheetah,
    'hopper': hopper.Hopper,
    'humanoid': humanoid.Humanoid,
    'humanoidstandup': humanoidstandup.HumanoidStandup,
    'inverted_pendulum': inverted_pendulum.InvertedPendulum,
    'inverted_double_pendulum': inverted_double_pendulum.InvertedDoublePendulum,
    'pusher': pusher.Pusher,
    'reacher': reacher.Reacher,
    'swimmer': swimmer.Swimmer,
    'walker2d': walker2d.Walker2d,
    # Safety environment suites. Naming convention: safe_[task]_[agent]
    'safe_button_point': SafeButtonPoint,
    'safe_circle_point': SafeCirclePoint,
    'safe_goal_point': SafeGoalPoint,
    'safe_height_humanoid': SafeHeightHumanoid,
    'safe_lift_ant': SafeLiftAnt,
    'safe_lift_humanoid': SafeLiftHumanoid,
    'safe_lift_spider': SafeLiftSpider,
    'safe_pathway_walker2d': SafePathwayWalker2D,
    'safe_push_point': SafePushPoint,
    # Velocity-constrained environments
    'safe_velocity': SafeVelocity,  # Factory (backward compat)
    'safe_velocity_ant': SafeVelocityAnt,
    'safe_velocity_humanoid': SafeVelocityHumanoid,
    'safe_velocity_halfcheetah': SafeVelocityHalfcheetah,
    'safe_velocity_hopper': SafeVelocityHopper,
    'safe_velocity_swimmer': SafeVelocitySwimmer,
    'safe_velocity_walker2d': SafeVelocityWalker2d,
    # Stand-alone tasks
    'safe_reacher': safe_reacher.SafeReacher,
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


def get_environment(
    env_name: str,
    level: Optional[int] = None,
    vision: bool = False,
    vision_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Env:
    """Returns an environment from the environment registry.

    This is the primary function for creating environments. It handles:
    - Looking up the environment class from the registry
    - Applying difficulty overrides if a level is specified
    - Wrapping with UnifiedEnvAdapter for consistent cost/info fields
    - Optionally wrapping with PixelObservationWrapper for vision-based training

    Args:
      env_name: environment name string
      level: optional difficulty level (1, 2, 3). If provided, applies difficulty
             overrides for supported environments.
      vision: if True, wraps the environment with PixelObservationWrapper to
             provide pixel observations from MuJoCo cameras.
      vision_kwargs: keyword arguments passed to PixelObservationWrapper.
             Supported keys: cameras, height, width, obs_mode, frame_stack,
             render_every_n, grayscale, num_render_workers.
      **kwargs: keyword arguments that get passed to the Env class constructor

    Returns:
      env: an environment wrapped with UnifiedEnvAdapter (and optionally
           PixelObservationWrapper for vision mode)
    """
    if env_name not in _envs:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(_envs.keys())}")

    # Apply difficulty overrides if level is specified
    if level is not None:
        kwargs = apply_difficulty(env_name, kwargs, level)

    env_cls = _envs[env_name]
    base_env = env_cls(**kwargs)
    # Wrap with unified adapter so downstream code can rely on cost/info fields
    env = UnifiedEnvAdapter(base_env, **kwargs)

    if vision:
        from brax.envs.wrappers.pixel_observation import PixelObservationWrapper
        env = PixelObservationWrapper(env, **(vision_kwargs or {}))

    return env


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
        vision: bool = False,
        vision_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
) -> Env:
    """Creates an environment with training wrappers.

    This is a convenience function that calls get_environment() and then
    optionally wraps the result with training wrappers (EpisodeWrapper,
    VmapWrapper, AutoResetWrapper).

    For basic environment creation without training wrappers, use get_environment().

    Args:
      env_name: environment name string
      episode_length: length of episode (if None, uses env's default if available)
      action_repeat: how many repeated actions to take per environment step
      auto_reset: whether to auto reset the environment after an episode is done
      batch_size: the number of environments to batch together
      level: optional difficulty level (1, 2, 3). If provided, applies difficulty
             overrides for supported environments.
      vision: if True, wraps with PixelObservationWrapper for pixel observations.
      vision_kwargs: keyword arguments for PixelObservationWrapper.
      **kwargs: keyword arguments that get passed to the Env class constructor

    Returns:
      env: an environment with training wrappers applied
    """
    # Get the base environment using get_environment
    env = get_environment(env_name, level=level, vision=vision,
                          vision_kwargs=vision_kwargs, **kwargs)

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
