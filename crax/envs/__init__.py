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

from crax.envs import ant
from crax.envs import fast
from crax.envs import half_cheetah
from crax.envs import hopper
from crax.envs import humanoid
from crax.envs import humanoidstandup
from crax.envs import inverted_double_pendulum
from crax.envs import inverted_pendulum
from crax.envs import pusher
from crax.envs import reacher
from crax.envs import safe_ant
from crax.envs import safe_height
from crax.envs import safe_lift
from crax.envs import safe_pathway
from crax.envs import safe_reacher
from crax.envs import safe_spider
from crax.envs import safe_velocity
from crax.envs import swimmer
from crax.envs import walker2d
from crax.envs.base import Env, PipelineEnv, State, Wrapper
from crax.envs.difficulty import apply_difficulty, get_supported_levels, get_task_for_env, register_env_task_mapping, \
    register_task_difficulty, supports_difficulty
from crax.envs.safe_ant import SafeLiftAnt
from crax.envs.safe_button import SafeButton, SafeButtonPoint
from crax.envs.safe_circle import SafeCircle, SafeCirclePoint
from crax.envs.safe_goal import SafeGoal, SafeGoalPoint
from crax.envs.safe_height import SafeHeight, SafeHeightHumanoid
from crax.envs.safe_lift import SafeLift, SafeLiftHumanoid
from crax.envs.safe_pathway import SafePathway, SafePathwayWalker2D
from crax.envs.safe_push import SafePush, SafePushPoint
from crax.envs.safe_reacher import SafeReacher
from crax.envs.safe_spider import SafeLiftSpider
from crax.envs.safe_velocity import (
    SafeVelocity,
    SafeVelocityAnt,
    SafeVelocityHumanoid,
    SafeVelocityHalfcheetah,
    SafeVelocityHopper,
    SafeVelocitySwimmer,
    SafeVelocityWalker2d,
)
from crax.envs.wrappers import training

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

    Args:
      env_name: environment name string.
      level: optional difficulty level (1, 2, 3).
      vision: wrap with GpuPixelObservationWrapper for GPU-rendered
             egocentric pixel observations via MJWarp. Requires
             backend='mjx' (see **kwargs) and 'num_envs' in vision_kwargs
             (MJWarp's render context needs a static batch size). The
             returned env is vmapped internally to satisfy that. Callers
             get back an already-batched env ready for .reset(rng)/.step().
      vision_kwargs: kwargs forwarded to GpuPixelObservationWrapper.
             Required: num_envs. Supported: camera, height, width, obs_mode,
             frame_stack, use_shadows.
      **kwargs: kwargs passed to the Env class constructor.

    Returns:
      env: environment wrapped with UnifiedEnvAdapter (+ VmapWrapper and
           GpuPixelObservationWrapper when vision=True).
    """
    if env_name not in _envs:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(_envs.keys())}")

    if level is not None:
        kwargs = apply_difficulty(env_name, kwargs, level)

    env_cls = _envs[env_name]
    base_env = env_cls(**kwargs)
    env = UnifiedEnvAdapter(base_env, **kwargs)

    if vision:
        from crax.envs.wrappers.pixel_observation_gpu import GpuPixelObservationWrapper
        from crax.envs.wrappers.training import VmapWrapper
        vision_kwargs = dict(vision_kwargs or {})
        num_envs = vision_kwargs.pop('num_envs', None)
        if num_envs is None:
            raise ValueError(
                "vision_kwargs must include 'num_envs': MJWarp's render "
                "context needs a static batch size, so the env is vmapped "
                "internally here rather than by an outer VmapWrapper."
            )
        env = VmapWrapper(env, batch_size=num_envs)
        env = GpuPixelObservationWrapper(env, num_envs=num_envs, **vision_kwargs)

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
      vision: use MJWarp for pixel observations. Applied after Episode/Vmap/AutoReset
      vision_kwargs: keyword arguments for GpuPixelObservationWrapper
      **kwargs: keyword arguments that get passed to the Env class constructor

    Returns:
      env: an environment with training wrappers applied
    """
    # Get the base environment
    env = get_environment(env_name, level=level, vision=False, **kwargs)

    # Resolve episode length: prefer explicit, else environment default if available
    if episode_length is None:
        episode_length = getattr(env, 'episode_length', None)
    if episode_length is not None:
        env = training.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = training.VmapWrapper(env, batch_size)
    if auto_reset:
        env = training.AutoResetWrapper(env)

    if vision:
        from crax.envs.wrappers.pixel_observation_gpu import GpuPixelObservationWrapper
        vision_kwargs = dict(vision_kwargs or {})
        num_envs = vision_kwargs.pop('num_envs', batch_size)
        if num_envs is None:
            raise ValueError(
                "vision=True requires either batch_size or "
                "vision_kwargs['num_envs'] (MJWarp needs a static batch size)."
            )
        env = GpuPixelObservationWrapper(env, num_envs=num_envs, **vision_kwargs)

    return env
