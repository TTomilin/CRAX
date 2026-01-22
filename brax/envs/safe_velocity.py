"""Unified velocity-constrained environment for multiple agents."""

from __future__ import annotations

from typing import Callable, Type

import jax
from jax import numpy as jp

from brax.envs.ant import Ant
from brax.envs.base import PipelineEnv, State
from brax.envs.half_cheetah import Halfcheetah
from brax.envs.hopper import Hopper
from brax.envs.humanoid import Humanoid
from brax.envs.swimmer import Swimmer
from brax.envs.velocity_constraints import (
    add_velocity_cost_metrics,
    binary_velocity_cost,
    build_velocity_info,
    compute_velocity,
    hinge_velocity_cost,
    planar_speed,
)
from brax.envs.walker2d import Walker2d

# Default velocity thresholds per agent (used as level 1 baseline)
DEFAULT_THRESHOLDS = {
    "ant": 2.6222,
    "halfcheetah": 3.2096,
    "hopper": 0.7402,
    "humanoid": 1.4149,
    "swimmer": 0.2282,
    "walker2d": 2.3415,
}

# Difficulty level multipliers (lower = harder, must move slower)
_LEVEL_MULTIPLIERS = {
    1: 1.0,    # Baseline - easiest
    2: 0.75,   # 75% of baseline - medium
    3: 0.5,    # 50% of baseline - hardest
}

# Pre-computed thresholds per (level, agent) for convenience
LEVEL_THRESHOLDS: dict[int, dict[str, float]] = {
    level: {agent: thresh * mult for agent, thresh in DEFAULT_THRESHOLDS.items()}
    for level, mult in _LEVEL_MULTIPLIERS.items()
}


def get_threshold_for_level(agent: str, level: int | None = None) -> float:
    """Get the velocity threshold for a given agent and difficulty level.

    Args:
        agent: Agent name (ant, halfcheetah, hopper, humanoid, swimmer, walker2d).
        level: Difficulty level (1, 2, or 3). If None, uses level 1 (baseline).

    Returns:
        The velocity threshold for the specified agent and level.
    """
    agent = agent.lower()
    if agent not in DEFAULT_THRESHOLDS:
        raise ValueError(f"Unknown agent: {agent}")

    if level is None:
        level = 1

    if level not in LEVEL_THRESHOLDS:
        raise ValueError(f"Unknown level: {level}. Supported levels: {list(LEVEL_THRESHOLDS.keys())}")

    return LEVEL_THRESHOLDS[level][agent]

# Base environment classes
_BASE_ENVS: dict[str, Type[PipelineEnv]] = {
    "ant": Ant,
    "halfcheetah": Halfcheetah,
    "hopper": Hopper,
    "humanoid": Humanoid,
    "swimmer": Swimmer,
    "walker2d": Walker2d,
}

# Velocity computation modes
VELOCITY_MODE_X = "x"  # x-axis velocity only
VELOCITY_MODE_PLANAR = "planar"  # 2D planar speed
VELOCITY_MODE_COM_PLANAR = "com_planar"  # Planar speed from center of mass
VELOCITY_MODE_Q = "q"  # Velocity from generalized coordinates

_VELOCITY_MODES = {
    "ant": VELOCITY_MODE_PLANAR,
    "halfcheetah": VELOCITY_MODE_X,
    "hopper": VELOCITY_MODE_X,
    "humanoid": VELOCITY_MODE_COM_PLANAR,
    "swimmer": VELOCITY_MODE_Q,
    "walker2d": VELOCITY_MODE_X,
}


def _compute_x_velocity(env, pipeline_state0, pipeline_state) -> jax.Array:
    """Compute x-axis velocity from body position."""
    return compute_velocity(
        pipeline_state0.x.pos[0, 0], pipeline_state.x.pos[0, 0], env.dt
    )


def _compute_planar_velocity(env, pipeline_state0, pipeline_state) -> jax.Array:
    """Compute planar speed from body position."""
    velocity = compute_velocity(
        pipeline_state0.x.pos[0], pipeline_state.x.pos[0], env.dt
    )
    return planar_speed(velocity[..., :2])


def _compute_com_planar_velocity(env, pipeline_state0, pipeline_state) -> jax.Array:
    """Compute planar speed from center of mass (for humanoid)."""
    com_before, *_ = env._com(pipeline_state0)
    com_after, *_ = env._com(pipeline_state)
    velocity = compute_velocity(com_before, com_after, env.dt)
    return planar_speed(velocity[..., :2])


def _compute_q_velocity(env, pipeline_state0, pipeline_state) -> jax.Array:
    """Compute x-velocity from generalized coordinates (for swimmer)."""
    return compute_velocity(pipeline_state0.q[0], pipeline_state.q[0], env.dt)


_VELOCITY_COMPUTERS: dict[str, Callable] = {
    VELOCITY_MODE_X: _compute_x_velocity,
    VELOCITY_MODE_PLANAR: _compute_planar_velocity,
    VELOCITY_MODE_COM_PLANAR: _compute_com_planar_velocity,
    VELOCITY_MODE_Q: _compute_q_velocity,
}


def create_safe_velocity_env(
    agent: str,
    level: int | None = None,
    velocity_threshold: float | None = None,
    velocity_cost_weight: float = 1.0,
    cost_mode: str = "binary",
    reward_scaler: float = 0.01,
    **kwargs,
) -> PipelineEnv:
    """Factory function to create a velocity-constrained environment.

    Args:
        agent: Name of the agent ('ant', 'halfcheetah', 'hopper', 'humanoid', 'swimmer', 'walker2d').
        level: Difficulty level (1, 2, or 3). Level 1 is easiest, level 3 is hardest.
               If provided, overrides velocity_threshold with the level-specific value.
        velocity_threshold: Maximum allowed velocity. If None, uses the threshold for
                           the specified level (or level 1 if level is also None).
        velocity_cost_weight: Weight for velocity constraint violation cost.
        cost_mode: Cost computation mode ('binary' or 'hinge').
        reward_scaler: Coefficient for the reward.
        **kwargs: Additional arguments passed to the base environment.

    Returns:
        A velocity-constrained environment instance.
    """
    agent = agent.lower()
    if agent not in _BASE_ENVS:
        raise ValueError(f"Unknown agent: {agent}. Supported agents: {list(_BASE_ENVS.keys())}")

    base_cls = _BASE_ENVS[agent]

    # Determine threshold: explicit > level-based > default (level 1)
    if velocity_threshold is not None:
        threshold = velocity_threshold
    else:
        threshold = get_threshold_for_level(agent, level)
    velocity_mode = _VELOCITY_MODES[agent]
    compute_vel = _VELOCITY_COMPUTERS[velocity_mode]

    class SafeVelocityEnv(base_cls):
        """Velocity-constrained locomotion environment.

        Extends the base locomotion environment by adding velocity constraints
        as safety constraints for constrained RL training.
        """

        def __init__(self, **init_kwargs):
            super().__init__(**init_kwargs)
            self._velocity_threshold = float(threshold)
            self._velocity_cost_weight = float(velocity_cost_weight)
            self._cost_mode = cost_mode
            self._velocity_mode = velocity_mode
            self._agent_name = agent
            self._reward_scaler = reward_scaler

        def step(self, state: State, action: jax.Array) -> State:
            pipeline_state0 = state.pipeline_state
            assert pipeline_state0 is not None
            next_state = super().step(state, action)
            pipeline_state = next_state.pipeline_state

            velocity_value = compute_vel(self, pipeline_state0, pipeline_state)

            if self._cost_mode == "binary":
                cost, violation = binary_velocity_cost(
                    velocity_value, self._velocity_threshold, self._velocity_cost_weight
                )
            elif self._cost_mode == "hinge":
                cost, violation = hinge_velocity_cost(
                    velocity_value, self._velocity_threshold, self._velocity_cost_weight
                )
            else:
                raise ValueError(f"Unknown cost_mode: {self._cost_mode}")

            metrics = dict(next_state.metrics)
            add_velocity_cost_metrics(
                metrics,
                next_state.reward,
                velocity_value=velocity_value,
                threshold=self._velocity_threshold,
                violation=violation,
                cost=cost,
            )

            current_info = next_state.info if isinstance(next_state.info, dict) else {}
            step_count = current_info.get("step_count", 0) + 1
            info = build_velocity_info(
                current_info,
                cost=cost,
                velocity_value=velocity_value,
                threshold=self._velocity_threshold,
                violation=violation,
                step_count=step_count,
            )

            return next_state.replace(reward=next_state.reward * self._reward_scaler, metrics=metrics, info=info)

        def reset(self, rng: jax.Array) -> State:
            state = super().reset(rng)
            zero = jp.zeros_like(state.reward)
            metrics = dict(state.metrics)
            add_velocity_cost_metrics(
                metrics,
                state.reward,
                velocity_value=zero,
                threshold=self._velocity_threshold,
                violation=zero,
                cost=zero,
            )
            info = build_velocity_info(
                state.info,
                cost=zero,
                velocity_value=zero,
                threshold=self._velocity_threshold,
                violation=zero,
                step_count=0,
            )
            return state.replace(metrics=metrics, info=info)

    return SafeVelocityEnv(**kwargs)


class SafeVelocity:
    """Wrapper class that provides a consistent interface for safe velocity environments.

    This class acts as a factory that creates the appropriate velocity-constrained
    environment based on the specified agent and difficulty level.

    Difficulty Levels:
        - Level 1: Baseline thresholds (easiest)
        - Level 2: 75% of baseline (medium)
        - Level 3: 50% of baseline (hardest)

    Usage:
        env = SafeVelocity(agent="ant", level=1)
        # or via brax.envs.create:
        env = brax.envs.create("safe_velocity", agent="ant", level=2)
    """

    def __new__(
        cls,
        agent: str = "ant",
        level: int | None = None,
        velocity_threshold: float | None = None,
        velocity_cost_weight: float = 1.0,
        cost_mode: str = "binary",
        reward_scaler: float = 0.01,
        **kwargs,
    ):
        """Create a new velocity-constrained environment.

        Args:
            agent: Name of the agent ('ant', 'halfcheetah', 'hopper', 'humanoid',
                   'swimmer', 'walker2d').
            level: Difficulty level (1, 2, or 3). Higher levels have stricter
                   velocity constraints. Defaults to 1 if not specified.
            velocity_threshold: Maximum allowed velocity. If specified, overrides
                               the level-based threshold.
            velocity_cost_weight: Weight for velocity constraint violation cost.
            cost_mode: Cost computation mode ('binary' or 'hinge').
            reward_scaler: Coefficient for the reward.
            **kwargs: Additional arguments passed to the base environment.

        Returns:
            A velocity-constrained environment instance.
        """
        return create_safe_velocity_env(
            agent=agent,
            level=level,
            velocity_threshold=velocity_threshold,
            velocity_cost_weight=velocity_cost_weight,
            cost_mode=cost_mode,
            reward_scaler=reward_scaler,
            **kwargs,
        )
