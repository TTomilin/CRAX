"""Walker2d environment with a velocity safety constraint."""

import jax
from jax import numpy as jp

from brax.envs.base import State
from brax.envs.velocity_constraints import (
    add_velocity_cost_metrics,
    binary_velocity_cost,
    build_velocity_info,
    compute_velocity,
)
from brax.envs.walker2d import Walker2d


class Walker2dVelocityConstrained(Walker2d):
    """Walker2d locomotion with a binary velocity constraint cost."""

    def __init__(
            self,
            velocity_threshold: float = 2.3415,
            velocity_cost_weight: float = 1.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self._velocity_threshold = float(velocity_threshold)
        self._velocity_cost_weight = float(velocity_cost_weight)

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        next_state = super().step(state, action)
        pipeline_state = next_state.pipeline_state

        x_velocity = compute_velocity(
            pipeline_state0.x.pos[0, 0], pipeline_state.x.pos[0, 0], self.dt
        )
        cost, violation = binary_velocity_cost(
            x_velocity, self._velocity_threshold, self._velocity_cost_weight
        )

        metrics = dict(next_state.metrics)
        add_velocity_cost_metrics(
            metrics,
            next_state.reward,
            velocity_value=x_velocity,
            threshold=self._velocity_threshold,
            violation=violation,
            cost=cost,
        )

        current_info = next_state.info if isinstance(next_state.info, dict) else {}
        step_count = current_info.get("step_count", 0) + 1
        info = build_velocity_info(
            current_info,
            cost=cost,
            velocity_value=x_velocity,
            threshold=self._velocity_threshold,
            violation=violation,
            step_count=step_count,
        )

        return next_state.replace(metrics=metrics, info=info)

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
