"""Humanoid hop environment variant with airtime incentive."""

from __future__ import annotations

import jax
from jax import numpy as jp

from brax.envs.humanoid_hop import HumanoidHop
from brax.envs.base import State


class HumanoidHopAirtime(HumanoidHop):
    """Extends ``HumanoidHop`` with an airtime bonus in the reward."""

    def __init__(
        self,
        *,
        airtime_reward_weight: float = 5.0,
        airtime_exponent: float = 1.0,
        enable_contact_metrics: bool = True,
        **kwargs,
    ):
        """Initialise the airtime-augmented humanoid hop environment.

        Args:
            airtime_reward_weight: Scalar multiplier for airtime bonus.
            airtime_exponent: Controls shaping; >1 emphasises near-perfect airtime.
            enable_contact_metrics: Forwarded so contact info is available by default.
            **kwargs: Remaining arguments passed to ``HumanoidHop``.
        """

        super().__init__(
            enable_contact_metrics=enable_contact_metrics,
            **kwargs,
        )
        self._airtime_reward_weight = airtime_reward_weight
        self._airtime_exponent = airtime_exponent

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)
        reward_shape = jp.shape(state.reward)
        zero = jp.zeros(reward_shape)

        metrics = state.metrics.copy()
        metrics.setdefault("airtime_reward", zero)

        info = dict(getattr(state, "info", {}))
        info.setdefault("airtime_reward", zero)

        return state.replace(metrics=metrics, info=info)

    def step(self, state: State, action: jax.Array) -> State:
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        com_before, *_ = self._com(pipeline_state0)
        com_after, *_ = self._com(pipeline_state)
        velocity = (com_after - com_before) / self.dt
        forward_reward = self._forward_reward_weight * velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        left_foot_force, right_foot_force = self._get_foot_contact_forces(
            pipeline_state
        )
        contact_cost, violation = self._compute_contact_cost(
            left_foot_force, right_foot_force
        )

        any_contact = jp.maximum(left_foot_force, right_foot_force)
        airborne = jp.clip(1.0 - any_contact, 0.0, 1.0)
        airtime_score = jp.power(airborne, self._airtime_exponent)
        airtime_reward = self._airtime_reward_weight * airtime_score

        obs = self._get_obs(pipeline_state, action)

        reward = forward_reward + healthy_reward - ctrl_cost + airtime_reward
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        reward_shape = jp.shape(reward)

        state.metrics.update(
            forward_reward=forward_reward,
            reward_linvel=forward_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            x_position=com_after[0],
            y_position=com_after[1],
            distance_from_origin=jp.linalg.norm(com_after),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            airtime_reward=jp.broadcast_to(airtime_reward, reward_shape),
        )
        if self._debug:
            state.metrics.update(
                left_foot_contact_force=jp.broadcast_to(left_foot_force, reward_shape),
                right_foot_contact_force=jp.broadcast_to(right_foot_force, reward_shape),
                contact_violation=jp.broadcast_to(violation, reward_shape),
                contact_cost=jp.broadcast_to(contact_cost, reward_shape),
                cost=jp.broadcast_to(contact_cost, reward_shape),
            )

        current_info = getattr(state, "info", {})
        step_count = current_info.get("step_count", 0) + 1

        new_info = current_info.copy() if isinstance(current_info, dict) else {}
        new_info["cost"] = contact_cost
        new_info["step_count"] = step_count
        new_info["airtime_reward"] = airtime_reward

        if self._debug:
            new_info.update(
                {
                    "left_foot_contact_force": left_foot_force,
                    "right_foot_contact_force": right_foot_force,
                    "contact_violation": violation,
                }
            )

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            info=new_info,
        )


