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

"""Velocity-constrained ant locomotion environment."""

import jax
from jax import numpy as jp

from brax.envs.ant import Ant
from brax.envs.base import State
from brax.envs.velocity_constraints import (
    add_velocity_cost_metrics,
    binary_velocity_cost,
    build_velocity_info,
    compute_velocity,
    hinge_velocity_cost,
    planar_speed,
)


class AntVelocityConstrained(Ant):
  """Ant locomotion environment with velocity constraints.
  
  This environment extends the standard Ant environment by adding velocity
  constraints that serve as safety constraints for constrained RL training.
  The constraint violation (cost) is computed based on the ant's velocity
  exceeding a specified threshold.
  """

  def __init__(
      self,
      max_velocity: float = 0.5,  # Set below ant's natural max velocity (~0.81) to ensure violations
      velocity_cost_weight: float = 1.0,
      cost_mode: str = "hinge",
      velocity_threshold: float | None = None,
      **kwargs,
  ):
    """Initialize the velocity-constrained ant environment.
    
    Args:
      max_velocity: Maximum allowed velocity magnitude (m/s).
      velocity_cost_weight: Weight for velocity constraint violation cost.
      **kwargs: Additional arguments passed to parent Ant class.
    """
    super().__init__(**kwargs)
    self._cost_mode = cost_mode
    self._max_velocity = float(max_velocity)
    self._velocity_cost_weight = float(velocity_cost_weight)
    self._velocity_threshold = 2.6222 if velocity_threshold is None else float(velocity_threshold)

  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics with velocity constraints."""
    pipeline_state0 = state.pipeline_state
    assert pipeline_state0 is not None
    pipeline_state = self.pipeline_step(pipeline_state0, action)

    # Calculate velocity components
    velocity = compute_velocity(pipeline_state0.x.pos[0], pipeline_state.x.pos[0], self.dt)
    velocity_value = planar_speed(velocity[..., :2])
    forward_reward = velocity[..., 0]
    
    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
    is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
    
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy
      
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
    contact_cost = 0.0
    
    if self._cost_mode == "binary":
        velocity_cost, velocity_violation = binary_velocity_cost(
            velocity_value, self._velocity_threshold, self._velocity_cost_weight
        )
        threshold_value = self._velocity_threshold
    elif self._cost_mode == "hinge":
        velocity_cost, velocity_violation = hinge_velocity_cost(
            velocity_value, self._max_velocity, self._velocity_cost_weight
        )
        threshold_value = self._max_velocity
    else:
        raise ValueError(f"Unknown cost_mode: {self._cost_mode}")

    obs = self._get_obs(pipeline_state)
    reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    
    metrics = dict(state.metrics)
    metrics.update(
        reward_forward=forward_reward,
        reward_survive=healthy_reward,
        reward_ctrl=-ctrl_cost,
        reward_contact=-contact_cost,
        x_position=pipeline_state.x.pos[0, 0],
        y_position=pipeline_state.x.pos[0, 1],
        distance_from_origin=jp.linalg.norm(pipeline_state.x.pos[0]),
        x_velocity=velocity[..., 0],
        y_velocity=velocity[..., 1],
        forward_reward=forward_reward,
    )

    add_velocity_cost_metrics(
        metrics,
        reward,
        velocity_value=velocity_value,
        threshold=threshold_value,
        violation=velocity_violation,
        cost=velocity_cost,
    )

    current_info = state.info if isinstance(state.info, dict) else {}
    step_count = current_info.get("step_count", 0) + 1
    new_info = build_velocity_info(
        current_info,
        cost=velocity_cost,
        velocity_value=velocity_value,
        threshold=threshold_value,
        violation=velocity_violation,
        step_count=step_count,
    )

    return state.replace(
        pipeline_state=pipeline_state,
        obs=obs,
        reward=reward,
        done=done,
        metrics=metrics,
        info=new_info,
    )

  def reset(self, rng: jax.Array) -> State:
    """Reset the environment with velocity constraint metrics."""
    state = super().reset(rng)
    
    zero = jp.zeros_like(state.reward)
    metrics = dict(state.metrics)
    threshold_value = self._max_velocity if self._cost_mode == "hinge" else self._velocity_threshold
    add_velocity_cost_metrics(
        metrics,
        state.reward,
        velocity_value=zero,
        threshold=threshold_value,
        violation=zero,
        cost=zero,
    )
    info = build_velocity_info(
        state.info,
        cost=zero,
        velocity_value=zero,
        threshold=threshold_value,
        violation=zero,
        step_count=0,
    )
    return state.replace(metrics=metrics, info=info)