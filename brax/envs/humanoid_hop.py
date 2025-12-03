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

"""Humanoid one-legged hopping locomotion environment with contact constraints."""

import jax
from jax import numpy as jp
from brax.envs.humanoid import Humanoid
from brax.envs.base import State


class HumanoidHop(Humanoid):
  """Humanoid locomotion environment with one-legged hopping constraint.
  
  This environment extends the standard Humanoid environment by adding a
  contact-based safety constraint: the humanoid must hop on ONE designated leg,
  keeping the opposite leg off the ground. This creates a challenging locomotion
  task that tests contact pattern constraints and dynamic balance.
  
  The cost (constraint violation) is continuous: the non-hopping foot incurs
  cost proportional to contact force above a threshold. This encourages safe RL
  algorithms to learn to minimize unwanted contacts.
  """

  def __init__(
      self,
      hopping_leg: str = 'left',
      contact_threshold: float = 0.5,  # Newtons, threshold for foot contact
    cost_weight: float = 1.0,
    contact_scale: float | None = None,
    enable_contact_metrics: bool = True,
    **kwargs,
  ):
    """Initialize the one-legged hopping humanoid environment.
    
    Args:
      hopping_leg: Which leg to hop on ('left' or 'right').
      contact_threshold: Minimum contact force (Newtons) to count as violation.
      cost_weight: Weight for contact constraint violation cost.
      **kwargs: Additional arguments passed to parent HumanoidStandup class.
    """
    debug = kwargs.pop('debug', enable_contact_metrics)
    super().__init__(debug=debug, **kwargs)
    
    assert hopping_leg in ['left', 'right'], f"hopping_leg must be 'left' or 'right', got {hopping_leg}"
    self._hopping_leg = hopping_leg
    self._contact_threshold = contact_threshold
    self._cost_weight = cost_weight
    self._contact_scale = contact_scale
    self._debug = debug
    
    # Body indices for feet (initialize after sys is loaded)
    # In humanoidstandup.xml:
    # Body 0: torso (root)
    # We need to find indices for left_foot and right_foot bodies
    self._left_foot_body_idx = None
    self._right_foot_body_idx = None
    
    # Find foot indices now that sys is initialized
    self._find_foot_indices()

  def reset(self, rng: jax.Array) -> State:
    """Reset the environment with contact constraint metrics.
    
    Inherits from Humanoid which starts in an upright standing pose,
    perfect for learning hopping constraints.
    """
    # Use parent Humanoid reset (starts upright at 1.4m height)
    state = super().reset(rng)
    
    # Initialize contact-related metrics with consistent shapes
    reward_shape = jp.shape(state.reward)
    zero = jp.zeros(reward_shape)
    
    # Create updated metrics dict with contact-specific metrics
    updated_metrics = state.metrics.copy()
    if self._debug:
        updated_metrics.update({
            'left_foot_contact_force': zero,
            'right_foot_contact_force': zero,
            'contact_violation': zero,
            'contact_cost': zero,
            'cost': zero,  # Initialize cost for PPO Lagrange
        })
    
    # Initialize info dictionary with cost (required for PPO Lagrange v2)
    # Note: Don't store strings (like hopping_leg) as they're not JAX types
    info = {
        "cost": zero,
        "left_foot_contact_force": zero,
        "right_foot_contact_force": zero,
        "contact_violation": zero,
        "step_count": 0,
    }
    
    return state.replace(metrics=updated_metrics, info=info)

  def step(self, state: State, action: jax.Array) -> State:
    """Run one timestep of the environment's dynamics with contact constraints."""
    # Scale action from [-1,1] to actuator limits (like Humanoid parent)
    action_min = self.sys.actuator.ctrl_range[:, 0]
    action_max = self.sys.actuator.ctrl_range[:, 1]
    action = (action + 1) * (action_max - action_min) * 0.5 + action_min
    
    # Get standard humanoid step behavior (forward motion, health, etc.)
    pipeline_state0 = state.pipeline_state
    assert pipeline_state0 is not None
    pipeline_state = self.pipeline_step(pipeline_state0, action)

    # Calculate velocity using COM (matching Humanoid parent class exactly)
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
    
    # Extract contact forces for feet
    left_foot_force, right_foot_force = self._get_foot_contact_forces(pipeline_state)
    
    # Compute contact violation cost (this is our safety constraint)
    contact_cost, violation = self._compute_contact_cost(left_foot_force, right_foot_force)
    
    obs = self._get_obs(pipeline_state, action)
    
    # Reward: forward movement + staying alive - control cost
    # Note: contact_cost is NOT subtracted from reward, it's handled by PPO-Lagrange
    reward = forward_reward + healthy_reward - ctrl_cost
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    
    # Ensure all metrics have consistent shapes
    reward_shape = jp.shape(reward)
    
    # Update metrics with contact-related information
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
    )
    if self._debug:
        state.metrics.update(
            left_foot_contact_force=jp.broadcast_to(left_foot_force, reward_shape),
            right_foot_contact_force=jp.broadcast_to(right_foot_force, reward_shape),
            contact_violation=jp.broadcast_to(violation, reward_shape),
            contact_cost=jp.broadcast_to(contact_cost, reward_shape),
            cost=jp.broadcast_to(contact_cost, reward_shape),
        )
    
    # Update info dictionary with cost (required for PPO Lagrange v2)
    current_info = getattr(state, 'info', {})
    step_count = current_info.get('step_count', 0) + 1
    
    # Update info dictionary (copy existing and update)
    # Note: Don't store strings (like hopping_leg) as they're not JAX types
    new_info = current_info.copy() if isinstance(current_info, dict) else {}
    # ALWAYS update cost (required for PPO-Lagrange training)
    new_info["cost"] = contact_cost
    new_info["step_count"] = step_count
    
    # Add debug metrics if enabled
    if self._debug:
        new_info.update({
            "left_foot_contact_force": left_foot_force,
            "right_foot_contact_force": right_foot_force,
            "contact_violation": violation,
        })
    
    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, info=new_info
    )

  def _find_foot_indices(self):
    """Find body indices for shin bodies (which contain foot geoms).
    
    Note: In the humanoidstandup model, the foot geoms are attached to the 
    shin bodies, not separate foot bodies. So we track shin contacts for
    foot contact detection.
    """
    # Use link_names from Brax System
    # The feet geoms are part of the shin bodies in humanoidstandup.xml
    if hasattr(self.sys, 'link_names'):
      link_names = self.sys.link_names
      try:
        # In Brax, indices start from 0 for first body
        # We need shin indices since foot geoms are attached to shins
        self._left_foot_body_idx = link_names.index('left_shin')
        self._right_foot_body_idx = link_names.index('right_shin')
        return
      except ValueError:
        pass
    
    # Fallback: try MuJoCo model directly
    mj_model = self.sys.mj_model
    if mj_model is not None:
      try:
        self._left_foot_body_idx = mj_model.body('left_shin').id
        self._right_foot_body_idx = mj_model.body('right_shin').id
        return
      except Exception as e:
        print(f"Warning: Could not find shin bodies: {e}")
    
    # If all methods fail, raise error
    raise RuntimeError(
        f"Could not find 'left_shin' and 'right_shin' bodies in model. "
        f"Available link_names: {getattr(self.sys, 'link_names', 'N/A')}"
    )

  def _get_foot_contact_forces(self, pipeline_state) -> tuple[jax.Array, jax.Array]:
    """Extract contact forces for left and right feet.
    
    Args:
      pipeline_state: Current physics pipeline state (mjx.Data).
    
    Returns:
      Tuple of (left_foot_force_magnitude, right_foot_force_magnitude).
      Returns a binary indicator: 1.0 if foot is in contact, 0.0 otherwise.
    """
    # Find foot indices if not yet cached
    if self._left_foot_body_idx is None:
      self._find_foot_indices()
    
    # Use contact field to detect ground contact
    # In generalized pipeline, contact info is only populated when debug=True
    if hasattr(pipeline_state, 'contact') and pipeline_state.contact is not None:
      contact = pipeline_state.contact
      
      # contact.link_idx is a tuple of (link_idx_a, link_idx_b)
      # Ground contacts have link_idx_a = -1 (or similar sentinel value)
      # We check if our foot body indices appear in the contact pairs
      # Note: foot can be in either position (a or b) of the contact pair
      if hasattr(contact, 'link_idx'):
        link_idx_a, link_idx_b = contact.link_idx
        if hasattr(contact, 'dist'):
            # dist < 0 indicates actual penetration/contact; positive = separation
            active_mask = contact.dist < 0.0
        else:
            # If distance not available, assume entries correspond to active contacts
            active_mask = jp.ones_like(link_idx_a, dtype=bool)
        active_mask = jp.asarray(active_mask)
        left_mask = active_mask & ((link_idx_a == self._left_foot_body_idx) | (link_idx_b == self._left_foot_body_idx))
        right_mask = active_mask & ((link_idx_a == self._right_foot_body_idx) | (link_idx_b == self._right_foot_body_idx))
        left_foot_in_contact = jp.where(jp.any(left_mask), 1.0, 0.0)
        right_foot_in_contact = jp.where(jp.any(right_mask), 1.0, 0.0)
        return left_foot_in_contact.astype(jp.float32), right_foot_in_contact.astype(jp.float32)
    
    # Fallback if no contact information available
    # This should rarely happen in MJX
    return jp.array(0.0), jp.array(0.0)

  def _compute_contact_cost(
      self, 
      left_foot_force: jax.Array,
      right_foot_force: jax.Array
  ) -> tuple[jax.Array, jax.Array]:
    """Compute cost based on which foot should not touch the ground.
    
    Args:
      left_foot_force: Contact force magnitude on left foot.
      right_foot_force: Contact force magnitude on right foot.
    
    Returns:
      Tuple of (cost, violation).
    """
    # Determine which foot should NOT touch the ground
    if self._hopping_leg == 'left':
      # Hopping on left, so right foot should NOT touch
      violation_force = right_foot_force
    else:
      # Hopping on right, so left foot should NOT touch
      violation_force = left_foot_force

    # Continuous violation: optionally normalize force above threshold.
    force_over_threshold = jp.maximum(0.0, violation_force - self._contact_threshold)
    scale = self._contact_scale
    if scale is None:
      normalized = force_over_threshold
    else:
      normalized = force_over_threshold / jp.maximum(1e-6, scale)
    violation = jp.clip(normalized, 0.0, 1.0)

    # Cost is weighted continuous violation
    cost = self._cost_weight * violation

    return cost, violation

