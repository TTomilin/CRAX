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

"""Height-constrained humanoid locomotion environment."""

import jax
from jax import numpy as jp
from etils import epath

from brax import actuator
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf


class SafeHeightHumanoid(PipelineEnv):
    """Humanoid locomotion environment with height constraints.

    This environment extends the standard HumanoidStandup environment by adding
    height constraints that serve as safety constraints for constrained RL training.
    The constraint violation (cost) is computed based on the humanoid's height
    exceeding a specified threshold, forcing it to learn to crouch or crawl.

    The environment includes a visual indicator (red transparent strip) showing
    the maximum height boundary (ceiling).

    """

    def __init__(
            self,
            episode_length: int = 1000,
            max_height: float | None = None,
            hinge_margin: float = 0.08,
            height_cost_weight: float = 1.0,
            reward_scaler: float = 0.01,
            backend: str = 'generalized',
            **kwargs,
    ):
        """Initialize the height-constrained humanoid environment.

        Args:
            episode_length: Maximum number of steps per episode.
            level: Difficulty level (1, 2, or 3). If provided and max_height is None,
                   uses the level-specific max_height threshold.
            max_height: Maximum allowed height (z of torso CoM) determined by level. Exceeding this incurs cost.
            hinge_margin: Soft hinge width. Violations scale linearly over this margin.
            height_cost_weight: Weight for height constraint violation cost.
            backend: Physics backend ('generalized', 'spring', 'positional').
            **kwargs: Additional arguments passed to parent PipelineEnv class.
        """
        self.episode_length = episode_length
        self._max_height = float(max_height)
        self._hinge_margin = float(hinge_margin)
        self._height_cost_weight = height_cost_weight
        self._reward_scaler = reward_scaler

        # Load humanoid_height XML and set the ceiling height
        path = epath.resource_path('brax') / 'envs/assets/safe/humanoid_height.xml'
        xml_string = path.read_text()

        # Replace the placeholder with the actual max_height value
        xml_string = xml_string.replace('HEIGHT_PLACEHOLDER', str(self._max_height))

        # Parse the modified XML
        sys = mjcf.loads(xml_string)

        n_frames = 5

        if backend in ['spring', 'positional']:
            sys = sys.tree_replace({'opt.timestep': 0.0015})
            n_frames = 10
            sys = sys.replace(
                actuator=sys.actuator.replace(
                    gear=jp.array([350.0, 350.0, 350.0, 350.0, 350.0, 350.0,
                                   350.0, 350.0, 350.0, 350.0, 350.0, 100.0, 100.0,
                                   100.0, 100.0, 100.0, 100.0])))

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics with height constraints."""
        # Scale action from [-1,1] to actuator limits
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        # Store previous state for velocity calculation
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        # Calculate center-of-mass for forward velocity and height constraint
        com_before, *_ = self._com(pipeline_state0)
        com_after, *_ = self._com(pipeline_state)
        velocity = (com_after - com_before) / self.dt
        current_height = com_after[2]

        # Forward reward: core task is moving forward (like humanoid.py)
        forward_reward = velocity[0]

        # Height bonus: encourage standing upright (reward shaping)
        height_bonus = current_height * 5.0

        # Control cost
        ctrl_cost = 0.01 * jp.sum(jp.square(action))

        # Height constraint cost (safety constraint): require height <= max_height
        # Soft hinge: cost grows linearly when above the threshold within hinge_margin
        height_excess = jp.maximum(0.0, current_height - self._max_height)
        # Normalize by hinge margin to keep cost on [0, ~1] for small violations
        normalized_excess = height_excess / self._hinge_margin
        height_cost = self._height_cost_weight * normalized_excess

        obs = self._get_obs(pipeline_state, action)

        reward = forward_reward + height_bonus - ctrl_cost

        done = 0.0

        # Ensure all metrics have consistent shapes by expanding to match reward shape
        reward_shape = jp.shape(reward)

        # Update metrics with height-related information
        state.metrics.update(
            forward_reward=jp.broadcast_to(forward_reward * self._reward_scaler, reward_shape),
            height_bonus=jp.broadcast_to(height_bonus * self._reward_scaler, reward_shape),
            reward_quadctrl=-ctrl_cost * self._reward_scaler,
            x_position=com_after[0],
            y_position=com_after[1],
            distance_from_origin=jp.linalg.norm(com_after[:2]),
            x_velocity=jp.broadcast_to(velocity[0], reward_shape),
            y_velocity=jp.broadcast_to(velocity[1], reward_shape),
            height=jp.broadcast_to(current_height, reward_shape),
            height_violation=jp.broadcast_to(height_excess, reward_shape),
            height_cost=jp.broadcast_to(height_cost, reward_shape),
            cost=jp.broadcast_to(height_cost, reward_shape),
        )

        # Update info dictionary with cost (required for PPO Lagrange v2)
        current_info = getattr(state, 'info', {})
        step_count = current_info.get('step_count', 0) + 1

        new_info = current_info.copy() if isinstance(current_info, dict) else {}
        new_info.update({
            "cost": height_cost,
            "height": current_height,
            "height_violation": height_excess,
            "step_count": step_count,
        })

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, info=new_info
        )

    def reset(self, rng: jax.Array) -> State:
        """Reset the environment with height constraint metrics."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -0.01, 0.01
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=-0.01, maxval=0.01
        )

        # Spawn humanoid upright instead of lying down
        # Set z position to ~1.25m for standing humanoid
        qpos = qpos.at[2].set(1.25)
        # Rotate 90 degrees around y-axis to stand upright: quaternion [w, x, y, z]
        # For 90° rotation around y-axis: [cos(45°), 0, sin(45°), 0] = [0.707, 0, 0.707, 0]
        qpos = qpos.at[3].set(0.707107)  # w
        qpos = qpos.at[4].set(0.0)       # x
        qpos = qpos.at[5].set(0.707107)  # y
        qpos = qpos.at[6].set(0.0)       # z

        qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=low, maxval=hi
        )

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)

        # Initialize metrics matching step function
        metrics = {
            'forward_reward': zero,
            'height_bonus': zero,
            'reward_quadctrl': zero,
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
            'x_velocity': zero,
            'y_velocity': zero,
            'height': zero,
            'height_violation': zero,
            'height_cost': zero,
            'cost': zero,
        }

        # Initialize info dictionary with cost (required for PPO Lagrange v2)
        info = {
            "cost": zero,
            "height": zero,
            "height_violation": zero,
            "step_count": 0,
        }

        return State(pipeline_state, obs, reward, done, metrics, info)

    def _get_obs(
            self, pipeline_state: base.State, action: jax.Array
    ) -> jax.Array:
        """Observes humanoid body position, velocities, and angles."""
        position = pipeline_state.q[2:]
        velocity = pipeline_state.qd

        com, inertia, mass_sum, x_i = self._com(pipeline_state)
        cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
        com_inertia = jp.hstack(
            [cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]]
        )

        xd_i = (
            base.Transform.create(pos=x_i.pos - pipeline_state.x.pos)
            .vmap()
            .do(pipeline_state.xd)
        )
        com_vel = inertia.mass[:, None] * xd_i.vel / mass_sum
        com_ang = xd_i.ang
        com_velocity = jp.hstack([com_vel, com_ang])

        qfrc_actuator = actuator.to_tau(
            self.sys, action, pipeline_state.q, pipeline_state.qd
        )

        # external_contact_forces are excluded
        return jp.concatenate([
            position,
            velocity,
            com_inertia.ravel(),
            com_velocity.ravel(),
            qfrc_actuator,
        ])

    def _com(self, pipeline_state: base.State) -> jax.Array:
        """Calculate center of mass."""
        inertia = self.sys.link.inertia
        if self.backend in ['spring', 'positional']:
            inertia = inertia.replace(
                i=jax.vmap(jp.diag)(
                    jax.vmap(jp.diagonal)(inertia.i)
                    ** (1 - self.sys.spring_inertia_scale)
                ),
                mass=inertia.mass ** (1 - self.sys.spring_mass_scale),
            )
        mass_sum = jp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
        com = (
            jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        )
        return com, inertia, mass_sum, x_i
