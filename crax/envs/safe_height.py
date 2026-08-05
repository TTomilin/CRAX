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

"""Height-constrained locomotion environment.

Abstract base class and agent-specific implementations for height-constrained tasks.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import jax
import mujoco
from jax import numpy as jp
from etils import epath

from crax import actuator
from crax import base
from crax import math as brax_math
from crax.envs.base import PipelineEnv, State
from crax.io import mjcf


class SafeHeight(PipelineEnv, ABC):
    """Abstract base class for height-constrained locomotion environments.

    The agent must learn to move forward while staying below a height constraint.
    The constraint violation (cost) is computed based on the agent's height
    exceeding a specified threshold.

    Subclasses must implement agent-specific properties for:
    - XML file path
    - Body configuration
    - Height tracking geometry
    """

    @property
    @abstractmethod
    def agent_xml_path(self) -> str:
        """Return the path to the agent's XML file."""
        pass

    @property
    @abstractmethod
    def head_local_offset(self) -> jp.ndarray:
        """Return the local offset of the head from the torso body."""
        pass

    @property
    @abstractmethod
    def head_radius(self) -> float:
        """Return the radius of the head geometry for height calculation."""
        pass

    @property
    @abstractmethod
    def default_spawn_height(self) -> float:
        """Return the default z-position for spawning the agent."""
        pass

    @property
    @abstractmethod
    def default_spawn_rotation(self) -> Tuple[float, float, float, float]:
        """Return the default quaternion [w, x, y, z] for spawning."""
        pass

    @property
    @abstractmethod
    def default_healthy_z_range(self) -> Tuple[float, float]:
        """Return the default (min, max) z-range for healthy state."""
        pass

    @abstractmethod
    def _get_gear_for_backend(self, backend: str) -> jp.ndarray:
        """Return actuator gear values for the given backend."""
        pass

    def __init__(
            self,
            forward_reward_weight=5.0,
            ctrl_cost_weight=0.1,
            healthy_reward=5.0,
            terminate_when_unhealthy=False,
            healthy_z_range: Tuple[float, float] = None,
            reset_noise_scale=1e-2,
            exclude_current_positions_from_observation=True,
            max_height: float = None,
            hinge_margin: float = 0.08,
            height_cost_weight: float = 0.1,
            progress_check_window: int = 200,
            min_forward_progress: float = 0.5,
            episode_length: int = 1000,
            backend: str = 'generalized',
            **kwargs,
    ):
        """Initialize the height-constrained environment.

        Args:
            forward_reward_weight: Weight for forward velocity reward.
            ctrl_cost_weight: Weight for control cost penalty.
            healthy_reward: Reward for staying healthy (alive).
            terminate_when_unhealthy: Whether to terminate episode when unhealthy.
            healthy_z_range: (min, max) z-range for healthy state.
            reset_noise_scale: Scale of noise added to initial state.
            exclude_current_positions_from_observation: Whether to exclude x,y from obs.
            max_height: Maximum allowed height. Exceeding incurs cost.
            hinge_margin: Soft hinge width for height constraint.
            height_cost_weight: Weight for height constraint violation cost.
            progress_check_window: Number of steps to check for forward progress.
            min_forward_progress: Minimum distance to move in progress_check_window steps.
            episode_length: Maximum number of steps per episode.
            backend: Physics backend ('generalized', 'spring', 'positional', 'mjx').
        """
        self.episode_length = episode_length
        self._max_height = float(max_height)
        self._hinge_margin = float(hinge_margin)
        self._height_cost_weight = height_cost_weight

        # Use default healthy z range if not provided
        if healthy_z_range is None:
            healthy_z_range = self.default_healthy_z_range

        # Load XML and set the ceiling height
        path = epath.resource_path('crax') / self.agent_xml_path
        xml_string = path.read_text()

        # Replace the placeholder with the actual max_height value
        xml_string = xml_string.replace('HEIGHT_PLACEHOLDER', str(self._max_height))

        # Parse the modified XML
        sys = mjcf.loads(xml_string)

        n_frames = 5

        if backend in ['spring', 'positional']:
            sys = sys.tree_replace({'opt.timestep': 0.0015})
            n_frames = 10
            gear = self._get_gear_for_backend(backend)
            sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

        if backend == 'mjx':
            sys = sys.tree_replace({
                'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
                'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                'opt.iterations': 1,
                'opt.ls_iterations': 4,
            })

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self.episode_length = episode_length
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        # Forward progress termination settings
        self._progress_check_window = progress_check_window
        self._min_forward_progress = min_forward_progress

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics with height constraints."""
        # Scale action from [-1,1] to actuator limits
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        # Store previous state for velocity calculation
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        # Calculate center-of-mass for forward velocity
        com_before, *_ = self._com(pipeline_state0)
        com_after, *_ = self._com(pipeline_state)
        velocity = (com_after - com_before) / self.dt

        # Forward reward
        forward_reward = self._forward_reward_weight * velocity[0]

        # Healthy reward
        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        # Control cost
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        # Calculate height for constraint
        head_tip_height = self._get_head_tip_height(pipeline_state)

        # Height constraint cost (safety constraint): require height <= max_height
        height_excess = jp.maximum(0.0, head_tip_height - self._max_height)
        normalized_excess = height_excess / self._hinge_margin
        height_cost = self._height_cost_weight * normalized_excess

        obs = self._get_obs(pipeline_state, action)

        # Reward structure
        reward = forward_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        # Forward progress termination check
        current_info = getattr(state, 'info', {})
        checkpoint_x = current_info.get('checkpoint_x', 0.0)
        steps_since_checkpoint = current_info.get('steps_since_checkpoint', 0) + 1

        # Check progress every N steps
        progress_since_checkpoint = com_after[0] - checkpoint_x
        no_progress = jp.logical_and(
            steps_since_checkpoint >= self._progress_check_window,
            progress_since_checkpoint < self._min_forward_progress
        )
        done = jp.where(no_progress, 1.0, done)

        # Reset checkpoint if we've made progress or hit the window
        new_checkpoint_x = jp.where(
            steps_since_checkpoint >= self._progress_check_window,
            com_after[0],
            checkpoint_x
        )
        new_steps_since_checkpoint = jp.where(
            steps_since_checkpoint >= self._progress_check_window,
            0,
            steps_since_checkpoint
        )

        # Update metrics
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
            # Height constraint metrics
            head_height=head_tip_height,
            height_violation=height_excess,
            height_cost=height_cost,
            cost=height_cost,
        )

        # Update info dictionary with cost and progress tracking
        step_count = current_info.get('step_count', 0) + 1

        new_info = current_info.copy() if isinstance(current_info, dict) else {}
        new_info.update({
            "cost": height_cost,
            "head_height": head_tip_height,
            "height_violation": height_excess,
            "step_count": step_count,
            "checkpoint_x": new_checkpoint_x,
            "steps_since_checkpoint": new_steps_since_checkpoint,
        })

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, info=new_info
        )

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )

        # Set spawn position and rotation from subclass defaults
        qpos = qpos.at[2].set(self.default_spawn_height)
        w, x, y, z = self.default_spawn_rotation
        qpos = qpos.at[3].set(w)
        qpos = qpos.at[4].set(x)
        qpos = qpos.at[5].set(y)
        qpos = qpos.at[6].set(z)

        qvel = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=low, maxval=hi
        )

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        reward, done, zero = jp.zeros(3)

        # Get initial x position for progress tracking
        initial_com, *_ = self._com(pipeline_state)
        initial_x = initial_com[0]

        # Initialize metrics
        metrics = {
            'forward_reward': zero,
            'reward_linvel': zero,
            'reward_quadctrl': zero,
            'reward_alive': zero,
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
            'x_velocity': zero,
            'y_velocity': zero,
            'head_height': zero,
            'height_violation': zero,
            'height_cost': zero,
            'cost': zero,
        }

        # Initialize info dictionary with cost and progress tracking
        info = {
            "cost": zero,
            "head_height": zero,
            "height_violation": zero,
            "step_count": 0,
            "checkpoint_x": initial_x,
            "steps_since_checkpoint": 0,
        }

        return State(pipeline_state, obs, reward, done, metrics, info)

    def _get_head_tip_height(self, pipeline_state: base.State) -> jax.Array:
        """Calculate the height of the highest point (head tip) of the agent."""
        torso_pos = pipeline_state.x.pos[0]
        torso_rot = pipeline_state.x.rot[0]
        head_offset_world = brax_math.rotate(self.head_local_offset, torso_rot)
        head_center = torso_pos + head_offset_world
        head_tip_height = head_center[2] + self.head_radius
        return head_tip_height

    def _get_obs(
            self, pipeline_state: base.State, action: jax.Array
    ) -> jax.Array:
        """Observes body position, velocities, and angles."""
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


class SafeHeightHumanoid(SafeHeight):
    """Humanoid locomotion environment with height constraints.

    The humanoid must learn to move forward while staying below a height ceiling,
    forcing it to learn to crouch or crawl.
    """

    @property
    def agent_xml_path(self) -> str:
        return 'envs/assets/safe/humanoid_height.xml'

    @property
    def head_local_offset(self) -> jp.ndarray:
        return jp.array([-0.15, 0.0, 0.0])

    @property
    def head_radius(self) -> float:
        return 0.09

    @property
    def default_spawn_height(self) -> float:
        return 1.25

    @property
    def default_spawn_rotation(self) -> Tuple[float, float, float, float]:
        # Rotate 90 degrees around y-axis to stand upright: quaternion [w, x, y, z]
        return (0.707107, 0.0, 0.707107, 0.0)

    @property
    def default_healthy_z_range(self) -> Tuple[float, float]:
        return (1.0, 2.0)

    def _get_gear_for_backend(self, backend: str) -> jp.ndarray:
        return jp.array([
            350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0,
            350.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0
        ])
