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

"""Leg-lifting constraint environment.

Abstract base class and agent-specific implementations for leg-lifting tasks.
The agent must learn to walk while keeping certain legs off the ground.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple, Union

import jax
import mujoco
import numpy as np
from jax import numpy as jp
from etils import epath

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf


class SafeLift(PipelineEnv, ABC):
    """Abstract base class for leg-lifting constraint environments.

    The agent receives a cost each timestep when a restricted leg touches the ground.
    Subclasses must implement agent-specific properties for:
    - XML file path
    - Foot geom definitions
    - Restricted feet based on difficulty
    """

    @property
    @abstractmethod
    def agent_xml_path(self) -> str:
        """Return the path to the agent's XML file."""
        pass

    @property
    @abstractmethod
    def foot_geoms(self) -> Dict[str, Tuple[str, Tuple[float, float, float]]]:
        """Return dict mapping foot names to (geom_name, local_offset) tuples."""
        pass

    @abstractmethod
    def get_restricted_feet(self, difficulty: int) -> List[str]:
        """Return list of foot names that must stay off ground for given difficulty."""
        pass

    @property
    @abstractmethod
    def default_healthy_z_range(self) -> Tuple[float, float]:
        """Return the default (min, max) z-range for healthy state."""
        pass

    @property
    def uses_contact_detection(self) -> bool:
        """Whether to use contact-based detection (True) or z-position (False)."""
        return False

    def __init__(
            self,
            difficulty: int = 1,
            restricted_feet: Optional[List[str]] = None,
            ground_contact_threshold: float = 0.15,
            cost_scale: float = 1.0,
            reward_scale: float = 1.0,
            ctrl_cost_weight: float = 0.5,
            healthy_reward: float = 1.0,
            terminate_when_unhealthy: bool = True,
            healthy_z_range: Tuple[float, float] = None,
            reset_noise_scale: float = 0.1,
            exclude_current_positions_from_observation: bool = True,
            episode_length: int = 2000,
            backend: str = 'mjx',
            **kwargs,
    ):
        """Initialize the leg-lifting constraint environment.

        Args:
            difficulty: Difficulty level (determines which feet are restricted).
            restricted_feet: Explicit list of restricted feet (overrides difficulty).
            ground_contact_threshold: Z-position below which a foot is touching.
            cost_scale: Multiplier for the per-step cost.
            reward_scale: Multiplier for the reward.
            ctrl_cost_weight: Weight for control cost penalty.
            healthy_reward: Reward for staying healthy.
            terminate_when_unhealthy: Whether to terminate when unhealthy.
            healthy_z_range: (min, max) z-range for healthy state.
            reset_noise_scale: Scale of noise added to initial state.
            exclude_current_positions_from_observation: Whether to exclude x,y from obs.
            episode_length: Maximum number of steps per episode.
            backend: Physics backend.
        """
        path = epath.resource_path('brax') / self.agent_xml_path
        mj_model = mujoco.MjModel.from_xml_path(str(path))
        sys = mjcf.load_model(mj_model)

        n_frames = 5

        if backend in ['spring', 'positional']:
            sys = sys.tree_replace({'opt.timestep': 0.005})
            n_frames = 10

        if backend == 'mjx':
            sys = sys.tree_replace({
                'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
                'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                'opt.iterations': 1,
                'opt.ls_iterations': 4,
            })

        if backend == 'positional':
            sys = sys.replace(
                actuator=sys.actuator.replace(
                    gear=200 * jp.ones_like(sys.actuator.gear)
                )
            )

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self.episode_length = episode_length
        self._difficulty = difficulty
        self._ground_threshold = ground_contact_threshold
        self._cost_scale = cost_scale
        self._reward_scale = reward_scale
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range if healthy_z_range else self.default_healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

        # Get foot geom info from subclass
        foot_geoms = self.foot_geoms

        # Get body indices and local offsets for foot geoms
        self._foot_body_ids = {}
        self._foot_local_offsets = {}
        self._foot_geom_ids = {}

        for foot_key, (geom_name, local_offset) in foot_geoms.items():
            geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                body_id = mj_model.geom_bodyid[geom_id]
                self._foot_body_ids[foot_key] = body_id
                self._foot_local_offsets[foot_key] = jp.array(local_offset)
                self._foot_geom_ids[foot_key] = geom_id

        # Get floor geom ID for contact detection
        self._floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')

        # Determine which feet are restricted
        if restricted_feet is not None:
            self._restricted_feet = restricted_feet
        else:
            self._restricted_feet = self.get_restricted_feet(difficulty)

        # Store restricted geom IDs for contact checking
        if self._restricted_feet:
            self._restricted_geom_ids = jp.array(
                [self._foot_geom_ids[k] for k in self._restricted_feet if k in self._foot_geom_ids],
                dtype=jp.int32
            )
        else:
            self._restricted_geom_ids = jp.array([], dtype=jp.int32)

        # Store body IDs and local offsets as arrays for vectorized computation
        self._restricted_body_ids = jp.array(
            [self._foot_body_ids[k] for k in self._restricted_feet if k in self._foot_body_ids],
            dtype=jp.int32
        )
        self._restricted_local_offsets = jp.stack(
            [self._foot_local_offsets[k] for k in self._restricted_feet if k in self._foot_local_offsets]
        ) if self._restricted_feet else jp.zeros((0, 3))

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jp.zeros(3)
        metrics = {
            'reward_forward': zero,
            'reward_survive': zero,
            'reward_ctrl': zero,
            'cost': zero,
            'x_position': zero,
            'y_position': zero,
            'distance_from_origin': zero,
            'x_velocity': zero,
            'y_velocity': zero,
            'feet_on_ground': zero,
        }
        info = {'cost': zero}

        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        # Forward velocity reward
        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = velocity[0]

        # Health check
        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)

        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        # Calculate safety cost: check if restricted feet are on the ground
        cost = self._calculate_foot_contact_cost(pipeline_state)

        obs = self._get_obs(pipeline_state)
        reward = (forward_reward + healthy_reward - ctrl_cost) * self._reward_scale
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        # Count how many restricted feet are on ground
        feet_on_ground = self._count_feet_on_ground(pipeline_state)

        state.metrics.update(
            reward_forward=forward_reward * self._reward_scale,
            reward_survive=healthy_reward * self._reward_scale,
            reward_ctrl=-ctrl_cost * self._reward_scale,
            cost=cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            feet_on_ground=feet_on_ground,
        )

        info = dict(state.info)
        info['cost'] = cost

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            info=info,
        )

    def _get_foot_world_positions(self, pipeline_state: base.State) -> jax.Array:
        """Compute world z-positions of restricted foot geoms."""
        foot_z_positions = []
        for foot_key in self._restricted_feet:
            if foot_key not in self._foot_body_ids:
                continue
            body_id = self._foot_body_ids[foot_key]
            body_pos = pipeline_state.x.pos[body_id]
            body_rot = pipeline_state.x.rot[body_id]
            local_offset = self._foot_local_offsets[foot_key]

            # Transform local offset to world coordinates
            world_offset = math.rotate(body_rot, local_offset)
            foot_world_z = body_pos[2] + world_offset[2]
            foot_z_positions.append(foot_world_z)

        return jp.stack(foot_z_positions) if foot_z_positions else jp.array([])

    def _check_foot_floor_contacts(self, pipeline_state: base.State) -> jax.Array:
        """Check which restricted feet are in contact with the floor using MuJoCo contacts."""
        contact_geom = pipeline_state.contact.geom
        contact_dist = pipeline_state.contact.dist
        active_contacts = contact_dist <= 0

        contacts = []
        for foot_key in self._restricted_feet:
            if foot_key not in self._foot_geom_ids:
                contacts.append(False)
                continue
            foot_geom_id = self._foot_geom_ids[foot_key]

            is_foot_floor_contact = (
                ((contact_geom[:, 0] == foot_geom_id) & (contact_geom[:, 1] == self._floor_geom_id)) |
                ((contact_geom[:, 1] == foot_geom_id) & (contact_geom[:, 0] == self._floor_geom_id))
            )
            in_contact = jp.any(is_foot_floor_contact & active_contacts)
            contacts.append(in_contact)

        return jp.stack(contacts) if contacts else jp.array([], dtype=bool)

    def _calculate_foot_contact_cost(self, pipeline_state: base.State) -> jax.Array:
        """Calculate cost based on restricted feet touching the ground."""
        if len(self._restricted_feet) == 0:
            return jp.float32(0.0)

        if self.uses_contact_detection:
            feet_touching = self._check_foot_floor_contacts(pipeline_state)
        else:
            foot_z_positions = self._get_foot_world_positions(pipeline_state)
            feet_touching = foot_z_positions < self._ground_threshold

        cost = jp.sum(feet_touching.astype(jp.float32))
        return self._cost_scale * cost

    def _count_feet_on_ground(self, pipeline_state: base.State) -> jax.Array:
        """Count how many restricted feet are on the ground."""
        if len(self._restricted_feet) == 0:
            return jp.float32(0.0)

        if self.uses_contact_detection:
            feet_touching = self._check_foot_floor_contacts(pipeline_state)
        else:
            foot_z_positions = self._get_foot_world_positions(pipeline_state)
            feet_touching = foot_z_positions < self._ground_threshold

        return jp.sum(feet_touching.astype(jp.float32))

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe body position and velocities."""
        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        if self._exclude_current_positions_from_observation:
            qpos = pipeline_state.q[2:]

        return jp.concatenate([qpos, qvel])

    @property
    def difficulty(self) -> int:
        """Return the current difficulty level."""
        return self._difficulty

    @property
    def restricted_feet(self) -> list:
        """Return list of feet that must stay off ground."""
        return self._restricted_feet


class SafeLiftHumanoid(SafeLift):
    """Humanoid that must hop on one leg while moving forward.

    Unlike ant/spider which have 4+ legs, humanoid only has 2 legs.
    Difficulty is implemented via cost weight and airtime bonuses:
      - Level 1: One leg restricted, base cost weight
      - Level 2: One leg restricted, higher cost weight
      - Level 3: One leg restricted, highest cost weight + airtime bonus

    This integrates the functionality from humanoid_hop.py and humanoid_hop_airtime.py.
    """

    @property
    def agent_xml_path(self) -> str:
        return 'envs/assets/humanoid.xml'

    @property
    def foot_geoms(self) -> Dict[str, Tuple[str, Tuple[float, float, float]]]:
        # In humanoid, foot geoms are attached to shin bodies
        return {
            'left': ('left_foot', (0.0, 0.0, -0.2)),
            'right': ('right_foot', (0.0, 0.0, -0.2)),
        }

    def get_restricted_feet(self, difficulty: int) -> List[str]:
        # Humanoid always restricts one leg (the non-hopping leg)
        # The hopping_leg parameter determines which leg to hop on
        return [self._non_hopping_leg]

    @property
    def default_healthy_z_range(self) -> Tuple[float, float]:
        return (1.0, 2.1)

    @property
    def uses_contact_detection(self) -> bool:
        return True  # Humanoid uses contact-based detection

    # Difficulty-based cost weights and airtime settings
    _DIFFICULTY_SETTINGS = {
        1: {'cost_scale': 1.0, 'airtime_reward_weight': 0.0},
        2: {'cost_scale': 2.0, 'airtime_reward_weight': 0.0},
        3: {'cost_scale': 3.0, 'airtime_reward_weight': 5.0},
    }

    def __init__(
            self,
            difficulty: int = 1,
            hopping_leg: str = 'left',
            contact_threshold: float = 0.5,
            airtime_exponent: float = 1.0,
            **kwargs,
    ):
        """Initialize the humanoid hopping environment.

        Args:
            difficulty: Difficulty level (1, 2, or 3). Higher = stricter constraints.
            hopping_leg: Which leg to hop on ('left' or 'right').
            contact_threshold: Threshold for contact detection.
            airtime_exponent: Exponent for airtime bonus shaping (level 3 only).
            **kwargs: Additional arguments passed to SafeLift.
        """
        assert difficulty in [1, 2, 3], "Difficulty must be 1, 2, or 3"
        assert hopping_leg in ['left', 'right'], "hopping_leg must be 'left' or 'right'"

        self._hopping_leg = hopping_leg
        self._non_hopping_leg = 'right' if hopping_leg == 'left' else 'left'
        self._contact_threshold = contact_threshold
        self._airtime_exponent = airtime_exponent

        # Get difficulty-based settings
        settings = self._DIFFICULTY_SETTINGS[difficulty]
        cost_scale = kwargs.pop('cost_scale', settings['cost_scale'])
        self._airtime_reward_weight = kwargs.pop(
            'airtime_reward_weight', settings['airtime_reward_weight']
        )

        super().__init__(
            difficulty=difficulty,
            cost_scale=cost_scale,
            ground_contact_threshold=contact_threshold,
            **kwargs,
        )

        # Find shin body indices for contact detection
        self._left_shin_body_idx = None
        self._right_shin_body_idx = None
        self._find_shin_indices()

    def _find_shin_indices(self):
        """Find body indices for shin bodies (which contain foot geoms)."""
        if hasattr(self.sys, 'link_names'):
            link_names = self.sys.link_names
            try:
                self._left_shin_body_idx = link_names.index('left_shin')
                self._right_shin_body_idx = link_names.index('right_shin')
                return
            except ValueError:
                pass

        # Fallback to foot body names
        if hasattr(self.sys, 'link_names'):
            link_names = self.sys.link_names
            try:
                self._left_shin_body_idx = link_names.index('left_foot')
                self._right_shin_body_idx = link_names.index('right_foot')
                return
            except ValueError:
                pass

    def _check_foot_floor_contacts(self, pipeline_state: base.State) -> jax.Array:
        """Check which feet are in contact using contact detection."""
        if not hasattr(pipeline_state, 'contact') or pipeline_state.contact is None:
            return jp.array([False, False])

        contact = pipeline_state.contact
        if not hasattr(contact, 'link_idx'):
            return jp.array([False, False])

        link_idx_a, link_idx_b = contact.link_idx
        if hasattr(contact, 'dist'):
            active_mask = contact.dist < 0.0
        else:
            active_mask = jp.ones_like(link_idx_a, dtype=bool)

        active_mask = jp.asarray(active_mask)

        # Check left foot contact
        left_mask = active_mask & (
            (link_idx_a == self._left_shin_body_idx) |
            (link_idx_b == self._left_shin_body_idx)
        )
        left_contact = jp.any(left_mask)

        # Check right foot contact
        right_mask = active_mask & (
            (link_idx_a == self._right_shin_body_idx) |
            (link_idx_b == self._right_shin_body_idx)
        )
        right_contact = jp.any(right_mask)

        # Return contact status for restricted feet only
        contacts = []
        for foot_key in self._restricted_feet:
            if foot_key == 'left':
                contacts.append(left_contact)
            elif foot_key == 'right':
                contacts.append(right_contact)

        return jp.stack(contacts) if contacts else jp.array([], dtype=bool)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep with optional airtime bonus (level 3)."""
        # Get base step behavior
        next_state = super().step(state, action)

        # Add airtime bonus for level 3
        if self._airtime_reward_weight > 0:
            pipeline_state = next_state.pipeline_state

            # Check both feet contact
            left_contact, right_contact = self._get_foot_contacts(pipeline_state)
            any_contact = jp.maximum(left_contact, right_contact)

            # Airtime score: higher when both feet are off ground
            airborne = jp.clip(1.0 - any_contact, 0.0, 1.0)
            airtime_score = jp.power(airborne, self._airtime_exponent)
            airtime_reward = self._airtime_reward_weight * airtime_score

            # Add airtime to reward
            new_reward = next_state.reward + airtime_reward

            # Update metrics
            next_state.metrics.update(airtime_reward=airtime_reward)

            # Update info
            info = dict(next_state.info)
            info['airtime_reward'] = airtime_reward

            next_state = next_state.replace(reward=new_reward, info=info)

        return next_state

    def _get_foot_contacts(self, pipeline_state: base.State) -> Tuple[jax.Array, jax.Array]:
        """Get contact status for both feet."""
        if not hasattr(pipeline_state, 'contact') or pipeline_state.contact is None:
            return jp.array(0.0), jp.array(0.0)

        contact = pipeline_state.contact
        if not hasattr(contact, 'link_idx'):
            return jp.array(0.0), jp.array(0.0)

        link_idx_a, link_idx_b = contact.link_idx
        if hasattr(contact, 'dist'):
            active_mask = contact.dist < 0.0
        else:
            active_mask = jp.ones_like(link_idx_a, dtype=bool)

        active_mask = jp.asarray(active_mask)

        left_mask = active_mask & (
            (link_idx_a == self._left_shin_body_idx) |
            (link_idx_b == self._left_shin_body_idx)
        )
        right_mask = active_mask & (
            (link_idx_a == self._right_shin_body_idx) |
            (link_idx_b == self._right_shin_body_idx)
        )

        left_contact = jp.where(jp.any(left_mask), 1.0, 0.0)
        right_contact = jp.where(jp.any(right_mask), 1.0, 0.0)

        return left_contact.astype(jp.float32), right_contact.astype(jp.float32)
