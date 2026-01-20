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

"""Safe Ant environment with leg-lifting constraints.

The ant must learn to walk while keeping certain legs off the ground.

Difficulty levels:
  - Level 1: Keep front-left leg off the ground
  - Level 2: Keep front-left and back-right (diagonal) legs off the ground
  - Level 3: Only back-right leg may touch the ground (hop on one leg)
"""

from typing import Tuple

import jax
import mujoco
from jax import numpy as jp

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath


class SafeAnt(PipelineEnv):
    """Ant that must keep certain legs off the ground.

    The ant receives a cost each timestep when a restricted leg touches the ground.
    Ground contact is detected by checking if the foot z-position is below a threshold.

    Args:
        difficulty: 1, 2, or 3 - determines which legs must stay off ground
        ground_contact_threshold: z-position below which a foot is considered touching
        cost_scale: multiplier for the per-step cost
    """

    # Foot body names in the XML
    FOOT_BODIES = {
        'front_left': 'front_left_leg',   # The body chain ends here
        'front_right': 'front_right_leg',
        'back_left': 'back_leg',
        'back_right': 'right_back_leg',
    }

    def __init__(
            self,
            difficulty: int = 1,
            ground_contact_threshold: float = 0.15,
            cost_scale: float = 1.0,
            ctrl_cost_weight: float = 0.5,
            healthy_reward: float = 1.0,
            terminate_when_unhealthy: bool = True,
            healthy_z_range: Tuple[float, float] = (0.2, 1.0),
            reset_noise_scale: float = 0.1,
            exclude_current_positions_from_observation: bool = True,
            episode_length: int = 2000,
            backend: str = 'mjx',
            **kwargs,
    ):
        assert difficulty in [1, 2, 3], "Difficulty must be 1, 2, or 3"

        path = epath.resource_path('brax') / 'envs/assets/safe/ant.xml'
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
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

        # Get body indices for feet
        # The foot is at the end of each leg chain - we need the deepest body
        # Looking at XML: front_left_leg -> aux_1 -> (unnamed body with foot)
        # We'll get the body that contains the foot geom

        # Get geom-to-body mapping for foot geoms
        self._foot_geom_ids = {}
        self._foot_body_ids = {}

        geom_names = ['left_foot_geom', 'right_foot_geom', 'third_foot_geom', 'fourth_foot_geom']
        foot_keys = ['front_left', 'front_right', 'back_left', 'back_right']

        for key, geom_name in zip(foot_keys, geom_names):
            geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            body_id = mj_model.geom_bodyid[geom_id]
            self._foot_geom_ids[key] = geom_id
            self._foot_body_ids[key] = body_id

        # Determine which feet are restricted based on difficulty
        if difficulty == 1:
            # Level 1: Front-left leg must stay off ground
            self._restricted_feet = ['front_left']
        elif difficulty == 2:
            # Level 2: Diagonal legs (front-left and back-right) must stay off
            self._restricted_feet = ['front_left', 'back_right']
        else:  # difficulty == 3
            # Level 3: Only back-right can touch (all others restricted)
            self._restricted_feet = ['front_left', 'front_right', 'back_left']

        # Convert to body ID array for vectorized checking
        self._restricted_body_ids = jp.array(
            [self._foot_body_ids[k] for k in self._restricted_feet],
            dtype=jp.int32
        )

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
        reward = forward_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        # Count how many restricted feet are on ground
        feet_on_ground = self._count_feet_on_ground(pipeline_state)

        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
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

    def _calculate_foot_contact_cost(self, pipeline_state: base.State) -> jax.Array:
        """Calculate cost based on restricted feet touching the ground.

        A foot is considered touching if its z-position is below the threshold.
        """
        # Get z-positions of all restricted foot bodies
        foot_z_positions = pipeline_state.x.pos[self._restricted_body_ids, 2]

        # Check which feet are on the ground
        feet_touching = foot_z_positions < self._ground_threshold

        # Cost is the number of restricted feet touching ground
        cost = jp.sum(feet_touching.astype(jp.float32))

        return self._cost_scale * cost

    def _count_feet_on_ground(self, pipeline_state: base.State) -> jax.Array:
        """Count how many restricted feet are on the ground."""
        foot_z_positions = pipeline_state.x.pos[self._restricted_body_ids, 2]
        feet_touching = foot_z_positions < self._ground_threshold
        return jp.sum(feet_touching.astype(jp.float32))

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position and velocities."""
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
