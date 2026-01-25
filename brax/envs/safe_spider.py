"""Safe Spider environment with leg-lifting constraints.

A 6-legged spider that must learn to walk while keeping certain legs off the ground.

Difficulty levels:
  - Level 1: Keep 2 legs up (front-left + back-right diagonal)
  - Level 2: Keep 3 legs up (front-left + mid-right + back-left - alternating tripod)
  - Level 3: Keep 4 legs up (only mid-left + mid-right may touch - center legs only)
"""

from typing import List, Optional, Sequence, Tuple, Union

import jax
import mujoco
import numpy as np
from etils import epath
from jax import numpy as jp

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf


class SafeSpider(PipelineEnv):
    """Spider that must keep certain legs off the ground.

    The spider receives a cost each timestep when a restricted leg touches the ground.
    Ground contact is detected using MuJoCo's native contact detection system.

    Args:
        restricted_feet: list of foot names that must stay off ground
        cost_scale: multiplier for the per-step cost
    """

    # Foot geom names and their local offsets from parent body (from XML)
    FOOT_GEOMS = {
        'front_left': ('foot_fl_geom', (0.3, 0.3, 0.0)),
        'front_right': ('foot_fr_geom', (-0.3, 0.3, 0.0)),
        'mid_left': ('foot_ml_geom', (0.35, 0.0, 0.0)),
        'mid_right': ('foot_mr_geom', (-0.35, 0.0, 0.0)),
        'back_left': ('foot_bl_geom', (0.3, -0.3, 0.0)),
        'back_right': ('foot_br_geom', (-0.3, -0.3, 0.0)),
    }

    # Indicator geom names for visualization (transparent red spheres)
    INDICATOR_GEOMS = {
        'front_left': 'indicator_fl',
        'front_right': 'indicator_fr',
        'mid_left': 'indicator_ml',
        'mid_right': 'indicator_mr',
        'back_left': 'indicator_bl',
        'back_right': 'indicator_br',
    }

    def __init__(
            self,
            restricted_feet=None,
            cost_scale: float = 1.0,
            reward_scale: float = 0.01,
            ctrl_cost_weight: float = 0.5,
            healthy_reward: float = 1.0,
            terminate_when_unhealthy: bool = True,
            healthy_z_range: Tuple[float, float] = (0.2, 1.0),
            reset_noise_scale: float = 0.1,
            exclude_current_positions_from_observation: bool = True,
            episode_length: int = 500,
            backend: str = 'mjx',
            **kwargs,
    ):
        path = epath.resource_path('brax') / 'envs/assets/safe/spider.xml'
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
        self._cost_scale = cost_scale
        self._reward_scale = reward_scale
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._restricted_feet = restricted_feet

        # Get geom IDs for contact-based detection
        self._floor_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self._foot_geom_ids = {}
        for foot_key, (geom_name, _) in self.FOOT_GEOMS.items():
            self._foot_geom_ids[foot_key] = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

        # Store restricted foot geom IDs as array for vectorized contact checking
        if self._restricted_feet:
            self._restricted_geom_ids = jp.array(
                [self._foot_geom_ids[k] for k in self._restricted_feet],
                dtype=jp.int32
            )
        else:
            self._restricted_geom_ids = jp.array([], dtype=jp.int32)

        # Get indicator geom IDs for visualization
        self._indicator_geom_ids = {}
        for foot_key, indicator_name in self.INDICATOR_GEOMS.items():
            self._indicator_geom_ids[foot_key] = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_GEOM, indicator_name
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

    def _check_foot_floor_contacts(self, pipeline_state: base.State) -> jax.Array:
        """Check which restricted feet are in contact with the floor.

        Uses MuJoCo's actual contact detection for accurate results.
        Returns a boolean array of shape (num_restricted_feet,).
        """
        # Get contact geom pairs from MJX - shape (max_ncon, 2)
        contact_geom = pipeline_state.contact.geom
        # Contact is active when dist <= 0 (penetrating or touching)
        contact_dist = pipeline_state.contact.dist
        active_contacts = contact_dist <= 0

        # Check each restricted foot for contact with floor
        contacts = []
        for foot_key in self._restricted_feet:
            foot_geom_id = self._foot_geom_ids[foot_key]

            # Check if this foot geom is in contact with the floor
            # Contact pairs can be in either order
            is_foot_floor_contact = (
                    ((contact_geom[:, 0] == foot_geom_id) & (contact_geom[:, 1] == self._floor_geom_id)) |
                    ((contact_geom[:, 1] == foot_geom_id) & (contact_geom[:, 0] == self._floor_geom_id))
            )
            # Only count if contact is active
            in_contact = jp.any(is_foot_floor_contact & active_contacts)
            contacts.append(in_contact)

        return jp.stack(contacts) if contacts else jp.array([], dtype=bool)

    def _calculate_foot_contact_cost(self, pipeline_state: base.State) -> jax.Array:
        """Calculate cost based on restricted feet touching the ground.

        Uses MuJoCo's contact detection for accurate floor contact sensing.
        """
        if len(self._restricted_feet) == 0:
            return jp.float32(0.0)

        feet_touching = self._check_foot_floor_contacts(pipeline_state)
        cost = jp.sum(feet_touching.astype(jp.float32))

        return self._cost_scale * cost

    def _count_feet_on_ground(self, pipeline_state: base.State) -> jax.Array:
        """Count how many restricted feet are in contact with the floor."""
        if len(self._restricted_feet) == 0:
            return jp.float32(0.0)

        feet_touching = self._check_foot_floor_contacts(pipeline_state)
        return jp.sum(feet_touching.astype(jp.float32))

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe spider body position and velocities."""
        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        if self._exclude_current_positions_from_observation:
            qpos = pipeline_state.q[2:]

        return jp.concatenate([qpos, qvel])

    @property
    def restricted_feet(self) -> list:
        """Return list of feet that must stay off ground."""
        return self._restricted_feet

    def render(
            self,
            trajectory: Union[List[base.State], base.State],
            height: int = 240,
            width: int = 320,
            camera: Optional[str] = None,
    ) -> Union[Sequence[np.ndarray], np.ndarray]:
        """Renders trajectory with violation indicators shown as red spheres.

        When a restricted foot touches the ground, its indicator sphere becomes
        visible (red, semi-transparent) to visualize the constraint violation.
        """
        renderer = mujoco.Renderer(self.sys.mj_model, height=height, width=width)
        camera = camera or -1

        # Store original indicator colors to restore after rendering
        original_rgba = {}
        for foot_key in self._restricted_feet:
            indicator_id = self._indicator_geom_ids[foot_key]
            original_rgba[indicator_id] = self.sys.mj_model.geom_rgba[indicator_id].copy()

        def get_image(state: base.State):
            # Check which restricted feet are violating (touching floor)
            feet_touching = self._check_foot_floor_contacts_render(state)

            # Update indicator colors based on violations
            for i, foot_key in enumerate(self._restricted_feet):
                indicator_id = self._indicator_geom_ids[foot_key]
                if feet_touching[i]:
                    # Violation: show a red semi-transparent sphere
                    self.sys.mj_model.geom_rgba[indicator_id] = [1.0, 0.0, 0.0, 0.5]
                else:
                    # No violation: keep the sphere transparent
                    self.sys.mj_model.geom_rgba[indicator_id] = [1.0, 0.0, 0.0, 0.0]

            # Render the frame
            d = mujoco.MjData(self.sys.mj_model)
            d.qpos, d.qvel = np.asarray(state.q), np.asarray(state.qd)
            if hasattr(state, 'mocap_pos') and hasattr(state, 'mocap_quat'):
                d.mocap_pos, d.mocap_quat = state.mocap_pos, state.mocap_quat
            mujoco.mj_forward(self.sys.mj_model, d)
            renderer.update_scene(d, camera=camera)
            return renderer.render()

        if isinstance(trajectory, list):
            images = [get_image(s) for s in trajectory]
        else:
            images = get_image(trajectory)

        # Restore original colors
        for indicator_id, rgba in original_rgba.items():
            self.sys.mj_model.geom_rgba[indicator_id] = rgba

        return images

    def _check_foot_floor_contacts_render(self, state: base.State) -> List[bool]:
        """Check foot contacts for rendering"""

        contact_geom = np.asarray(state.contact.geom)
        contact_dist = np.asarray(state.contact.dist)
        active_contacts = contact_dist <= 0

        contacts = []
        for foot_key in self._restricted_feet:
            foot_geom_id = self._foot_geom_ids[foot_key]

            is_foot_floor_contact = (
                    ((contact_geom[:, 0] == foot_geom_id) & (contact_geom[:, 1] == self._floor_geom_id)) |
                    ((contact_geom[:, 1] == foot_geom_id) & (contact_geom[:, 0] == self._floor_geom_id))
            )
            in_contact = np.any(is_foot_floor_contact & active_contacts)
            contacts.append(bool(in_contact))

        return contacts
