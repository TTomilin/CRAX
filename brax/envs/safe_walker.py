from typing import List, Optional, Tuple

import jax
import mujoco
from jax import numpy as jp

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.envs.env_utils import generate_goal_xml_from_base
from brax.envs.goals import GoalManager
from brax.envs.hazards import HazardManager
from brax.io import mjcf


class SafeWalker(PipelineEnv):
    """Walker2d with randomly placed, passable hazards along its path.

    - Hazards (cylinder/cube; 'sphere' maps to cylinder) are non-collidable and
      only contribute to a per-step safety cost when a foot is inside them.
    - Hazards are placed ahead of the agent along +x with random gaps between
      consecutive hazards and small lateral jitter.
    - Costs accumulate if both feet are inside hazards and/or multiple hazards
      overlap.
    """

    def __init__(
            self,
            num_hazards: int = 100,
            hazard_types: Optional[List[str]] = None,
            hazard_radius: float = 0.25,
            cube_half_extent: float = 0.20,
            hazard_height: float = 0.05,
            min_gap: float = 0.5,
            max_gap: float = 2.0,
            lateral_jitter: float = 0.25,
            ctrl_cost_weight: float = 1e-3,
            healthy_reward: float = 1.0,
            terminate_when_unhealthy: bool = True,
            healthy_z_range: Tuple[float, float] = (0.8, 2.0),
            healthy_angle_range: Tuple[float, float] = (-1.0, 1.0),
            reset_noise_scale: float = 5e-3,
            exclude_current_positions_from_observation: bool = True,
            **kwargs,
    ):
        self._reward_scaler = kwargs.get("reward", {}).get("scaler", 0.01)
        self._cost_scaler = kwargs.get("cost", {}).get("scaler", 0.1)
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # safety + hazard params
        self._num_hazards = int(num_hazards)
        self._hazard_types = hazard_types or ["cylinder", "cube"]
        self._hazard_radius = hazard_radius
        self._cube_half_extent = cube_half_extent
        self._hazard_height = hazard_height
        self._min_gap = min_gap
        self._max_gap = max_gap
        self._lateral_jitter = lateral_jitter
        self._cost_scale = kwargs.get("cost", {}).get("scaler", 1.0)

        # Build hazards via HazardManager; passable (collidable=False)
        hz_mgr = HazardManager()
        for i in range(self._num_hazards):
            t = self._hazard_types[i % len(self._hazard_types)].lower()
            if t == "sphere":
                # TODO implement sphere hazard
                t = "cylinder"
            if t == "cylinder":
                size = self._hazard_radius
            elif t in ("cube", "rect"):
                size = self._cube_half_extent
                t = "cube"  # use cube type for XML
            else:
                raise ValueError(f"Unsupported hazard type: {t}")

            hz_mgr.add_hazards(
                t,
                1,
                positions=[(0.0, 0.0, self._hazard_height)],
                size=size,
                height=self._hazard_height,
                collidable=False,
                fixed=False,
                density=1.0,
            )

        # No explicit goals for walker; but GoalManager is required by builder
        goal_mgr = GoalManager()

        # Build XML from base walker2d asset and injected hazards
        base_name = "walker2d.xml"
        xml_path = generate_goal_xml_from_base(base_name, goal_manager=goal_mgr, hazard_manager=hz_mgr)
        try:
            mj_model = mujoco.MjModel.from_xml_path(xml_path)
        finally:
            import os
            if os.path.exists(xml_path):
                os.unlink(xml_path)

        physics = kwargs.get("physics", {})
        backend = physics.get("backend", "mjx")
        n_frames = physics.get("n_frames", 4)

        sys = mjcf.load_model(mj_model)
        super().__init__(sys, backend=backend, n_frames=n_frames)

        # Cache key body ids: torso, feet
        def _link_idx(name: str) -> int:
            try:
                return sys.link_names.index(name)
            except ValueError as e:
                raise ValueError(f"Link '{name}' not found in sys.link_names: {self.sys.link_names}") from e

        # NOTE: names must match sys.link_names exactly
        self._torso_link_id = _link_idx("torso")
        self._right_foot_link_id = _link_idx("foot")
        self._left_foot_link_id = _link_idx("foot_left")

        # Hazard mocap ids + shape buffers
        self._hazard_mocap_ids = []
        hazard_is_rect = []
        hazard_radii = []
        hazard_half_extents = []
        for idx, hz in enumerate(hz_mgr.hazards, start=1):
            b = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, f"hazard{idx}")
            mid = int(mj_model.body_mocapid[b])
            if mid < 0:
                raise RuntimeError("Hazard body is not mocap")
            self._hazard_mocap_ids.append(mid)
            htype = getattr(hz, "hazard_type", "cylinder")
            if htype == "cylinder":
                hazard_is_rect.append(False)
                hazard_radii.append(float(hz.size))
                hazard_half_extents.append(jp.array([0.0, 0.0]))
            else:
                hazard_is_rect.append(True)
                if isinstance(hz.size, tuple):
                    sx, sy = hz.size
                else:
                    sx = sy = self._cube_half_extent
                hazard_radii.append(0.0)
                hazard_half_extents.append(jp.array([float(sx), float(sy)]))

        self._hazard_is_rect = jp.array(hazard_is_rect, dtype=jp.bool_)
        self._hazard_radii = jp.array(hazard_radii, dtype=jp.float32)
        self._hazard_half_extents = (
            jp.stack(hazard_half_extents) if len(hazard_half_extents) > 0 else jp.zeros((0, 2))
        )

    # ---------------- Core RL API ----------------

    def reset(self, rng: jax.Array) -> State:
        # noise init like walker2d baseline
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        rng, r1, r2, r_h = jax.random.split(rng, 4)
        q = self.sys.init_q + jax.random.uniform(r1, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = jax.random.uniform(r2, (self.sys.qd_size(),), minval=low, maxval=hi)

        # sample hazard positions along +x with random gaps and lateral jitter
        n_h = len(self._hazard_mocap_ids)

        def sample_positions(key):
            # start a bit ahead of the agent
            key, k0 = jax.random.split(key)
            start = jax.random.uniform(k0, (), minval=1.0, maxval=2.0)
            xs = []
            cur = start
            key_loop = key
            for i in range(n_h):
                key_loop, kg, ky = jax.random.split(key_loop, 3)
                gap = jax.random.uniform(kg, (), minval=self._min_gap, maxval=self._max_gap)
                cur = cur + gap
                y = jax.random.uniform(ky, (), minval=-self._lateral_jitter, maxval=self._lateral_jitter)
                xs.append(jp.array([cur, y, self._hazard_height], dtype=jp.float32))
            return jp.stack(xs) if n_h > 0 else jp.zeros((0, 3), dtype=jp.float32)

        hazards_pos = sample_positions(r_h)

        pipeline_state = self.pipeline_init(q, qd)

        # set hazard mocap positions
        ids = jp.array(self._hazard_mocap_ids, dtype=jp.int32)
        mpos = pipeline_state.mocap_pos.at[ids].set(hazards_pos)
        pipeline_state = pipeline_state.replace(mocap_pos=mpos)

        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)

        # metrics/info
        metrics = {
            "x_position": pipeline_state.q[0],
            "x_velocity": pipeline_state.qd[0],
            "reward_forward": zero,
            "reward_healthy": zero,
            "ctrl_cost": zero,
            "cost": zero,
        }
        info = {"hazard_positions": hazards_pos, "cost": zero}

        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state_prev = state.pipeline_state
        assert pipeline_state_prev is not None
        pipeline_state = self.pipeline_step(pipeline_state_prev, action)

        x_pos_prev = pipeline_state_prev.x.pos[0, 0]
        x_pos = pipeline_state.x.pos[0, 0]
        forward_reward = x_velocity = (x_pos - x_pos_prev) / self.dt

        z, angle = pipeline_state.x.pos[0, 2], pipeline_state.q[2]
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range
        is_healthy = (
                (z > min_z) & (z < max_z) & (angle > min_angle) & (angle < max_angle)
        )
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        obs = self._get_obs(pipeline_state)
        reward = (forward_reward + healthy_reward) * self._reward_scaler
        hazard_positions = state.info["hazard_positions"]
        cost = self._calculate_safety_cost(pipeline_state, hazard_positions)  # safety cost: feet inside any hazards

        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        metrics = dict(state.metrics)
        metrics.update(
            x_position=x_pos,
            x_velocity=x_velocity,
            reward_forward=forward_reward,
            reward_healthy=healthy_reward,
            ctrl_cost=ctrl_cost,
            cost=cost,
        )

        info = dict(state.info)
        info["cost"] = cost

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    # ---------------- Observations ----------------

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        # Mirror base Walker2d observation shaping:
        # - replace position[1] with torso z
        # - optionally exclude rootx by slicing position[1:]
        # - concatenate positions and clipped velocities
        position = pipeline_state.q
        position = position.at[1].set(pipeline_state.x.pos[self._torso_link_id, 2])
        velocity = jp.clip(pipeline_state.qd, -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        return jp.concatenate((position, velocity))

    # ---------------- Safety logic ----------------

    def _calculate_safety_cost(self, pipeline_state: base.State, hazard_positions: jax.Array) -> jax.Array:
        n_h = hazard_positions.shape[0]
        if n_h == 0:
            return jp.array(0.0, dtype=jp.float32)

        # feet world positions (use body COM positions as proxy)
        left_pos = pipeline_state.x.take(self._left_foot_link_id).pos
        right_pos = pipeline_state.x.take(self._right_foot_link_id).pos
        feet_xy = jp.stack([left_pos[:2], right_pos[:2]], axis=0)  # (2,2)

        centers = hazard_positions[:, :2]  # (N,2)

        # distance feet->hazard centers
        d = jp.sqrt(jp.sum((feet_xy[:, None, :] - centers[None, :, :]) ** 2, axis=2) + 1e-8)  # (2,N)

        # cylinder membership: any foot within radius
        cyl_inside = (d <= self._hazard_radii[None, :])  # (2,N)
        cyl_any_per_h = jp.any(cyl_inside, axis=0).astype(jp.float32)  # (N,)

        # rect/cube (axis-aligned in world): |dx|<=hx & |dy|<=hy
        dxdy = jp.abs(feet_xy[:, None, :] - centers[None, :, :])  # (2,N,2)
        rect_inside = jp.logical_and(
            dxdy[:, :, 0] <= self._hazard_half_extents[None, :, 0],
            dxdy[:, :, 1] <= self._hazard_half_extents[None, :, 1],
        )  # (2,N)
        rect_any_per_h = jp.any(rect_inside, axis=0).astype(jp.float32)

        is_rect = self._hazard_is_rect.astype(jp.float32)
        is_cyl = 1.0 - is_rect
        per_hazard_cost = cyl_any_per_h * is_cyl + rect_any_per_h * is_rect  # (N,)
        total_cost = jp.sum(per_hazard_cost)
        return jp.asarray(self._cost_scale, dtype=jp.float32) * total_cost
