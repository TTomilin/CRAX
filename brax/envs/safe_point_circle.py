"""
SafePointCircle Environment

A point agent task that rewards maintaining a circular orbit around a target
radius. Hazards are optional and contribute safety costs.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

import jax
import mujoco
from jax import numpy as jp
from mujoco import mjx

from brax.envs.base import PipelineEnv, State
from brax.envs.env_utils import (
    create_hazard_manager_from_specs,
    generate_goal_xml_from_base,
    safe_norm,
    place_objects,
    base_xml_file_path,
    add_walls_to_specs,
)
from brax.envs.goals import GoalManager
from brax.envs.hazards import _type_defaults_from_registry
from brax.io import mjcf


class SafePointCircle(PipelineEnv):
    """Point agent environment that rewards circular orbiting."""

    def __init__(
            self,
            # Episode settings
            episode_length: int = 1000,
            base_agent_file_name: str = "point_circle.xml",
            # Physics settings
            backend: str = "mjx",
            n_frames: int = 5,
            timestep: float = 0.02,
            terminate_when_unhealthy: bool = True,
            healthy_z_range: Tuple[float, float] = (0.05, 0.5),
            reset_noise_scale: float = 0.1,
            init_xy_range: float = 0.8,
            init_angle_range: float = 3.141592653589793,
            # Circle task settings
            circle_radius: float = 1.5,
            circle_center: Tuple[float, float] = (0.0, 0.0),
            reward_factor: float = 0.1,
            circle_lidar: bool = True,
            orbit_features: bool = False,
            circle_visual: bool = True,
            circle_visual_height: float = 0.01,
            boundary_visual: bool = True,
            boundary_visual_x: Optional[float] = None,
            boundary_visual_y: Optional[float] = None,
            boundary_visual_thickness: float = 0.05,
            boundary_visual_height: float = 0.1,
            # Cost settings
            proximity_cost_scaler: float = 1.0,
            collision_cost: float = 1.0,
            ctrl_cost_weight: float = 0.0,
            boundary_x: Optional[float] = None,
            boundary_y: Optional[float] = None,
            boundary_cost: float = 0.1,
            # Lidar settings
            lidar_bins: int = 16,
            lidar_max_dist: float = 6.0,
            hazard_compass_k: int = 8,
            include_hazard_lidar: bool = True,
            # Placement settings
            placement_extents: Tuple[float, float, float, float] = (-3.0, -3.0, 3.0, 3.0),
            agent_keepout: float = 0.1,
            placement_margin: float = 0.01,
            max_placement_attempts: int = 100,
            max_layout_attempts: int = 1000,
            # Hazard settings
            hazard_specs: Optional[List[Dict]] = None,
            # Debug
            debug: bool = False,
            **kwargs,
    ):
        self._debug = debug

        if hazard_specs is None:
            hazard_specs = []

        # Expand hazard specs with type defaults
        type_defaults = _type_defaults_from_registry()
        expanded_specs = []
        for spec in hazard_specs:
            if not isinstance(spec, dict):
                expanded_specs.append(spec)
                continue
            spec_type = spec.get("type")
            base = dict(type_defaults.get(spec_type, {}))
            base.update(spec)
            expanded_specs.append(base)
        hazard_specs = expanded_specs

        # Add outer walls to hazard specs
        hazard_specs = add_walls_to_specs(hazard_specs, placement_extents)

        # Circle task params (set before XML generation)
        self._circle_radius = float(circle_radius)
        self._circle_center = jp.array(circle_center, dtype=jp.float32)
        self._reward_factor = float(reward_factor)
        self._include_circle_lidar = bool(circle_lidar)
        self._include_orbit_features = bool(orbit_features)
        self._circle_visual = bool(circle_visual)
        self._circle_visual_height = float(circle_visual_height)

        self._boundary_x = None if boundary_x is None else float(boundary_x)
        self._boundary_y = None if boundary_y is None else float(boundary_y)
        self._boundary_cost = float(boundary_cost)
        self._boundary_visual = bool(boundary_visual)
        self._boundary_visual_x = boundary_visual_x
        self._boundary_visual_y = boundary_visual_y
        self._boundary_visual_thickness = float(boundary_visual_thickness)
        self._boundary_visual_height = float(boundary_visual_height)

        # Compute hazard placement extents based on boundaries (if set)
        # Hazards should only spawn within the boundary area
        if self._boundary_x is not None or self._boundary_y is not None:
            haz_x = self._boundary_x if self._boundary_x is not None else placement_extents[2]
            haz_y = self._boundary_y if self._boundary_y is not None else placement_extents[3]
            self._hazard_placement_extents = (-haz_x, -haz_y, haz_x, haz_y)
        else:
            self._hazard_placement_extents = placement_extents

        # Build managers
        self._hazard_manager = create_hazard_manager_from_specs(hazard_specs)
        self._goal_manager = GoalManager()
        self._boundary_goal_ids = []
        # Circle indicator uses a non-collidable goal cylinder
        if self._circle_visual:
            circle_pos = [(
                float(self._circle_center[0]),
                float(self._circle_center[1]),
                self._circle_visual_height,
            )]
            self._goal_manager.add_goals(
                goal_type="cylinder",
                count=1,
                positions=circle_pos,
                size=self._circle_radius,
                height=self._circle_visual_height,
            )

        # Visual boundary walls (non-colliding) for browser viewer
        if self._boundary_visual:
            visual_x = self._boundary_visual_x if self._boundary_visual_x is not None else self._boundary_x
            visual_y = self._boundary_visual_y if self._boundary_visual_y is not None else self._boundary_y
            if visual_x is not None or visual_y is not None:
                zc = self._boundary_visual_height * 0.5
                thickness = self._boundary_visual_thickness

                if visual_x is not None:
                    y_extent = visual_y if visual_y is not None else visual_x
                    start_id = self._goal_manager.get_goal_count() + 1
                    self._goal_manager.add_goals(
                        goal_type="cube",
                        count=2,
                        positions=[(-visual_x, 0.0, zc), (visual_x, 0.0, zc)],
                        size=(thickness, y_extent),
                        height=self._boundary_visual_height,
                    )
                    self._boundary_goal_ids.extend([start_id, start_id + 1])

                if visual_y is not None:
                    x_extent = visual_x if visual_x is not None else visual_y
                    start_id = self._goal_manager.get_goal_count() + 1
                    self._goal_manager.add_goals(
                        goal_type="cube",
                        count=2,
                        positions=[(0.0, -visual_y, zc), (0.0, visual_y, zc)],
                        size=(x_extent, thickness),
                        height=self._boundary_visual_height,
                    )
                    self._boundary_goal_ids.extend([start_id, start_id + 1])

        goals = self._goal_manager.goals
        hazards = self._hazard_manager.hazards

        self._hazard_keepouts = jp.array(
            [h.get_keepout_radius() for h in hazards], dtype=jp.float32
        ) if hazards else jp.zeros((0,), dtype=jp.float32)

        # Generate XML dynamically with goals/hazards
        xml_path = generate_goal_xml_from_base(base_agent_file_name, self._goal_manager, self._hazard_manager)
        if self._boundary_goal_ids:
            with open(xml_path, "r", encoding="utf-8") as f:
                xml_text = f.read()
            for goal_id in self._boundary_goal_ids:
                pattern = rf'(<geom[^>]*name="goal{goal_id}"[^>]*rgba=")[^"]+(")'
                xml_text, _ = re.subn(
                    pattern,
                    lambda m: f"{m.group(1)}1 1 0 0.3{m.group(2)}",
                    xml_text,
                )
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(xml_text)
        self._xml_base_file_path = base_xml_file_path(base_agent_file_name)

        try:
            mj_model = mujoco.MjModel.from_xml_path(xml_path)
            mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
            mj_model.opt.timestep = timestep
            mj_model.opt.iterations = 4
            mj_model.opt.ls_iterations = 4
        finally:
            if os.path.exists(xml_path):
                os.unlink(xml_path)

        def _mocap_id_for_body(name: str) -> int:
            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                raise RuntimeError(f"Missing body named {name}")
            mocap_id = int(mj_model.body_mocapid[body_id])
            if mocap_id < 0:
                raise RuntimeError(f"Body {name} is not mocap (body_mocapid < 0)")
            return mocap_id

        sys = mjcf.load_model(mj_model)
        super().__init__(sys, backend=backend, n_frames=n_frames)

        self.episode_length = episode_length

        # Body IDs
        self._agent_body = 1

        # Hazards (names match XMLBuilder)
        self._hazard_mocap_ids = []
        for hazard in hazards:
            self._hazard_mocap_ids.append(_mocap_id_for_body(f"hazard{hazard.hazard_id}"))

        # Goals (visuals only)
        self._goal_mocap_ids = []
        for goal in goals:
            self._goal_mocap_ids.append(_mocap_id_for_body(f"goal{goal.goal_id}"))
        self._goal_positions = (
            jp.array([g.position for g in goals], dtype=jp.float32)
            if goals else jp.zeros((0, 3), dtype=jp.float32)
        )

        # Cache agent geom IDs for contact checks
        self._agent_geom_ids = jp.array(
            [i for i in range(mj_model.ngeom) if mj_model.geom_bodyid[i] == self._agent_body],
            dtype=jp.int32,
        )

        # Assign geom_id to each hazard by name "hazard{i}"
        for hazard in hazards:
            gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"hazard{hazard.hazard_id}")
            hazard.geom_id = gid

        # Hazard counts
        self._num_hazards = self._hazard_manager.get_hazard_count()
        self._num_fixed_hazards = self._hazard_manager.get_fixed_hazard_count()
        self._num_movable_hazards = self._num_hazards - self._num_fixed_hazards

        # Sensor info
        self._sensor_info = {}
        required_sensors = ["accelerometer", "velocimeter", "gyro", "magnetometer"]
        sensor_found_flags = {name: False for name in required_sensors}
        if mj_model.nsensor > 0:
            for i in range(mj_model.nsensor):
                name = mj_model.sensor(i).name
                if name in required_sensors:
                    start_adr = mj_model.sensor_adr[i]
                    dim = mj_model.sensor_dim[i]
                    self._sensor_info[name] = (start_adr, dim)
                    sensor_found_flags[name] = True
        else:
            print("Warning: Model has no sensors defined (mj_model.nsensor = 0).")

        missing_sensors = [name for name, found in sensor_found_flags.items() if not found]
        if missing_sensors:
            print(f"Warning: Could not find the following required sensors: {missing_sensors}")

        # Costs
        self._proximity_cost_scaler = float(proximity_cost_scaler)
        self._collision_cost = float(collision_cost)
        self._ctrl_cost_weight = float(ctrl_cost_weight)

        # Physics
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._init_xy_range = float(init_xy_range)
        self._init_angle_range = float(init_angle_range)

        # Lidar
        self._lidar_num_bins = lidar_bins
        self._lidar_max_dist = lidar_max_dist
        self._hazard_compass_k = hazard_compass_k
        self._include_hazard_lidar = bool(include_hazard_lidar) and self._num_hazards > 0
        self._include_hazard_compass = self._num_hazards > 0 and self._hazard_compass_k > 0

        # Placement
        self._placement_extents = placement_extents
        self._agent_keepout = agent_keepout
        self._placement_margin = placement_margin
        self._max_placement_attempts = max_placement_attempts
        self._max_layout_attempts = max_layout_attempts

        if self._debug:
            print(f"SafePointCircle initialized with {self._num_hazards} hazards")

    def reset(self, rng: jp.ndarray) -> State:
        """Reset the environment with constrained hazard placement."""
        rng, rng1, rng2, rng_layout, rng_pos, rng_angle = jax.random.split(rng, 6)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng2, (self.sys.nv,), minval=low, maxval=hi
        )

        qpos = jax.lax.cond(
            qpos.shape[0] > 6,
            lambda qp: qp.at[3:7].set(qp[3:7] / (safe_norm(qp[3:7]) + 1e-8)),
            lambda qp: qp,
            qpos,
        )

        # Match Safety-Gymnasium: randomize starting xy and heading
        if qpos.shape[0] >= 2:
            xy = jax.random.uniform(
                rng_pos,
                (2,),
                minval=-self._init_xy_range,
                maxval=self._init_xy_range,
            )
            qpos = qpos.at[0:2].set(xy)
        if qpos.shape[0] >= 3:
            heading = jax.random.uniform(
                rng_angle,
                (),
                minval=-self._init_angle_range,
                maxval=self._init_angle_range,
            )
            qpos = qpos.at[2].set(heading)

        data = self.pipeline_init(qpos, qvel)
        agent_pos = data.xpos[self._agent_body]

        num_candidates = self._max_placement_attempts

        max_entries = 1 + self._num_movable_hazards
        positions_xy = jp.zeros((max_entries, 2))
        keepouts = jp.zeros((max_entries,))

        positions_xy = positions_xy.at[0].set(agent_pos[:2])
        keepouts = keepouts.at[0].set(self._agent_keepout)
        count = jp.array(1, dtype=jp.int32)

        mpos = data.mocap_pos
        if self._goal_positions.shape[0] > 0:
            goal_ids = jp.array(self._goal_mocap_ids, dtype=jp.int32)
            mpos = mpos.at[goal_ids].set(self._goal_positions)

        if self._num_movable_hazards > 0:
            # Simple uniform random placement within hazard extents
            min_x, min_y, max_x, max_y = self._hazard_placement_extents
            rng_layout, rng_x, rng_y = jax.random.split(rng_layout, 3)
            hazard_xs = jax.random.uniform(
                rng_x, (self._num_movable_hazards,), minval=min_x, maxval=max_x
            )
            hazard_ys = jax.random.uniform(
                rng_y, (self._num_movable_hazards,), minval=min_y, maxval=max_y
            )
            hazard_zs = jp.full((self._num_movable_hazards,), 0.09)
            hazard_positions = jp.stack([hazard_xs, hazard_ys, hazard_zs], axis=1)

            hazard_ids = jp.array(self._hazard_mocap_ids[:self._num_movable_hazards], dtype=jp.int32)
            mpos = mpos.at[hazard_ids].set(hazard_positions)

        data = data.replace(mocap_pos=mpos)
        hazard_positions = data.mocap_pos[jp.array(self._hazard_mocap_ids, dtype=jp.int32)]

        info = {
            "hazard_positions": hazard_positions,
            "step_count": 0,
            "cost": 0.0,
            "out_of_boundary": 0.0,
        }

        obs = self._get_obs(data)
        reward, cost, ctrl_cost, done = jp.zeros(4)
        _, radial_error, tangent_vel, _ = self._calculate_orbit_components(data)
        metrics = self._get_metrics(
            data,
            reward,
            cost,
            radial_error,
            tangent_vel,
            ctrl_cost,
            jp.array(0.0),
            jp.array(0.0),
        )

        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Execute one step in the environment."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        agent_pos = data.xpos[self._agent_body]
        hazard_positions = state.info["hazard_positions"]

        orbit_reward, radial_error, tangent_vel, _ = self._calculate_orbit_components(data)

        ctrl_cost = jp.sum(jp.square(action)) * self._ctrl_cost_weight
        hazard_cost = self._calculate_safety_cost(data, hazard_positions)
        boundary_cost, out_of_boundary = self._calculate_boundary_cost(agent_pos)
        cost = hazard_cost + boundary_cost
        reward = orbit_reward

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.logical_and(
            agent_pos[2] >= min_z,
            agent_pos[2] <= max_z
        ).astype(jp.float32)

        done = jp.logical_or(
            (1.0 - is_healthy) * self._terminate_when_unhealthy,
            jp.any(jp.isnan(agent_pos)),
        )

        obs = self._get_obs(data)
        metrics = self._get_metrics(
            data,
            reward,
            cost,
            radial_error,
            tangent_vel,
            ctrl_cost,
            out_of_boundary,
            boundary_cost,
        )

        new_info = dict(state.info)
        new_info.update({
            "step_count": state.info["step_count"] + 1,
            "cost": cost,
            "out_of_boundary": out_of_boundary,
        })

        return State(data, obs, reward, done.astype(jp.float32), metrics, new_info)

    def _calculate_orbit_components(self, data: mjx.Data) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray]:
        """Compute orbit reward, radial error, tangent velocity, and radius."""
        agent_pos = data.xpos[self._agent_body]
        agent_vel = data.qvel

        dx = agent_pos[0] - self._circle_center[0]
        dy = agent_pos[1] - self._circle_center[1]

        radius = jp.sqrt(dx * dx + dy * dy + 1e-8)
        radial_error = jp.abs(radius - self._circle_radius)

        u = agent_vel[0]
        v = agent_vel[1]
        tangent_vel = (-u * dy + v * dx) / radius

        radial_penalty = 1.0 / (1.0 + radial_error)
        reward = tangent_vel * radial_penalty * self._reward_factor

        return reward, radial_error, tangent_vel, radius

    def _calculate_safety_cost(self, data: mjx.Data, hazard_positions: jp.ndarray) -> jp.ndarray:
        """Sum of per-hazard costs. Binary collision for collidables, proximity for others."""
        agent_xy = data.xpos[self._agent_body][:2]

        ids1 = getattr(data.contact, "geom1", None)
        ids2 = getattr(data.contact, "geom2", None)
        dist = getattr(data.contact, "dist", None)
        ncon = getattr(data, "ncon", None)

        total = jp.array(0.0)
        for i, h in enumerate(self._hazard_manager.hazards):
            hz_xy = hazard_positions[i, :2]
            total = total + h.calculate_cost(
                agent_xy=agent_xy,
                hazard_xy=hz_xy,
                proximity_cost_scaler=self._proximity_cost_scaler,
                collision_cost=self._collision_cost,
                contact_geom1=ids1,
                contact_geom2=ids2,
                contact_dist=dist,
                ncon=ncon,
                agent_geom_ids=self._agent_geom_ids,
            )
        return total

    def _calculate_boundary_cost(self, agent_pos: jp.ndarray) -> Tuple[jp.ndarray, jp.ndarray]:
        """Cost for crossing boundary limits (sigwalls behavior)."""
        if self._boundary_x is None and self._boundary_y is None:
            return jp.array(0.0), jp.array(0.0)

        rel_x = agent_pos[0] - self._circle_center[0]
        rel_y = agent_pos[1] - self._circle_center[1]

        violation = jp.array(False)
        if self._boundary_x is not None:
            violation = jp.logical_or(violation, jp.abs(rel_x) > self._boundary_x)
        if self._boundary_y is not None:
            violation = jp.logical_or(violation, jp.abs(rel_y) > self._boundary_y)

        violation = violation.astype(jp.float32)
        cost = violation * self._boundary_cost
        return cost, violation

    def _get_obs(self, data: mjx.Data) -> jp.ndarray:
        """Creates an observation with hazard lidar and circle-orbit cues."""
        agent_pos = data.xpos[self._agent_body]

        # 1. Agent sensor observations
        sensor_data = data.sensordata
        default_val = jp.zeros(3, dtype=sensor_data.dtype)

        accel_adr, accel_dim = self._sensor_info.get("accelerometer", (0, 0))
        accelerometer = jax.lax.dynamic_slice(sensor_data, (accel_adr,), (accel_dim,))
        accelerometer = jp.where(accel_dim == 3, accelerometer, default_val)

        velo_adr, velo_dim = self._sensor_info.get("velocimeter", (0, 0))
        velocimeter = jax.lax.dynamic_slice(sensor_data, (velo_adr,), (velo_dim,))
        velocimeter = jp.where(velo_dim == 3, velocimeter, default_val)

        gyro_adr, gyro_dim = self._sensor_info.get("gyro", (0, 0))
        gyro = jax.lax.dynamic_slice(sensor_data, (gyro_adr,), (gyro_dim,))
        gyro = jp.where(gyro_dim == 3, gyro, default_val)

        mag_adr, mag_dim = self._sensor_info.get("magnetometer", (0, 0))
        magnetometer = jax.lax.dynamic_slice(sensor_data, (mag_adr,), (mag_dim,))
        magnetometer = jp.where(mag_dim == 3, magnetometer, default_val)

        # 2. Circle-related features in agent-centric frame
        world_dx = agent_pos[0] - self._circle_center[0]
        world_dy = agent_pos[1] - self._circle_center[1]

        agent_z_angle = data.qpos[2]
        cos_a = jp.cos(agent_z_angle)
        sin_a = jp.sin(agent_z_angle)

        agent_centric_dx = world_dx * cos_a + world_dy * sin_a
        agent_centric_dy = -world_dx * sin_a + world_dy * cos_a

        radius = jp.sqrt(world_dx * world_dx + world_dy * world_dy + 1e-8)
        radial_error = jp.abs(radius - self._circle_radius)

        tangent_vec = jp.array([-agent_centric_dy, agent_centric_dx])
        tangent_dir = tangent_vec / (safe_norm(tangent_vec) + 1e-8)

        # 3. Circle lidar
        _lidar_num_bins = self._lidar_num_bins
        _lidar_max_dist = self._lidar_max_dist
        bin_size = (2 * jp.pi) / _lidar_num_bins

        circle_lidar_obs = jp.zeros(_lidar_num_bins)
        if self._include_circle_lidar:
            angles = jp.linspace(0.0, 2 * jp.pi, _lidar_num_bins, endpoint=False)
            dirs = jp.stack([jp.cos(angles), jp.sin(angles)], axis=1)  # (B, 2)
            center_rel = jp.array([-agent_centric_dx, -agent_centric_dy])  # center relative to agent

            dot = dirs @ center_rel
            center_norm_sq = jp.sum(center_rel * center_rel)
            disc = dot * dot - (center_norm_sq - self._circle_radius * self._circle_radius)
            disc = jp.maximum(disc, 0.0)
            sqrt_disc = jp.sqrt(disc)
            t1 = dot - sqrt_disc
            t2 = dot + sqrt_disc

            t = jp.where(t1 > 0.0, t1, jp.where(t2 > 0.0, t2, jp.inf))
            valid = disc >= 0.0
            t = jp.where(valid, t, jp.inf)

            circle_lidar_obs = jp.maximum(0.0, _lidar_max_dist - t) / _lidar_max_dist
            circle_lidar_obs = jp.where(t <= _lidar_max_dist, circle_lidar_obs, 0.0)

        # 4. Hazard lidar
        hazard_lidar_obs = jp.zeros(_lidar_num_bins)
        if self._include_hazard_lidar:
            _lidar_alias = True

            def process_hazard_lidar(carry, hazard_mocap_id):
                hazard_lidar, agent_pos, cos_a, sin_a = carry

                hazard_pos_3d = jp.where(
                    hazard_mocap_id >= 0,
                    data.mocap_pos[hazard_mocap_id],
                    jp.array([0.0, 0.0, 0.0]),
                )

                rel_hazard_pos_3d_world = hazard_pos_3d - agent_pos

                world_dx_hazard = rel_hazard_pos_3d_world[0]
                world_dy_hazard = rel_hazard_pos_3d_world[1]

                agent_centric_dx_hazard = world_dx_hazard * cos_a + world_dy_hazard * sin_a
                agent_centric_dy_hazard = -world_dx_hazard * sin_a + world_dy_hazard * cos_a

                dist_hazard = safe_norm(jp.array([agent_centric_dx_hazard, agent_centric_dy_hazard]))
                angle_hazard = jp.arctan2(agent_centric_dy_hazard, agent_centric_dx_hazard)
                angle_hazard = (angle_hazard + 2 * jp.pi) % (2 * jp.pi)

                bin_idx_float_hazard = angle_hazard / bin_size
                bin_idx_hazard = jp.floor(bin_idx_float_hazard)
                bin_idx_hazard = jp.minimum(bin_idx_hazard, _lidar_num_bins - 1).astype(int)

                sensor_val_hazard = jp.maximum(0.0, _lidar_max_dist - dist_hazard) / _lidar_max_dist
                sensor_val_hazard = jp.where(dist_hazard > _lidar_max_dist, 0.0, sensor_val_hazard)
                sensor_val_hazard = jp.where(hazard_mocap_id >= 0, sensor_val_hazard, 0.0)

                hazard_lidar = hazard_lidar.at[bin_idx_hazard].set(
                    jp.maximum(hazard_lidar[bin_idx_hazard], sensor_val_hazard)
                )

                if _lidar_alias:
                    alias_factor_hazard = bin_idx_float_hazard - bin_idx_hazard

                    bin_plus_idx = (bin_idx_hazard + 1) % _lidar_num_bins
                    hazard_lidar = hazard_lidar.at[bin_plus_idx].set(
                        jp.maximum(hazard_lidar[bin_plus_idx], alias_factor_hazard * sensor_val_hazard)
                    )

                    bin_minus_idx = (bin_idx_hazard - 1 + _lidar_num_bins) % _lidar_num_bins
                    hazard_lidar = hazard_lidar.at[bin_minus_idx].set(
                        jp.maximum(hazard_lidar[bin_minus_idx], (1.0 - alias_factor_hazard) * sensor_val_hazard)
                    )

                return (hazard_lidar, agent_pos, cos_a, sin_a), None

            hazard_mocap_ids_array = jp.array(self._hazard_mocap_ids, dtype=jp.int32)
            init_carry = (hazard_lidar_obs, agent_pos, cos_a, sin_a)
            (hazard_lidar_obs, _, _, _), _ = jax.lax.scan(
                process_hazard_lidar, init_carry, hazard_mocap_ids_array
            )

        # 5. Hazard compasses (closest-k)
        hazard_compasses_flat = jp.zeros((0,))
        if self._include_hazard_compass:
            def compute_compass_for_hazard(mocap_idx):
                hazard_pos_3d = jp.where(
                    mocap_idx >= 0,
                    data.mocap_pos[mocap_idx],
                    jp.array([0.0, 0.0, 0.0]),
                )

                rel_hazard_pos_3d_world = hazard_pos_3d - agent_pos
                world_dx_hazard = rel_hazard_pos_3d_world[0]
                world_dy_hazard = rel_hazard_pos_3d_world[1]

                agent_centric_dx_hazard = world_dx_hazard * cos_a + world_dy_hazard * sin_a
                agent_centric_dy_hazard = -world_dx_hazard * sin_a + world_dy_hazard * cos_a

                rel_vec = jp.array([agent_centric_dx_hazard, agent_centric_dy_hazard])
                compass = rel_vec / (safe_norm(rel_vec) + 1e-8)
                return jp.where(mocap_idx >= 0, compass, jp.zeros(2))

            all_hz_ids = jp.array(self._hazard_mocap_ids, dtype=jp.int32)
            hazard_count = all_hz_ids.shape[0]

            k = int(self._hazard_compass_k)
            k_eff = min(k, hazard_count)

            def _pick_and_pad():
                all_hz_pos = data.mocap_pos[all_hz_ids]
                rel_xy = all_hz_pos[:, :2] - agent_pos[:2]
                d2 = jp.sum(rel_xy * rel_xy, axis=1)
                order = jp.argsort(d2)

                closest = all_hz_ids[order[:k_eff]]

                if k_eff < k:
                    pad = -jp.ones((k - k_eff,), dtype=jp.int32)
                    closest = jp.concatenate([closest, pad], axis=0)
                return closest

            def _no_hazards():
                return -jp.ones((k,), dtype=jp.int32)

            closest_ids = jax.lax.cond(
                hazard_count > 0, lambda _: _pick_and_pad(), lambda _: _no_hazards(), operand=None
            )

            hazard_compasses = jax.vmap(compute_compass_for_hazard)(closest_ids)
            hazard_compasses_flat = hazard_compasses.reshape((-1,))

        obs_parts = [
            accelerometer,
            velocimeter,
            gyro,
            magnetometer,
        ]
        if self._include_orbit_features:
            obs_parts.extend([jp.array([radial_error]), tangent_dir])
        if self._include_circle_lidar:
            obs_parts.append(circle_lidar_obs)
        if self._include_hazard_lidar:
            obs_parts.append(hazard_lidar_obs)
        if self._include_hazard_compass:
            obs_parts.append(hazard_compasses_flat)

        obs = jp.concatenate(obs_parts)

        return obs

    def _get_metrics(self,
                     data: mjx.Data,
                     reward: jp.ndarray,
                     cost: jp.ndarray,
                     radial_error: jp.ndarray,
                     tangent_vel: jp.ndarray,
                     ctrl_cost: jp.ndarray,
                     out_of_boundary: jp.ndarray,
                     boundary_cost: jp.ndarray) -> Dict:
        agent_pos = data.xpos[self._agent_body]
        dx = agent_pos[0] - self._circle_center[0]
        dy = agent_pos[1] - self._circle_center[1]
        radius = jp.sqrt(dx * dx + dy * dy + 1e-8)

        return {
            "reward": reward,
            "cost": cost,
            "x_position": agent_pos[0],
            "y_position": agent_pos[1],
            "radius": radius,
            "radial_error": radial_error,
            "tangent_velocity": tangent_vel,
            "ctrl_cost": ctrl_cost,
            "out_of_boundary": out_of_boundary,
            "boundary_cost": boundary_cost,
        }

    @property
    def observation_size(self) -> int:
        size = 12  # sensor data
        if self._include_orbit_features:
            size += 3
        if self._include_circle_lidar:
            size += self._lidar_num_bins
        if self._include_hazard_lidar:
            size += self._lidar_num_bins
        if self._include_hazard_compass:
            size += self._hazard_compass_k * 2
        return size
