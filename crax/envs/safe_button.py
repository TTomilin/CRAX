"""
SafeButton Environment - Modular Button Task

A navigation environment with configurable agents where the agent must press
the correct button among multiple buttons.

Base class provides task logic; agent-specific classes provide agent configuration.

Usage:
    env = SafeButtonPoint()  # Point agent
    # or via registry:
    env = crax.envs.get_environment("safe_button_point")
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import jax
import mujoco
from jax import numpy as jp
from mujoco import mjx

from crax.envs.base import PipelineEnv, State
from crax.envs.env_utils import (
    create_hazard_manager_from_specs,
    safe_norm,
    place_objects,
    base_xml_file_path,
    add_walls_to_specs,
    compute_gremlin_positions,
)
from crax.envs.hazards import _type_defaults_from_registry, compute_hazard_costs
from crax.envs.builder import XMLBuilder
from crax.io import mjcf


class SafeButton(PipelineEnv, ABC):
    """
    Abstract Safe Button Environment

    A navigation environment where the agent must press the correct button:
    - Multiple fixed buttons (default: 4)
    - One active goal button selected randomly
    - Dense reward for moving closer to goal button
    - Sparse reward when pressing the correct button
    - Wrong-button cost when buttons_constrained=True
    - Button timer/resampling delay (buttons hidden in lidar initially)
    - Support for hazards and gremlins

    Subclasses must implement agent-specific configuration.
    """

    @property
    @abstractmethod
    def agent_xml_file(self) -> str:
        """Return the XML file name for this agent."""
        pass

    @property
    @abstractmethod
    def agent_body_index(self) -> int:
        """Return the body index for this agent in the MuJoCo model."""
        pass

    @property
    @abstractmethod
    def default_healthy_z_range(self) -> Tuple[float, float]:
        """Return the default healthy z range for this agent."""
        pass

    @property
    @abstractmethod
    def default_agent_keepout(self) -> float:
        """Return the default keepout radius for this agent."""
        pass

    @property
    @abstractmethod
    def required_sensors(self) -> List[str]:
        """Return the list of required sensor names for this agent."""
        pass

    def get_agent_heading(self, data: mjx.Data) -> jp.ndarray:
        """Get the agent's current heading angle. Override for different agents."""
        # Default: use qpos[2] as z-rotation (suitable for point agents)
        return data.qpos[2]

    def __init__(
            self,
            # Episode settings
            episode_length: int = 1000,
            # Physics settings
            backend: str = 'mjx',
            n_frames: int = 4,
            timestep: float = 0.02,
            terminate_when_unhealthy: bool = True,
            healthy_z_range: Optional[Tuple[float, float]] = None,
            reset_noise_scale: float = 0.005,
            max_velocity: float = 5.0,
            # Reward settings
            reward_goal: float = 1.0,
            reward_distance_scale: float = 1.0,
            # Cost settings
            cost_scale: float = 2.0,
            proximity_cost_scaler: Optional[float] = None,
            collision_cost: float = 3.0,
            wrong_button_cost: float = 1.0,
            ctrl_cost_weight: float = 0.001,
            # Button settings
            button_count: int = 4,
            button_size: float = 0.1,
            button_height: float = 0.1,
            buttons_constrained: bool = False,
            resampling_delay: int = 10,
            continue_goal: bool = False,
            # Lidar settings
            lidar_bins: int = 16,
            lidar_max_dist: float = 3.0,
            lidar_alias: bool = True,
            hazard_compass_k: int = 8,
            # Placement settings
            placement_extents: Tuple[float, float, float, float] = (-1.0, -1.0, 1.0, 1.0),
            agent_keepout: Optional[float] = None,
            placement_margin: float = 0.01,
            max_placement_attempts: int = 100,
            max_layout_attempts: int = 1000,
            # Hazard settings - list of specs: {type, count, size, height, collidable, fixed, density, travel (for gremlins)}
            hazard_specs: Optional[List[Dict]] = None,
            # Debug
            debug: bool = False,
            **kwargs,
    ):
        self._debug = debug

        # Use agent-specific defaults if not provided
        if healthy_z_range is None:
            healthy_z_range = self.default_healthy_z_range
        if agent_keepout is None:
            agent_keepout = self.default_agent_keepout

        # Build default hazard specs if none provided
        if hazard_specs is None:
            hazard_specs = []

        # Expand hazard specs with type defaults
        type_defaults = _type_defaults_from_registry()
        expanded_specs = []
        for spec in hazard_specs:
            if not isinstance(spec, dict):
                expanded_specs.append(spec)
                continue
            t = spec.get("type")
            base = dict(type_defaults.get(t, {}))
            base.update(spec)
            expanded_specs.append(base)
        hazard_specs = expanded_specs

        # Add outer walls to hazard specs
        hazard_specs = add_walls_to_specs(hazard_specs, placement_extents)

        # MJX does not support cylinder-box collisions; convert cylinders to cubes
        if backend == "mjx":
            converted_specs = []
            for spec in hazard_specs:
                if isinstance(spec, dict) and spec.get("type") == "cylinder":
                    spec = dict(spec)
                    spec["type"] = "cube"
                converted_specs.append(spec)
            hazard_specs = converted_specs

        # Build hazard manager
        self._hazard_manager = create_hazard_manager_from_specs(hazard_specs)
        hazards = self._hazard_manager.hazards

        # Button settings
        self._button_count = button_count
        self._button_size = button_size
        self._button_height = button_height
        self._buttons_constrained = buttons_constrained
        self._resampling_delay = resampling_delay
        self._continue_goal = continue_goal
        self._button_keepout = button_size + 0.1  # Keepout radius for buttons
        self._marker_size = max(0.03, self._button_size * 0.5)
        self._marker_z = self._button_size + self._marker_size

        # Per-hazard keepouts
        self._hazard_keepouts = jp.array(
            [h.get_keepout_radius() for h in hazards], dtype=jp.float32
        ) if len(hazards) > 0 else jp.zeros((0,), dtype=jp.float32)

        # Hazard shape info
        is_rect = []
        half_ext = []
        radii = []

        for h in hazards:
            shape, param = h.get_keepout_shape()
            if shape == "rect":
                is_rect.append(True)
                half_ext.append(jp.array([float(param[0]), float(param[1])]))
                radii.append(0.0)
            else:  # "circle"
                is_rect.append(False)
                half_ext.append(jp.array([0.0, 0.0]))
                radii.append(float(param[0]))

        self._hazard_is_rect = jp.array(is_rect, dtype=jp.bool_) if len(hazards) > 0 else jp.zeros((0,), dtype=jp.bool_)
        self._hazard_half_extents = jp.stack(half_ext) if len(hazards) > 0 else jp.zeros((0, 2))
        self._hazard_radii = jp.array(radii) if len(hazards) > 0 else jp.zeros((0,))

        # Generate XML with buttons and hazards
        xml_path = self._generate_button_xml()
        self._xml_base_file_path = base_xml_file_path(self.agent_xml_file)

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
            b = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if b < 0:
                raise RuntimeError(f"Missing body named {name}")
            mid = int(mj_model.body_mocapid[b])
            if mid < 0:
                raise RuntimeError(f"Body {name} is not mocap (body_mocapid < 0)")
            return mid

        sys = mjcf.load_model(mj_model)

        super().__init__(sys, backend=backend, n_frames=n_frames)

        self.episode_length = episode_length
        self._agent_body = self.agent_body_index

        # Button mocap IDs
        self._button_mocap_ids = []
        for i in range(self._button_count):
            self._button_mocap_ids.append(_mocap_id_for_body(f"button{i}"))
        self._goal_marker_mocap_id = _mocap_id_for_body("goal_marker")

        # Hazard mocap IDs
        self._hazard_mocap_ids = []
        for hazard in hazards:
            self._hazard_mocap_ids.append(_mocap_id_for_body(f"hazard{hazard.hazard_id}"))
        self._hazard_mocap_ids_arr = jp.array(self._hazard_mocap_ids, dtype=jp.int32)

        # Cache agent and button geom ids for contact checks
        self._agent_geom_ids = jp.array(
            [i for i in range(mj_model.ngeom) if mj_model.geom_bodyid[i] == self._agent_body],
            dtype=jp.int32
        )

        # Button geom IDs for contact detection
        self._button_geom_ids = []
        for i in range(self._button_count):
            gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"button{i}")
            if gid < 0:
                raise RuntimeError(f"Missing geom named button{i}")
            self._button_geom_ids.append(gid)
        self._button_geom_ids = jp.array(self._button_geom_ids, dtype=jp.int32)

        # Assign geom_id to each hazard
        for hazard in hazards:
            gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"hazard{hazard.hazard_id}")
            hazard.geom_id = gid

        # Hazard info
        self._num_hazards = self._hazard_manager.get_hazard_count()
        self._num_fixed_hazards = self._hazard_manager.get_fixed_hazard_count()
        self._num_movable_hazards = self._num_hazards - self._num_fixed_hazards

        # Gremlin info
        gremlin_hazards = self._hazard_manager.get_hazards_by_type("gremlin")
        self._num_gremlins = len(gremlin_hazards)
        self._gremlin_indices = []
        self._gremlin_travel_radii = []
        for i, hazard in enumerate(self._hazard_manager.hazards):
            if hazard.hazard_type == "gremlin":
                self._gremlin_indices.append(i)
                self._gremlin_travel_radii.append(hazard.travel)
        self._gremlin_indices = jp.array(self._gremlin_indices, dtype=jp.int32) if self._num_gremlins > 0 else jp.array([], dtype=jp.int32)
        self._gremlin_travel_radii = jp.array(self._gremlin_travel_radii, dtype=jp.float32) if self._num_gremlins > 0 else jp.zeros((0,), dtype=jp.float32)

        # Sensor info
        self._sensor_info = {}
        sensor_found_flags = {name: False for name in self.required_sensors}
        if mj_model.nsensor > 0:
            if self._debug:
                print(f"Model has {mj_model.nsensor} sensors. Searching for required sensors...")
            for i in range(mj_model.nsensor):
                name = mj_model.sensor(i).name
                if name in self.required_sensors:
                    start_adr = mj_model.sensor_adr[i]
                    dim = mj_model.sensor_dim[i]
                    self._sensor_info[name] = (start_adr, dim)
                    sensor_found_flags[name] = True
                    if self._debug:
                        print(f"  Found sensor: {name}, ID: {i}, Address: {start_adr}, Dim: {dim}")
        else:
            print("Warning: Model has no sensors defined (mj_model.nsensor = 0).")

        missing_sensors = [name for name, found in sensor_found_flags.items() if not found]
        if missing_sensors:
            print(f"Warning: Could not find the following required sensors: {missing_sensors}")

        # Allow alias for cost scale used elsewhere in the codebase
        if proximity_cost_scaler is not None:
            cost_scale = proximity_cost_scaler

        # Reward
        self._reward_goal = reward_goal
        self._reward_distance = reward_distance_scale

        # Cost
        self._ctrl_cost_weight = ctrl_cost_weight
        self._proximity_cost_scaler = cost_scale
        self._collision_cost = collision_cost
        self._wrong_button_cost = wrong_button_cost

        # Physics
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._max_velocity = max_velocity

        # Lidar
        self._lidar_num_bins = lidar_bins
        self._lidar_max_dist = lidar_max_dist
        self._lidar_alias = lidar_alias
        self._hazard_compass_k = hazard_compass_k

        # Placement
        self._placement_extents = placement_extents
        self._agent_keepout = agent_keepout
        self._placement_margin = placement_margin
        self._max_placement_attempts = max_placement_attempts
        self._max_layout_attempts = max_layout_attempts

        if self._debug:
            print(f"SafeButton initialized with {self._button_count} buttons, {self._num_hazards} hazards ({self._num_gremlins} gremlins)")

    def _generate_button_xml(self) -> str:
        """Generate XML with buttons and hazards."""
        base_xml_path = base_xml_file_path(self.agent_xml_file)

        # Generate button bodies XML (sphere to avoid cylinder-box collisions in MJX)
        button_bodies = []
        for i in range(self._button_count):
            z = self._button_size
            button_bodies.append(f"""
        <body name="button{i}" pos="0 0 {z}" mocap="true">
            <geom type="sphere" name="button{i}" size="{self._button_size}"
                rgba="0.8 0.8 0.2 0.8" contype="1" conaffinity="1" solref="0.01 1"/>
        </body>""")

        # Visual marker for active button
        button_bodies.append(f"""
        <body name="goal_marker" pos="0 0 {self._marker_z}" mocap="true">
            <geom type="sphere" name="goal_marker" size="{self._marker_size}"
                rgba="0.1 0.9 0.1 0.9" contype="0" conaffinity="0" solref="0.01 1"/>
        </body>""")

        # Use XMLBuilder to combine base agent, buttons, and hazards
        builder = XMLBuilder("safe_button")
        xml_content = builder.build_xml(
            base_xml_path,
            goal_manager=None,  # No goals, we use buttons
            hazard_manager=self._hazard_manager,
            additional_bodies=button_bodies
        )

        # Create temporary file
        import tempfile
        temp_fd, temp_path = tempfile.mkstemp(suffix=".xml", prefix="safe_button_")
        try:
            with os.fdopen(temp_fd, "w") as f:
                f.write(xml_content)
        except:
            os.close(temp_fd)
            raise

        return temp_path

    def reset(self, rng: jp.ndarray) -> State:
        """Reset the environment with button and hazard placement."""
        rng, rng1, rng2, rng_layout, rng_button = jax.random.split(rng, 5)

        # Randomize initial position
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng2, (self.sys.nv,), minval=low, maxval=hi
        )

        # Ensure valid quaternion
        qpos = jax.lax.cond(
            qpos.shape[0] > 6,
            lambda qp: qp.at[3:7].set(qp[3:7] / (safe_norm(qp[3:7]) + 1e-8)),
            lambda qp: qp,
            qpos,
        )

        data = self.pipeline_init(qpos, qvel)
        agent_pos = data.xpos[self._agent_body]

        # Build layout: agent, buttons, then hazards
        num_candidates = self._max_placement_attempts
        max_entries = 1 + self._button_count + self._num_movable_hazards
        positions_xy = jp.zeros((max_entries, 2))
        keepouts = jp.zeros((max_entries,))

        # Seed with agent
        positions_xy = positions_xy.at[0].set(agent_pos[:2])
        keepouts = keepouts.at[0].set(self._agent_keepout)
        count = jp.array(1, dtype=jp.int32)

        # Place buttons
        button_keepouts = jp.full((self._button_count,), self._button_keepout, dtype=jp.float32)
        (rng_layout, positions_xy, keepouts, count, button_positions) = place_objects(
            rng_key=rng_layout,
            positions_xy=positions_xy,
            keepouts_array=keepouts,
            placed_count=count,
            per_item_keepouts=button_keepouts,
            num_items=self._button_count,
            num_candidates=num_candidates,
            placement_extents=self._placement_extents,
            placement_margin=self._placement_margin,
        )

        # Place hazards
        if self._num_movable_hazards > 0:
            (rng_layout, positions_xy, keepouts, count, hazard_positions) = place_objects(
                rng_key=rng_layout,
                positions_xy=positions_xy,
                keepouts_array=keepouts,
                placed_count=count,
                per_item_keepouts=self._hazard_keepouts[:self._num_movable_hazards],
                num_items=self._num_movable_hazards,
                num_candidates=num_candidates,
                placement_extents=self._placement_extents,
                placement_margin=self._placement_margin,
            )
        else:
            hazard_positions = jp.zeros((0, 3))

        # Set button and hazard positions in mocap
        button_ids = jp.array(self._button_mocap_ids, dtype=jp.int32)
        mpos = data.mocap_pos
        mpos = mpos.at[button_ids].set(button_positions)

        if self._num_movable_hazards > 0:
            hazard_ids = self._hazard_mocap_ids_arr[:self._num_movable_hazards]
            mpos = mpos.at[hazard_ids].set(hazard_positions)

        data = data.replace(mocap_pos=mpos)

        # Get all hazard positions (including fixed ones)
        all_hazard_positions = data.mocap_pos[self._hazard_mocap_ids_arr]

        # Select active button randomly
        active_button_idx = jax.random.randint(rng_button, (), 0, self._button_count)

        # Calculate initial distance to active button
        agent_xy = agent_pos[:2]
        active_button_xy = button_positions[active_button_idx, :2]
        initial_dist_goal = jp.sqrt(jp.sum(jp.square(active_button_xy - agent_xy)) + 1e-8)

        # Store gremlin center positions (for motion)
        gremlin_center_positions = all_hazard_positions[self._gremlin_indices] if self._num_gremlins > 0 else jp.zeros((0, 3))

        # Update goal marker position
        marker_pos = button_positions[active_button_idx]
        marker_pos = marker_pos.at[2].set(self._marker_z)
        mpos = data.mocap_pos
        mpos = mpos.at[self._goal_marker_mocap_id].set(marker_pos)
        data = data.replace(mocap_pos=mpos)

        info = {
            "button_positions": button_positions,
            "hazard_positions": all_hazard_positions,
            "gremlin_center_positions": gremlin_center_positions,
            "active_button_idx": active_button_idx,
            "step_count": 0,
            "last_dist_goal": initial_dist_goal,
            "button_timer": 0,
            "cost": 0.0,
            "respawn_rng": rng_layout,
            "done_goal": jp.array(False),
            "done_nan": jp.array(False),
            "done_unhealthy": jp.array(False),
        }

        obs = self._get_obs(data, info)
        reward, cost, ctrl_cost, done = jp.zeros(4)
        metrics = self._get_metrics(data, reward, cost, initial_dist_goal, initial_dist_goal, ctrl_cost, info)

        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Execute one step in the environment."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        # Get positions
        agent_pos = data.xpos[self._agent_body]
        button_positions = state.info['button_positions']
        hazard_positions = state.info['hazard_positions']
        gremlin_center_positions = state.info['gremlin_center_positions']
        active_button_idx = state.info['active_button_idx']
        last_dist_goal = state.info['last_dist_goal']
        button_timer = state.info['button_timer']
        step_count = state.info['step_count']

        # Update gremlin positions if any exist
        if self._num_gremlins > 0:
            gremlin_positions = compute_gremlin_positions(
                gremlin_center_positions,
                self._gremlin_travel_radii,
                step_count,
                self.dt
            )
            gremlin_mocap_ids = self._hazard_mocap_ids_arr[self._gremlin_indices]
            mpos = data.mocap_pos
            mpos = mpos.at[gremlin_mocap_ids].set(gremlin_positions)
            data = data.replace(mocap_pos=mpos)
            hazard_positions = hazard_positions.at[self._gremlin_indices].set(gremlin_positions)

        # Check button contacts (JAX-compatible)
        ids1 = getattr(data.contact, "geom1", None)
        ids2 = getattr(data.contact, "geom2", None)
        contact_dist = getattr(data.contact, "dist", None)
        ncon = getattr(data, "ncon", None)

        goal_achieved = jp.array(False)
        wrong_button_pressed = jp.array(False)

        if ids1 is not None and ids2 is not None and ncon is not None and contact_dist is not None:
            # Create valid mask for contacts
            max_contacts = ids1.shape[0]
            valid_mask = jp.arange(max_contacts) < ncon
            touch = contact_dist <= 0.0

            # Vectorized check: for each button, check if it contacts agent
            def check_button_contact(i):
                button_geom_id = self._button_geom_ids[i]
                # Check if button geom contacts any agent geom
                button_in_geom1 = (ids1 == button_geom_id) & valid_mask & touch
                button_in_geom2 = (ids2 == button_geom_id) & valid_mask & touch
                agent_in_geom1 = jp.any(ids1[:, None] == self._agent_geom_ids[None, :], axis=1) & valid_mask & touch
                agent_in_geom2 = jp.any(ids2[:, None] == self._agent_geom_ids[None, :], axis=1) & valid_mask & touch

                contact1 = jp.any(button_in_geom1 & agent_in_geom2)
                contact2 = jp.any(button_in_geom2 & agent_in_geom1)
                return contact1 | contact2

            # Check all buttons
            button_contacts = jax.vmap(check_button_contact)(jp.arange(self._button_count))

            # Goal achieved if active button is contacted
            goal_achieved = button_contacts[active_button_idx]

            # Wrong button pressed if any non-active button is contacted
            def check_wrong():
                non_active_mask = jp.arange(self._button_count) != active_button_idx
                return jp.any(button_contacts & non_active_mask)

            def no_wrong():
                return jp.array(False)

            wrong_button_pressed = jax.lax.cond(
                self._buttons_constrained,
                check_wrong,
                no_wrong
            )

        # Calculate distance to active button
        agent_xy = agent_pos[:2]
        active_button_xy = button_positions[active_button_idx, :2]
        dist_goal = jp.sqrt(jp.sum(jp.square(active_button_xy - agent_xy)) + 1e-8)

        # Rewards
        dist_reward = (last_dist_goal - dist_goal) * self._reward_distance
        goal_reward = self._reward_goal * goal_achieved.astype(jp.float32)

        # Costs
        ctrl_cost = jp.sum(jp.square(action)) * self._ctrl_cost_weight
        hazard_cost = self._calculate_safety_cost(data, hazard_positions)
        wrong_button_cost = self._wrong_button_cost * wrong_button_pressed.astype(jp.float32) * self._buttons_constrained
        cost = hazard_cost + wrong_button_cost

        # Handle goal achievement and optional resampling
        rng_for_respawn = state.info["respawn_rng"]
        resample_pred = jp.logical_and(goal_achieved, jp.array(self._continue_goal))

        def _resample(rng_in):
            rng_out, rng_button = jax.random.split(rng_in)
            new_idx = jax.random.randint(rng_button, (), 0, self._button_count)
            return rng_out, new_idx

        def _keep(rng_in):
            return rng_in, active_button_idx

        rng_for_respawn, new_active_button_idx = jax.lax.cond(
            resample_pred,
            _resample,
            _keep,
            rng_for_respawn
        )

        new_button_timer = jax.lax.cond(
            resample_pred,
            lambda _: jp.array(self._resampling_delay, dtype=jp.int32),
            lambda _: jp.maximum(0, button_timer - 1),
            operand=None
        )

        # Reset last_dist_goal when resampling
        new_active_button_xy = button_positions[new_active_button_idx, :2]
        new_dist_goal = safe_norm(new_active_button_xy - agent_xy)
        new_last_dist_goal = jp.where(resample_pred, new_dist_goal, dist_goal)

        # Health check
        min_z, max_z = self._healthy_z_range
        is_healthy = jp.logical_and(
            agent_pos[2] >= min_z,
            agent_pos[2] <= max_z
        )

        # Termination
        done_goal = jp.logical_and(goal_achieved, jp.logical_not(jp.array(self._continue_goal)))
        done_nan = jp.any(jp.isnan(agent_pos))
        done_unhealthy = jp.logical_and(
            jp.logical_not(is_healthy),
            jp.array(self._terminate_when_unhealthy)
        )
        done = jp.logical_or(
            jp.logical_or(
                done_unhealthy,
                done_nan
            ),
            done_goal
        )

        # Total reward
        reward = dist_reward + goal_reward

        # Update info
        new_info = state.info.copy()
        new_info.update({
            "button_positions": button_positions,
            "hazard_positions": hazard_positions,
            "gremlin_center_positions": gremlin_center_positions,
            "active_button_idx": new_active_button_idx,
            "step_count": step_count + 1,
            "last_dist_goal": new_last_dist_goal,
            "button_timer": new_button_timer,
            "cost": cost,
            "respawn_rng": rng_for_respawn,
            "done_goal": done_goal,
            "done_nan": done_nan,
            "done_unhealthy": done_unhealthy,
        })

        # Update goal marker to active button
        marker_pos = button_positions[new_active_button_idx]
        marker_pos = marker_pos.at[2].set(self._marker_z)
        mpos = data.mocap_pos
        mpos = mpos.at[self._goal_marker_mocap_id].set(marker_pos)
        data = data.replace(mocap_pos=mpos)

        # Get observation and metrics
        obs = self._get_obs(data, new_info)
        metrics = self._get_metrics(data, reward, cost, dist_goal, last_dist_goal, ctrl_cost, new_info)

        return State(data, obs, reward, done.astype(jp.float32), metrics, new_info)

    def _calculate_safety_cost(self, data: mjx.Data, hazard_positions: jp.ndarray) -> jp.ndarray:
        """Calculate safety cost from hazards (proximity-only).

        MJX does not correctly update geom_xpos for mocap box bodies, so
        contact distances are stale. We use proximity cost instead, computed
        directly from hazard_positions maintained in mocap_pos.
        """
        return compute_hazard_costs(
            hazards=self._hazard_manager.hazards,
            hazard_positions=hazard_positions,
            agent_xy=data.xpos[self._agent_body][:2],
            agent_geom_ids=self._agent_geom_ids,
            proximity_cost_scaler=self._collision_cost,
            collision_cost=self._collision_cost,
            contact_geom1=None,  # force proximity path: MJX geom_xpos is stale for mocap boxes
            contact_geom2=None,
            contact_dist=None,
            ncon=None,
        )

    def _get_obs(self, data: mjx.Data, info: Dict) -> jp.ndarray:
        """Create observation with button and hazard lidars."""
        agent_pos = data.xpos[self._agent_body]
        button_positions = info['button_positions']
        active_button_idx = info['active_button_idx']
        button_timer = info['button_timer']
        hazard_positions = info['hazard_positions']

        # Sensor observations
        sensor_data = data.sensordata
        default_val = jp.zeros(3, dtype=sensor_data.dtype)

        accel_adr, accel_dim = self._sensor_info.get('accelerometer', (0, 0))
        accelerometer = jax.lax.dynamic_slice(sensor_data, (accel_adr,), (accel_dim,))
        accelerometer = jp.where(accel_dim == 3, accelerometer, default_val)

        velo_adr, velo_dim = self._sensor_info.get('velocimeter', (0, 0))
        velocimeter = jax.lax.dynamic_slice(sensor_data, (velo_adr,), (velo_dim,))
        velocimeter = jp.where(velo_dim == 3, velocimeter, default_val)

        gyro_adr, gyro_dim = self._sensor_info.get('gyro', (0, 0))
        gyro = jax.lax.dynamic_slice(sensor_data, (gyro_adr,), (gyro_dim,))
        gyro = jp.where(gyro_dim == 3, gyro, default_val)

        mag_adr, mag_dim = self._sensor_info.get('magnetometer', (0, 0))
        magnetometer = jax.lax.dynamic_slice(sensor_data, (mag_adr,), (mag_dim,))
        magnetometer = jp.where(mag_dim == 3, magnetometer, default_val)

        sensor_obs = jp.concatenate([accelerometer, velocimeter, gyro, magnetometer])

        # Agent-centric transformation
        agent_z_angle = self.get_agent_heading(data)
        cos_a = jp.cos(agent_z_angle)
        sin_a = jp.sin(agent_z_angle)

        _lidar_num_bins = self._lidar_num_bins
        _lidar_max_dist = self._lidar_max_dist
        _lidar_alias = self._lidar_alias
        bin_size = (2 * jp.pi) / _lidar_num_bins

        def _accumulate_lidar(lidar_init, positions):
            def process_one(lidar, pos):
                rel_xy = pos[:2] - agent_pos[:2]
                rel_x = rel_xy[0] * cos_a + rel_xy[1] * sin_a
                rel_y = -rel_xy[0] * sin_a + rel_xy[1] * cos_a
                dist = safe_norm(jp.array([rel_x, rel_y]))
                angle = jp.arctan2(rel_y, rel_x)
                angle = (angle + 2 * jp.pi) % (2 * jp.pi)
                bin_idx_float = angle / bin_size
                bin_idx = jp.minimum(jp.floor(bin_idx_float), _lidar_num_bins - 1).astype(int)
                sensor_val = jp.maximum(0.0, _lidar_max_dist - dist) / _lidar_max_dist
                sensor_val = jp.where(dist > _lidar_max_dist, 0.0, sensor_val)
                lidar = lidar.at[bin_idx].set(jp.maximum(lidar[bin_idx], sensor_val))

                if _lidar_alias:
                    alias_factor = bin_idx_float - bin_idx
                    bin_plus = (bin_idx + 1) % _lidar_num_bins
                    bin_minus = (bin_idx - 1 + _lidar_num_bins) % _lidar_num_bins
                    lidar = lidar.at[bin_plus].set(jp.maximum(lidar[bin_plus], alias_factor * sensor_val))
                    lidar = lidar.at[bin_minus].set(jp.maximum(lidar[bin_minus], (1.0 - alias_factor) * sensor_val))
                return lidar, None

            lidar_out, _ = jax.lax.scan(process_one, lidar_init, positions)
            return lidar_out

        # Button lidar (all buttons, hidden when timer > 0)
        def _compute_button_lidar(_):
            return _accumulate_lidar(jp.zeros(_lidar_num_bins), button_positions)

        button_lidar = jax.lax.cond(
            button_timer == 0,
            _compute_button_lidar,
            lambda _: jp.zeros(_lidar_num_bins),
            operand=None
        )

        # Button compass (active button)
        active_button_xy = button_positions[active_button_idx, :2]
        rel_goal_xy = active_button_xy - agent_pos[:2]
        rel_x = rel_goal_xy[0] * cos_a + rel_goal_xy[1] * sin_a
        rel_y = -rel_goal_xy[0] * sin_a + rel_goal_xy[1] * cos_a
        button_comp = jp.array([rel_x, rel_y]) / (safe_norm(jp.array([rel_x, rel_y])) + 1e-8)

        # Hazard lidar
        hazard_lidar = _accumulate_lidar(jp.zeros(_lidar_num_bins), hazard_positions)

        # Hazard compasses (closest k, fixed-size)
        H = hazard_positions.shape[0]
        k = int(self._hazard_compass_k)
        if H == 0:
            hazard_compasses_flat = jp.zeros((k * 2,))
        else:
            rel_xy = hazard_positions[:, :2] - agent_pos[:2]
            d2 = jp.sum(rel_xy * rel_xy, axis=1)
            order = jp.argsort(d2)
            k_eff = min(k, H)
            closest = order[:k_eff]
            if k_eff < k:
                pad = -jp.ones((k - k_eff,), dtype=jp.int32)
                closest = jp.concatenate([closest, pad], axis=0)

            def compute_compass_for_index(idx):
                safe_idx = jp.maximum(idx, 0)
                hazard_pos_3d = hazard_positions[safe_idx]
                rel_hazard = hazard_pos_3d - agent_pos
                world_dx = rel_hazard[0]
                world_dy = rel_hazard[1]
                agent_centric_dx = world_dx * cos_a + world_dy * sin_a
                agent_centric_dy = -world_dx * sin_a + world_dy * cos_a
                rel_vec = jp.array([agent_centric_dx, agent_centric_dy])
                compass = rel_vec / (safe_norm(rel_vec) + 1e-8)
                return jp.where(idx >= 0, compass, jp.zeros(2))

            hazard_compasses = jax.vmap(compute_compass_for_index)(closest)
            hazard_compasses_flat = hazard_compasses.reshape((-1,))

        obs = jp.concatenate([
            sensor_obs,
            button_lidar,
            button_comp,
            hazard_lidar,
            hazard_compasses_flat,
        ])
        return obs

    def _get_metrics(self, data: mjx.Data, reward: jp.ndarray, cost: jp.ndarray,
                     dist_goal: jp.ndarray, last_dist_goal: jp.ndarray,
                     ctrl_cost: jp.ndarray, info: Dict) -> Dict:
        """Get metrics dictionary."""
        agent_pos = data.xpos[self._agent_body]
        return {
            "reward": reward,
            "cost": cost,
            "x_position": agent_pos[0],
            "y_position": agent_pos[1],
            "distance_to_goal": dist_goal,
            "ctrl_cost": ctrl_cost,
        }

    @property
    def observation_size(self) -> int:
        """Return the size of the observation vector."""
        return (
            12 +  # Sensor data
            self._lidar_num_bins +  # Button lidar
            2 +  # Button compass
            self._lidar_num_bins +  # Hazard lidar
            self._hazard_compass_k * 2  # Hazard compasses
        )


class SafeButtonPoint(SafeButton):
    """Point agent for button task.

    The point agent is a simple 2D navigation agent with thrust and yaw control.
    """

    @property
    def agent_xml_file(self) -> str:
        return "point_circle.xml"

    @property
    def agent_body_index(self) -> int:
        return 1  # Point agent body is at index 1

    @property
    def default_healthy_z_range(self) -> Tuple[float, float]:
        return (0.05, 0.3)

    @property
    def default_agent_keepout(self) -> float:
        return 0.1

    @property
    def required_sensors(self) -> List[str]:
        return ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']

    def get_agent_heading(self, data: mjx.Data) -> jp.ndarray:
        """Get the agent's current heading angle from z_hinge rotation."""
        return data.qpos[2]  # z_hinge_angle for point agent
