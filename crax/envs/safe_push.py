"""
BlockPushGoal Environment

Abstract base class and agent-specific implementations for block pushing tasks.
The agent must push a block to a goal while avoiding hazards.
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
    create_goal_manager_from_params,
    generate_goal_xml_from_base,
    safe_norm,
    sdf_cylinder,
    sdf_cube,
    place_objects,
    sample_position_in_extents,
    base_xml_file_path,
    add_walls_to_specs,
    choose_valid_position_shape_aware,
)
from crax.envs.hazards import _type_defaults_from_registry, compute_hazard_costs
from crax.io import mjcf


class SafePush(PipelineEnv, ABC):
    """
    Abstract base class for block pushing environments.

    The agent must push a block to a goal while avoiding hazards.
    The block (not the agent) must reach the goal to trigger success/respawn.

    Features:
    - Pushable block with configurable mass/friction
    - Agent avoids hazards (block can pass through hazards)
    - Goal respawns when block reaches it
    - Rich sensor suite (accelerometer, velocimeter, gyro, magnetometer)
    - Lidar for goals, hazards, and block
    - Agent-centric observations

    Subclasses must implement agent-specific properties.
    """

    @property
    @abstractmethod
    def agent_xml_file(self) -> str:
        """Return the XML file name for this agent (e.g., 'point_push.xml')."""
        pass

    @property
    @abstractmethod
    def agent_body_index(self) -> int:
        """Return the body index of the agent in the MuJoCo model."""
        pass

    @property
    @abstractmethod
    def default_healthy_z_range(self) -> Tuple[float, float]:
        """Return the default (min, max) z-range for healthy state."""
        pass

    @property
    @abstractmethod
    def default_agent_keepout(self) -> float:
        """Return the default keepout radius for placement."""
        pass

    @property
    @abstractmethod
    def required_sensors(self) -> List[str]:
        """Return list of required sensor names."""
        pass

    def get_agent_heading(self, data: mjx.Data) -> jp.ndarray:
        """Get the agent's current heading angle. Override for different agents."""
        return data.qpos[2]  # Default: z_hinge_angle for point agent

    def __init__(
            self,
            # Episode settings
            episode_length: int = 2000,
            # Physics settings
            backend: str = 'mjx',
            n_frames: int = 4,
            timestep: float = 0.02,
            terminate_when_unhealthy: bool = True,
            healthy_z_range: Tuple[float, float] = None,
            reset_noise_scale: float = 0.005,
            max_velocity: float = 5.0,
            # Reward settings
            reward_goal: float = 1.0,
            reward_distance_scale: float = 1.0,
            reward_agent_block_scale: float = 0.1,  # Reward for agent getting closer to block
            # Cost settings
            cost_scale: float = 2.0,
            collision_cost: float = 3.0,
            ctrl_cost_weight: float = 0.001,
            # Lidar settings
            lidar_bins: int = 16,
            lidar_max_dist: float = 3.0,
            lidar_alias: bool = True,
            hazard_compass_k: int = 8,
            # Placement settings
            placement_extents: Tuple[float, float, float, float] = (-2.5, -2.5, 2.5, 2.5),
            agent_keepout: float = None,
            placement_margin: float = 0.01,
            max_placement_attempts: int = 100,
            max_layout_attempts: int = 1000,
            # Goal settings
            goal_type: str = 'cube',
            goal_count: int = 1,
            goal_size: float = 0.2,
            goal_height: float = 0.2,
            goal_positions: Optional[List] = None,
            goal_collidable: bool = False,
            goal_velocity: float = 0.0,  # Goal movement speed (0 = stationary, >0 = moving)
            # Hazard settings - list of specs: {type, count, size, height, collidable, fixed, density}
            hazard_specs: Optional[List[Dict]] = None,
            # Block surface type: "stone" (moderate friction) or "icey" (very slippery, default)
            block_surface: str = "stone",
            # Debug
            debug: bool = False,
            **kwargs,
    ):
        # Store debug flag early for use in initialization
        self._debug = debug

        # Use defaults from abstract properties if not provided
        if healthy_z_range is None:
            healthy_z_range = self.default_healthy_z_range
        if agent_keepout is None:
            agent_keepout = self.default_agent_keepout

        # Get agent XML file from abstract property
        base_agent_file_name = self.agent_xml_file

        # Build default hazard specs if none provided
        if hazard_specs is None:
            hazard_specs = [
                dict(
                    type='cube',
                    count=6,
                    size=0.2,
                    height=0.2,
                    collidable=True,
                    movable=False,
                    density=1.0,
                ),
                dict(
                    type='cube',
                    count=6,
                    size=0.3,
                    height=0.01,
                    collidable=False,
                    movable=False,
                    density=1.0,
                ),
                dict(
                    type='outer_wall',
                    offset=1.0,  # Push walls out ~3 block widths beyond placement extents
                    thickness=0.1,
                    height=0.1,
                    collidable=True,
                    fixed=True,
                ),
            ]

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

        # Build managers
        self._hazard_manager = create_hazard_manager_from_specs(hazard_specs)
        self._goal_manager = create_goal_manager_from_params(
            goal_type=goal_type,
            goal_count=goal_count,
            goal_size=goal_size,
            goal_height=goal_height,
            goal_positions=goal_positions,
        )

        # Obtain lists of goals and hazards
        goals = self._goal_manager.goals
        hazards = self._hazard_manager.hazards

        # Per-object keepouts (no margin; margin is handled in placement math)
        self._goal_keepouts = jp.array(
            [g.get_keepout_radius() for g in goals], dtype=jp.float32
        )
        self._hazard_keepouts = jp.array(
            [h.get_keepout_radius() for h in hazards], dtype=jp.float32
        )

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

        self._hazard_is_rect = jp.array(is_rect, dtype=jp.bool_)
        self._hazard_half_extents = jp.stack(half_ext) if len(hazards) > 0 else jp.zeros((0, 2))
        self._hazard_radii = jp.array(radii) if len(hazards) > 0 else jp.zeros((0,))

        # For goal reachability checks
        packed = [g.encode_static_params() for g in goals]
        self._goal_type_ids = jp.array([p.type_id for p in packed], dtype=jp.int32)
        self._goal_radii = jp.array([p.radius for p in packed], dtype=jp.float32)
        self._goal_box_he = jp.array([p.half_extents_xy for p in packed], dtype=jp.float32)
        self._goal_yaws = jp.array([p.yaw for p in packed], dtype=jp.float32)

        # Generate XML dynamically with the configured goals and hazards
        xml_path = generate_goal_xml_from_base(base_agent_file_name, self._goal_manager, self._hazard_manager)
        self._xml_base_file_path = base_xml_file_path(base_agent_file_name)

        # Block surface physics settings
        # damping controls how quickly the block decelerates after being pushed
        surface_settings = {
            "stone": {"damping": 0.015},   # Default: moderate sliding (~2 block widths)
            "icey": {"damping": 0.0025},   # Slippery: long sliding (~10 block widths)
        }
        if block_surface not in surface_settings:
            raise ValueError(f"Unknown block_surface '{block_surface}'. Must be one of: {list(surface_settings.keys())}")
        self._block_surface = block_surface
        block_damping = surface_settings[block_surface]["damping"]

        try:
            mj_model = mujoco.MjModel.from_xml_path(xml_path)
            mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
            mj_model.opt.timestep = timestep
            mj_model.opt.iterations = 4
            mj_model.opt.ls_iterations = 4

            # Apply block surface damping to block joints
            block_x_joint = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "block_x")
            block_y_joint = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "block_y")
            if block_x_joint >= 0:
                mj_model.dof_damping[block_x_joint] = block_damping
            if block_y_joint >= 0:
                mj_model.dof_damping[block_y_joint] = block_damping

            if self._debug:
                print(f"Block surface: {block_surface}, damping: {block_damping}")
        finally:
            # Clean up temporary XML file
            if os.path.exists(xml_path):
                os.unlink(xml_path)

        # after loading mj_model
        def _mocap_id_for_body(name: str) -> int:
            b = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if b < 0:
                raise RuntimeError(f"Missing body named {name}")
            mid = int(mj_model.body_mocapid[b])
            if mid < 0:
                raise RuntimeError(f"Body {name} is not mocap (body_mocapid < 0)")
            return mid

        sys = mjcf.load_model(mj_model)

        # Pass physics settings to PipelineEnv
        super().__init__(sys, backend=backend, n_frames=n_frames)

        # Episode length for this task
        self.episode_length = episode_length

        # Get body IDs
        self._agent_body = self.agent_body_index

        # Block body ID - the pushable block
        self._block_body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "block")
        if self._block_body < 0:
            raise RuntimeError("Block body not found in model! Make sure point_push.xml has a body named 'block'")
        if self._debug:
            print(f"Block body ID: {self._block_body}")

        # goals (the names must match what XMLBuilder emits)
        self._goal_mocap_ids = []
        for goal in goals:
            self._goal_mocap_ids.append(
                _mocap_id_for_body(f"goal{goal.goal_id}"))

        # hazards (the names must match what XMLBuilder emits)
        self._hazard_mocap_ids = []
        for hazard in hazards:
            self._hazard_mocap_ids.append(_mocap_id_for_body(f"hazard{hazard.hazard_id}"))

        # Cache agent and hazard geom ids for contact checks
        self._agent_geom_ids = jp.array(
            [i for i in range(mj_model.ngeom) if mj_model.geom_bodyid[i] == self._agent_body],
            dtype=jp.int32
        )

        # Assign geom_id to each hazard by name "hazard{i}"
        for hazard in hazards:
            gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"hazard{hazard.hazard_id}")
            hazard.geom_id = gid

        # Get hazard information from HazardManager
        self._num_hazards = self._hazard_manager.get_hazard_count()
        self._num_fixed_hazards = self._hazard_manager.get_fixed_hazard_count()
        self._num_movable_hazards = self._num_hazards - self._num_fixed_hazards
        self._num_goals = self._goal_manager.get_goal_count()

        # --- Find Sensor Indices, Addresses, and Dimensions ---
        self._sensor_info = {}
        required_sensors = self.required_sensors
        sensor_found_flags = {name: False for name in required_sensors}
        if mj_model.nsensor > 0:
            if self._debug:
                print(f"Model has {mj_model.nsensor} sensors. Searching for required sensors...")
            for i in range(mj_model.nsensor):
                name = mj_model.sensor(i).name
                if name in required_sensors:
                    start_adr = mj_model.sensor_adr[i]
                    dim = mj_model.sensor_dim[i]
                    self._sensor_info[name] = (start_adr, dim)
                    sensor_found_flags[name] = True
                    if self._debug:
                        print(f"  Found sensor: {name}, ID: {i}, Address: {start_adr}, Dim: {dim}")
        else:
            print("Warning: Model has no sensors defined (mj_model.nsensor = 0).")

        # Check if all required sensors were found
        missing_sensors = [name for name, found in sensor_found_flags.items() if not found]
        if missing_sensors:
            print(f"Warning: Could not find the following required sensors: {missing_sensors}")
        # --- End Sensor Info ---

        # Reward
        self._reward_goal = reward_goal
        self._reward_distance = reward_distance_scale
        self._reward_agent_block = reward_agent_block_scale

        # Cost
        self._ctrl_cost_weight = ctrl_cost_weight
        self._proximity_cost_scaler = cost_scale
        self._collision_cost = collision_cost

        # Physics
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._max_velocity = max_velocity

        # Lidar
        self._lidar_num_bins = lidar_bins
        self._lidar_max_dist = lidar_max_dist
        self._hazard_compass_k = hazard_compass_k

        # Placement
        self._placement_extents = placement_extents
        self._agent_keepout = agent_keepout
        self._placement_margin = placement_margin
        self._max_placement_attempts = max_placement_attempts
        self._max_layout_attempts = max_layout_attempts

        # Goal movement
        self._goal_velocity = goal_velocity

        if self._debug:
            print(
                f"BlockPushGoal initialized with {self._num_hazards} hazards and {self._num_goals} goals")
            cube_hazards = self._hazard_manager.get_hazards_by_type("cube")
            cylinder_hazards = self._hazard_manager.get_hazards_by_type("cylinder")
            print(f"Hazard composition: {len(cube_hazards)} cubes, {len(cylinder_hazards)} cylinders")
            cube_goals = self._goal_manager.get_goals_by_type("cube")
            cylinder_goals = self._goal_manager.get_goals_by_type("cylinder")
            print(f"Goal composition: {len(cube_goals)} cubes, {len(cylinder_goals)} cylinders")
            print(f"Using modular goal and hazard system with dynamic XML generation")

    def reset(self, rng: jp.ndarray) -> State:
        """Reset the environment with constrained placement using JAX control flow."""
        rng, rng1, rng2, rng_layout = jax.random.split(rng, 4)

        # Randomize initial position with small noise
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

        # Build layout: goals then hazards using lax.scan
        num_candidates = self._max_placement_attempts

        # Arrays to accumulate positions: max entries = agent + goal + hazards
        max_entries = 1 + self._num_goals + self._num_movable_hazards
        positions_xy = jp.zeros((max_entries, 2))
        keepouts = jp.zeros((max_entries,))

        # seed with agent
        positions_xy = positions_xy.at[0].set(agent_pos[:2])
        keepouts = keepouts.at[0].set(self._agent_keepout)
        count = jp.array(1, dtype=jp.int32)

        # Place goals
        (rng_layout, positions_xy, keepouts, count, goal_positions) = place_objects(
            rng_key=rng_layout,
            positions_xy=positions_xy,
            keepouts_array=keepouts,
            placed_count=count,
            per_item_keepouts=self._goal_keepouts,
            num_items=self._num_goals,
            num_candidates=num_candidates,
            placement_extents=self._placement_extents,
            placement_margin=self._placement_margin,
        )

        # Place hazards
        (rng_layout, positions_xy, keepouts, count, hazard_positions) = place_objects(
            rng_key=rng_layout,
            positions_xy=positions_xy,
            keepouts_array=keepouts,
            placed_count=count,
            per_item_keepouts=self._hazard_keepouts,
            num_items=self._num_movable_hazards,
            num_candidates=num_candidates,
            placement_extents=self._placement_extents,
            placement_margin=self._placement_margin,
        )

        # Set goal and hazard positions in mocap
        goal_ids = jp.array(self._goal_mocap_ids, dtype=jp.int32)

        # Only include movable hazards in mocap positioning
        if self._num_movable_hazards > 0:
            hazard_ids = jp.array(self._hazard_mocap_ids[:self._num_movable_hazards], dtype=jp.int32)
        else:
            hazard_ids = jp.array([], dtype=jp.int32)

        # Only concatenate if we have movable hazards
        if self._num_movable_hazards > 0:
            all_ids = jp.concatenate([goal_ids, hazard_ids])
            all_pos = jp.concatenate([goal_positions, hazard_positions], axis=0)
        else:
            all_ids = goal_ids
            all_pos = goal_positions

        mpos = data.mocap_pos
        mpos = mpos.at[all_ids].set(all_pos)
        data = data.replace(mocap_pos=mpos)
        hazard_positions = data.mocap_pos[jp.array(self._hazard_mocap_ids)]

        # Get positions
        agent_pos = data.xpos[self._agent_body]
        block_pos = data.xpos[self._block_body]
        goals_xy = goal_positions[:, :2]
        agent_xy = agent_pos[:2]
        block_xy = block_pos[:2]

        # Calculate initial distance from BLOCK to nearest goal (not agent!)
        initial_dist_goal = jp.min(jp.sqrt(jp.sum(jp.square(goals_xy - block_xy[None, :]), axis=1) + 1e-8))

        # Calculate initial distance from AGENT to BLOCK
        initial_dist_block = safe_norm(agent_xy - block_xy)

        # Sample random movement directions for each goal (unit vectors)
        rng_layout, rng_dir = jax.random.split(rng_layout)
        goal_angles = jax.random.uniform(rng_dir, (self._num_goals,), minval=0.0, maxval=2.0 * jp.pi)
        goal_directions = jp.stack([jp.cos(goal_angles), jp.sin(goal_angles)], axis=-1)  # (num_goals, 2)

        if self._debug:
            print(f"[RESET] Agent pos: {agent_xy}, Block pos: {block_xy}, Goal pos: {goals_xy[0]}")
            print(f"[RESET] Initial block-to-goal distance: {initial_dist_goal}")
            print(f"[RESET] Initial agent-to-block distance: {initial_dist_block}")
            print(f"[RESET] Goal velocity: {self._goal_velocity}, Initial directions: {goal_directions}")

        info = {
            "goal_positions": goal_positions,
            "hazard_positions": hazard_positions,
            "block_position": block_pos,
            "step_count": 0,
            "last_dist_goal": initial_dist_goal,
            "last_dist_block": initial_dist_block,
            "goal_directions": goal_directions,  # Movement directions for moving goals
            "cost": 0.0,
            "respawn_rng": rng_layout,
        }

        obs = self._get_obs(data)
        reward, cost, ctrl_cost, done = jp.zeros(4)
        metrics = self._get_metrics(data, reward, cost, initial_dist_goal, initial_dist_goal, ctrl_cost)

        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Execute one step in the environment."""

        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        # Get positions
        agent_pos = data.xpos[self._agent_body]
        block_pos = data.xpos[self._block_body]
        hazard_positions = state.info['hazard_positions']
        goal_positions = state.info['goal_positions']
        last_dist_goal = state.info['last_dist_goal']
        last_dist_block = state.info['last_dist_block']
        goal_directions = state.info['goal_directions']

        # ============================== DISTANCE REWARD (before goal movement) ==============================
        # Calculate distance reward BEFORE goals move, so reward reflects agent's actions only.
        # This prevents unfair negative rewards when the goal moves away from the block.
        agent_xy = agent_pos[:2]
        block_xy = block_pos[:2]
        goals_xy_before_move = goal_positions[:, :2]

        # Distance from block to nearest goal (before goal moves this step)
        center_dists_before = jp.sqrt(jp.sum(jp.square(goals_xy_before_move - block_xy[None, :]), axis=1) + 1e-8)
        dist_goal_before_move = jp.min(center_dists_before)

        # Distance reward based on goal position BEFORE movement (fair to agent)
        dist_reward = (last_dist_goal - dist_goal_before_move) * self._reward_distance

        # Agent-to-block reward (also calculated before goal movement for consistency)
        dist_block = safe_norm(agent_xy - block_xy)
        agent_block_reward = (last_dist_block - dist_block) * self._reward_agent_block

        # ============================== GOAL MOVEMENT ==============================
        # Move goals if goal_velocity > 0
        rng_movement = state.info["respawn_rng"]

        def _move_goals(args):
            """Move goals and handle boundary bouncing."""
            gpos, gdirs, rng = args
            # Move goals: new_pos = old_pos + direction * velocity * dt
            dt = self.dt  # timestep
            velocity = self._goal_velocity
            new_xy = gpos[:, :2] + gdirs * velocity * dt

            # Check boundary collision for each goal and bounce
            minx, miny, maxx, maxy = self._placement_extents
            margin = self._placement_margin

            # For each goal, check if it hit a boundary and reflect direction
            def _bounce_goal(carry, idx):
                positions, directions, rng_key = carry
                pos_xy = positions[idx]
                dir_xy = directions[idx]

                # Check x boundaries
                hit_min_x = pos_xy[0] < minx + margin
                hit_max_x = pos_xy[0] > maxx - margin
                hit_x = jp.logical_or(hit_min_x, hit_max_x)

                # Check y boundaries
                hit_min_y = pos_xy[1] < miny + margin
                hit_max_y = pos_xy[1] > maxy - margin
                hit_y = jp.logical_or(hit_min_y, hit_max_y)

                hit_boundary = jp.logical_or(hit_x, hit_y)

                # Reflect direction: flip the component that hit the wall
                new_dir_x = jp.where(hit_x, -dir_xy[0], dir_xy[0])
                new_dir_y = jp.where(hit_y, -dir_xy[1], dir_xy[1])
                reflected_dir = jp.array([new_dir_x, new_dir_y])

                # Sample a completely new random direction if hit boundary (more variety)
                rng_key, rng_angle = jax.random.split(rng_key)
                random_angle = jax.random.uniform(rng_angle, (), minval=0.0, maxval=2.0 * jp.pi)
                random_dir = jp.array([jp.cos(random_angle), jp.sin(random_angle)])

                # Use random direction instead of simple reflection for more interesting movement
                new_dir = jp.where(hit_boundary, random_dir, dir_xy)

                # Clamp position to stay within bounds
                clamped_x = jp.clip(pos_xy[0], minx + margin, maxx - margin)
                clamped_y = jp.clip(pos_xy[1], miny + margin, maxy - margin)
                clamped_pos = jp.array([clamped_x, clamped_y])

                # Update arrays
                new_positions = positions.at[idx].set(clamped_pos)
                new_directions = directions.at[idx].set(new_dir)

                return (new_positions, new_directions, rng_key), None

            (final_positions, final_directions, final_rng), _ = jax.lax.scan(
                _bounce_goal, (new_xy, gdirs, rng), jp.arange(self._num_goals)
            )

            # Reconstruct full goal positions with z coordinate
            final_goal_positions = gpos.at[:, :2].set(final_positions)

            return final_goal_positions, final_directions, final_rng

        def _keep_goals(args):
            """Keep goals stationary."""
            gpos, gdirs, rng = args
            return gpos, gdirs, rng

        # Only move goals if velocity > 0
        goal_positions, goal_directions, rng_movement = jax.lax.cond(
            self._goal_velocity > 0.0,
            _move_goals,
            _keep_goals,
            (goal_positions, goal_directions, rng_movement)
        )

        # Update mocap positions with moved goals
        mocap_pos = data.mocap_pos
        mocap_pos = mocap_pos.at[jp.array(self._goal_mocap_ids)].set(goal_positions)
        data = data.replace(mocap_pos=mocap_pos)

        # ============================== GOAL REACHED DETECTION ==============================
        # NOTE: Goal detection uses BLOCK position and MOVED goal positions
        goals_xy = goal_positions[:, :2]

        # Goal reached when block center is INSIDE the goal box (SDF <= 0)
        is_cube = (self._goal_type_ids == 0)
        sdf_cube_vals = jax.vmap(lambda c, he, y: sdf_cube(block_xy, c, he, y))(
            goals_xy, self._goal_box_he, self._goal_yaws
        )
        sdf_cylinder_vals = jax.vmap(lambda c, r: sdf_cylinder(block_xy, c, r))(
            goals_xy, self._goal_radii
        )
        sdf_vals = jp.where(is_cube, sdf_cube_vals, sdf_cylinder_vals)
        reached_mask = (sdf_vals <= 0.0)
        num_goals_reached = jp.sum(reached_mask.astype(jp.int32))

        goal_reward = self._reward_goal * num_goals_reached

        # Debug output
        if self._debug:
            jax.debug.print("[STEP] Block: {b}, Goal: {g}, Dist: {d}, Reached: {r}, AgentBlockDist: {abd}",
                          b=block_xy, g=goals_xy[0], d=dist_goal_before_move, r=num_goals_reached, abd=dist_block)

        # ============================== GOAL RESPAWN ==============================

        # Build object arrays used during goal respawn checks:
        # we treat the agent, all hazards, and all goals as objects with per-object keepouts.
        hazard_positions_xy = hazard_positions[:, :2]
        goal_positions_xy = goal_positions[:, :2]

        total_objects = 1 + self._num_hazards + self._num_goals

        # Object state buffers (positions + shape for placement)
        object_positions_xy = jp.zeros((total_objects, 2))

        # Shapes:
        object_is_rect = jp.zeros((total_objects,), dtype=jp.bool_)
        object_half_extents = jp.zeros((total_objects, 2))  # only for rects; zeros otherwise
        object_radii = jp.zeros((total_objects,))  # only for circles; zeros otherwise

        # Agent as object 0 (treat agent as circle with keepout self._agent_keepout)
        object_positions_xy = object_positions_xy.at[0].set(agent_xy)
        object_is_rect = object_is_rect.at[0].set(False)
        object_radii = object_radii.at[0].set(self._agent_keepout)

        # Hazards as objects [1 : 1+H)
        hazard_span_start = 1
        hazard_span_end = hazard_span_start + self._num_hazards
        object_positions_xy = object_positions_xy.at[hazard_span_start:hazard_span_end].set(hazard_positions_xy)

        object_is_rect = object_is_rect.at[hazard_span_start:hazard_span_end].set(self._hazard_is_rect)
        object_half_extents = object_half_extents.at[hazard_span_start:hazard_span_end].set(self._hazard_half_extents)
        object_radii = object_radii.at[hazard_span_start:hazard_span_end].set(self._hazard_radii)

        # Goals as objects [hazard_span_end : hazard_span_end + G)
        goal_span_start = hazard_span_end
        goal_span_end = goal_span_start + self._num_goals
        object_positions_xy = object_positions_xy.at[goal_span_start:goal_span_end].set(goal_positions_xy)
        object_is_rect = object_is_rect.at[goal_span_start:goal_span_end].set(False)
        object_radii = object_radii.at[goal_span_start:goal_span_end].set(self._goal_keepouts)

        active_object_count = jp.array(total_objects, dtype=jp.int32)

        # Thread persistent RNG through respawns
        rng_for_goal_respawn = state.info["respawn_rng"]

        def _place_or_keep_goal(carry, goal_index):
            """
            If goal `goal_index` was reached, sample a new valid position for it.
            Otherwise keep its current position. We temporarily disable the goal's
            own object keepout while sampling, to avoid blocking itself.
            Also samples a new movement direction when goal is respawned.
            """
            rng_key, object_positions_xy, object_is_rect, object_half_extents, object_radii, new_goal_positions_out, new_goal_directions_out = carry
            object_slot = goal_span_start + goal_index  # where this goal sits in the object arrays

            def _place_new(_):
                # Temporarily disable this goal's own keepout while sampling
                keep_is_rect_wo_self = object_is_rect
                keep_half_extents_wo_self = object_half_extents
                keep_radii_wo_self = object_radii.at[object_slot].set(0.0)

                goal_keepout_radius = self._goal_keepouts[goal_index]

                # convert [minx, miny, maxx, maxy] -> half-extents [ex, ey]
                minx, miny, maxx, maxy = self._placement_extents
                placement_half_extents = jp.array([(maxx - minx) * 0.5, (maxy - miny) * 0.5], dtype=jp.float32)

                new_pos_xyz, next_rng = choose_valid_position_shape_aware(
                    rng_key,
                    object_positions_xy,
                    keep_is_rect_wo_self,
                    keep_half_extents_wo_self,
                    keep_radii_wo_self,
                    active_object_count,
                    goal_keepout_radius,
                    self._max_placement_attempts,
                    placement_half_extents,
                    self._placement_margin,
                )

                # Sample new random direction for the respawned goal
                next_rng, rng_dir = jax.random.split(next_rng)
                new_angle = jax.random.uniform(rng_dir, (), minval=0.0, maxval=2.0 * jp.pi)
                new_direction = jp.array([jp.cos(new_angle), jp.sin(new_angle)])

                # Update arrays at this slot and restore the radius
                updated_positions_xy = object_positions_xy.at[object_slot].set(new_pos_xyz[:2])
                updated_goal_positions_out = new_goal_positions_out.at[goal_index].set(new_pos_xyz)
                updated_goal_directions_out = new_goal_directions_out.at[goal_index].set(new_direction)

                updated_radii = object_radii.at[object_slot].set(goal_keepout_radius)

                return (next_rng, updated_positions_xy, object_is_rect, object_half_extents, updated_radii,
                        updated_goal_positions_out, updated_goal_directions_out)

            def _keep_old(_):
                updated_goal_positions_out = new_goal_positions_out.at[goal_index].set(goal_positions[goal_index])
                updated_goal_directions_out = new_goal_directions_out.at[goal_index].set(goal_directions[goal_index])
                next_rng, _ = jax.random.split(rng_key)
                return (next_rng, object_positions_xy, object_is_rect, object_half_extents, object_radii,
                        updated_goal_positions_out, updated_goal_directions_out)

            return jax.lax.cond(reached_mask[goal_index], _place_new, _keep_old, operand=None)

        # Compute new positions and directions for all goals (only those reached will move/get new direction)
        new_goal_positions = jp.zeros_like(goal_positions)
        new_goal_directions = jp.zeros_like(goal_directions)
        (rng_for_goal_respawn,
         object_positions_xy,
         object_is_rect,
         object_half_extents,
         object_radii,
         new_goal_positions,
         new_goal_directions) = jax.lax.fori_loop(
            0,
            self._num_goals,
            lambda i, carry: _place_or_keep_goal(carry, i),
            (rng_for_goal_respawn, object_positions_xy, object_is_rect, object_half_extents, object_radii,
             new_goal_positions, new_goal_directions),
        )

        # Scatter updated goal mocaps back into the physics state
        mocap_pos = data.mocap_pos
        mocap_pos = mocap_pos.at[jp.array(self._goal_mocap_ids)].set(new_goal_positions)
        data = data.replace(mocap_pos=mocap_pos)

        # Recalculate distance to goal AFTER respawn to avoid penalty on next step
        # When goal respawns, we need last_dist_goal to reflect the new goal position
        new_goals_xy = new_goal_positions[:, :2]
        new_center_dists = jp.sqrt(jp.sum(jp.square(new_goals_xy - block_xy[None, :]), axis=1) + 1e-8)
        dist_goal_after_respawn = jp.min(new_center_dists)

        # Health check
        min_z, max_z = self._healthy_z_range
        is_healthy = jp.logical_and(
            agent_pos[2] >= min_z,
            agent_pos[2] <= max_z
        ).astype(jp.float32)

        # Termination conditions
        done = jp.logical_or(
            (1.0 - is_healthy) * self._terminate_when_unhealthy,
            jp.any(jp.isnan(agent_pos))
        )

        # ============================== METRICS AGGREGATION ==============================

        # TODO control cost should be a separate cost component, not serve as a reward penalty
        ctrl_cost = jp.sum(jp.square(action)) * self._ctrl_cost_weight

        # Safety cost (distance-based penalty near hazards)
        cost = self._calculate_safety_cost(data, hazard_positions)

        # Total reward
        reward = dist_reward + goal_reward + agent_block_reward

        # Get observation and metrics
        obs = self._get_obs(data)
        metrics = self._get_metrics(data, reward, cost, dist_goal_before_move, last_dist_goal, ctrl_cost)

        # Update info
        # Use dist_goal_after_respawn for last_dist_goal to avoid penalty when goal respawns
        new_info = state.info.copy()
        new_info.update({
            "goal_positions": new_goal_positions,
            "goal_directions": new_goal_directions,
            "block_position": block_pos,
            "step_count": state.info['step_count'] + 1,
            "last_dist_goal": dist_goal_after_respawn,
            "last_dist_block": dist_block,
            "cost": cost,
            "respawn_rng": rng_for_goal_respawn,
        })

        return State(data, obs, reward, done.astype(jp.float32), metrics, new_info)

    def _check_position_valid(self, candidate_pos: jp.ndarray, existing_positions: jp.ndarray,
                              keepout_distances: jp.ndarray) -> bool:
        """Check if a candidate position is valid given existing positions and keepout distances."""
        if len(existing_positions) == 0:
            return True

        # Calculate distances to all existing positions
        distances = jp.sqrt(jp.sum(jp.square(candidate_pos[:2] - existing_positions[:, :2]), axis=1))

        # Check if candidate violates any keepout distance
        violations = distances < keepout_distances + self._placement_margin
        return jp.logical_not(jp.any(violations))

    def _sample_valid_position(self, rng_key: jp.ndarray, existing_positions: jp.ndarray,
                               existing_keepouts: jp.ndarray, keepout: float) -> jp.ndarray:
        """Sample a valid position that doesn't violate placement constraints."""

        def sample_attempt(carry):
            attempt_rng, _ = carry
            attempt_rng, subkey = jax.random.split(attempt_rng)
            candidate = sample_position_in_extents(subkey, self._placement_extents, keepout)
            return attempt_rng, candidate

        # Try multiple attempts to find a valid position
        for attempt in range(self._max_placement_attempts):
            rng_key, subkey = jax.random.split(rng_key)
            candidate = sample_position_in_extents(subkey, self._placement_extents, keepout)

            if self._check_position_valid(candidate, existing_positions, existing_keepouts):
                return candidate

        # If we can't find a valid position, return a fallback
        if self._debug:
            print(f"Warning: Could not find valid position after {self._max_placement_attempts} attempts")
        return sample_position_in_extents(rng_key, self._placement_extents, keepout)  # Return anyway

    def _calculate_safety_cost(self, data: mjx.Data, hazard_positions: jp.ndarray) -> jp.ndarray:
        """Sum of per-hazard costs. Binary collision for collidables, proximity for others."""
        return compute_hazard_costs(
            hazards=self._hazard_manager.hazards,
            hazard_positions=hazard_positions,
            agent_xy=data.xpos[self._agent_body][:2],
            agent_geom_ids=self._agent_geom_ids,
            proximity_cost_scaler=self._proximity_cost_scaler,
            collision_cost=self._collision_cost,
            contact_geom1=getattr(data.contact, "geom1", None),
            contact_geom2=getattr(data.contact, "geom2", None),
            contact_dist=getattr(data.contact, "dist", None),
            ncon=getattr(data, "ncon", None),
        )

    def _get_obs(self, data: mjx.Data) -> jp.ndarray:
        """Creates an observation with separate lidars for goals and hazards.

        Observation structure:
        - accelerometer (3 values)
        - velocimeter (3 values)
        - gyro (3 values)
        - magnetometer (3 values)
        - goal_lidar_obs (configurable bins, default 16) - lidar detecting the goal
        - hazard_lidar_obs (configurable bins, default 16) - lidar detecting hazards
        - goal_comp (2 values) - compass pointing to goal
        - hazard_comps (2 * num_hazards values) - compass pointing to each hazard

        Total: 12 + 2*lidar_num_bins + 2*(num_hazards+1) values
        """
        agent_pos = data.xpos[self._agent_body]
        block_pos = data.xpos[self._block_body]
        goal_pos = data.mocap_pos[self._goal_mocap_ids[0]]  # TODO handle multiple goals

        # 1. Agent sensor observations
        # Access the flat sensordata array
        sensor_data = data.sensordata

        # Extract sensor values using pre-calculated addresses and dimensions
        # Handle potential missing sensors by providing default zero vectors if info not found
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

        # 2. Calculate relative position to goal (world frame)
        rel_goal_pos_3d_world = goal_pos - agent_pos

        # --- Agent-centric transformation ---
        # Get agent's current Z rotation from qpos
        agent_z_angle = self.get_agent_heading(data)
        cos_a = jp.cos(agent_z_angle)
        sin_a = jp.sin(agent_z_angle)

        # World-frame relative XY vector to goal
        world_dx_goal = rel_goal_pos_3d_world[0]
        world_dy_goal = rel_goal_pos_3d_world[1]

        # Transform world-frame relative vector to agent's local frame
        agent_centric_dx_goal = world_dx_goal * cos_a + world_dy_goal * sin_a
        agent_centric_dy_goal = -world_dx_goal * sin_a + world_dy_goal * cos_a

        # 3. Create compass observation (agent-centric) - agent to goal
        agent_centric_rel_goal_xy = jp.array([agent_centric_dx_goal, agent_centric_dy_goal])
        goal_comp = agent_centric_rel_goal_xy / (safe_norm(agent_centric_rel_goal_xy) + 1e-8)

        # 4. Block compasses (agent-centric)
        # 4a. Agent-to-block compass
        rel_block_world = block_pos[:2] - agent_pos[:2]
        agent_centric_dx_block = rel_block_world[0] * cos_a + rel_block_world[1] * sin_a
        agent_centric_dy_block = -rel_block_world[0] * sin_a + rel_block_world[1] * cos_a
        agent_to_block_xy = jp.array([agent_centric_dx_block, agent_centric_dy_block])
        agent_to_block_comp = agent_to_block_xy / (safe_norm(agent_to_block_xy) + 1e-8)

        # 4b. Block-to-goal compass (in agent's frame for consistency)
        rel_goal_from_block_world = goal_pos[:2] - block_pos[:2]
        agent_centric_dx_goal_from_block = rel_goal_from_block_world[0] * cos_a + rel_goal_from_block_world[1] * sin_a
        agent_centric_dy_goal_from_block = -rel_goal_from_block_world[0] * sin_a + rel_goal_from_block_world[1] * cos_a
        block_to_goal_xy = jp.array([agent_centric_dx_goal_from_block, agent_centric_dy_goal_from_block])
        block_to_goal_comp = block_to_goal_xy / (safe_norm(block_to_goal_xy) + 1e-8)

        # 4c. Agent-to-block distance (normalized)
        agent_to_block_dist = safe_norm(rel_block_world) / self._lidar_max_dist

        # 4d. Block-to-goal distance (normalized)
        block_to_goal_dist = safe_norm(rel_goal_from_block_world) / self._lidar_max_dist

        # 5. Create Safety-Gymnasium style Lidars with configurable bins
        _lidar_num_bins = self._lidar_num_bins
        _lidar_max_dist = self._lidar_max_dist
        _lidar_alias = True  # Enable aliasing for smoother readings

        # Initialize separate Lidar observations for goals and hazards
        goal_lidar_obs = jp.zeros(_lidar_num_bins)
        hazard_lidar_obs = jp.zeros(_lidar_num_bins)

        # === GOAL LIDAR ===
        # Use the first goal for the compass to avoid changing obs semantics. For lidar, accumulate all goals.
        bin_size = (2 * jp.pi) / _lidar_num_bins

        def process_goal_lidar(carry, goal_mocap_id):
            """Accumulate lidar signal from a single goal."""
            goal_lidar, agent_pos, cos_a, sin_a = carry

            # Get goal position or a dummy if invalid
            goal_pos_3d = jp.where(
                goal_mocap_id >= 0,
                data.mocap_pos[goal_mocap_id],
                jp.array([0.0, 0.0, 0.0])
            )

            # Relative vector in world frame
            rel_goal_pos_3d_world = goal_pos_3d - agent_pos
            world_dx_goal = rel_goal_pos_3d_world[0]
            world_dy_goal = rel_goal_pos_3d_world[1]

            # Agent-centric transform
            agent_centric_dx_goal = world_dx_goal * cos_a + world_dy_goal * sin_a
            agent_centric_dy_goal = -world_dx_goal * sin_a + world_dy_goal * cos_a

            # Distance and angle
            dist_goal = safe_norm(jp.array([agent_centric_dx_goal, agent_centric_dy_goal]))
            angle_goal = jp.arctan2(agent_centric_dy_goal, agent_centric_dx_goal)
            angle_goal = (angle_goal + 2 * jp.pi) % (2 * jp.pi)

            # Bin index
            bin_idx_float_goal = angle_goal / bin_size
            bin_idx_goal = jp.floor(bin_idx_float_goal)
            bin_idx_goal = jp.minimum(bin_idx_goal, _lidar_num_bins - 1).astype(int)

            # Sensor value with range limit
            sensor_val_goal = jp.maximum(0.0, _lidar_max_dist - dist_goal) / _lidar_max_dist
            sensor_val_goal = jp.where(dist_goal > _lidar_max_dist, 0.0, sensor_val_goal)

            # Zero out if mocap id is invalid
            sensor_val_goal = jp.where(goal_mocap_id >= 0, sensor_val_goal, 0.0)

            # Primary bin: take max across goals
            goal_lidar = goal_lidar.at[bin_idx_goal].set(
                jp.maximum(goal_lidar[bin_idx_goal], sensor_val_goal)
            )

            if _lidar_alias:
                # Alias to neighbors
                alias_factor_goal = bin_idx_float_goal - bin_idx_goal

                bin_plus_idx_goal = (bin_idx_goal + 1) % _lidar_num_bins
                goal_lidar = goal_lidar.at[bin_plus_idx_goal].set(
                    jp.maximum(goal_lidar[bin_plus_idx_goal], alias_factor_goal * sensor_val_goal)
                )

                bin_minus_idx_goal = (bin_idx_goal - 1 + _lidar_num_bins) % _lidar_num_bins
                goal_lidar = goal_lidar.at[bin_minus_idx_goal].set(
                    jp.maximum(goal_lidar[bin_minus_idx_goal], (1.0 - alias_factor_goal) * sensor_val_goal)
                )

            return (goal_lidar, agent_pos, cos_a, sin_a), None

        # Scan over all goals and aggregate their contributions
        goal_mocap_ids_array = jp.array(self._goal_mocap_ids)
        init_goal_carry = (goal_lidar_obs, agent_pos, cos_a, sin_a)
        (goal_lidar_obs, _, _, _), _ = jax.lax.scan(
            process_goal_lidar, init_goal_carry, goal_mocap_ids_array
        )

        # === HAZARD LIDAR ===
        # Process hazards for the hazard lidar
        def process_hazard_lidar(carry, hazard_mocap_id):
            """Process a single hazard for the hazard lidar."""
            hazard_lidar, agent_pos, agent_z_angle, cos_a, sin_a = carry

            # Get hazard position from mocap if valid ID
            hazard_pos_3d = jp.where(
                hazard_mocap_id >= 0,
                data.mocap_pos[hazard_mocap_id],
                jp.array([0.0, 0.0, 0.0])  # Default position for invalid IDs
            )

            # Calculate relative position to hazard (world frame)
            rel_hazard_pos_3d_world = hazard_pos_3d - agent_pos

            # Transform world-frame relative vector to agent's local frame
            world_dx_hazard = rel_hazard_pos_3d_world[0]
            world_dy_hazard = rel_hazard_pos_3d_world[1]

            agent_centric_dx_hazard = world_dx_hazard * cos_a + world_dy_hazard * sin_a
            agent_centric_dy_hazard = -world_dx_hazard * sin_a + world_dy_hazard * cos_a

            # Calculate distance and angle for this hazard
            dist_hazard = safe_norm(jp.array([agent_centric_dx_hazard, agent_centric_dy_hazard]))
            angle_hazard = jp.arctan2(agent_centric_dy_hazard, agent_centric_dx_hazard)
            angle_hazard = (angle_hazard + 2 * jp.pi) % (2 * jp.pi)

            # Determine which bin the hazard falls into
            bin_idx_float_hazard = angle_hazard / bin_size
            bin_idx_hazard = jp.floor(bin_idx_float_hazard)
            bin_idx_hazard = jp.minimum(bin_idx_hazard, _lidar_num_bins - 1).astype(int)

            # Calculate sensor reading for hazard
            sensor_val_hazard = jp.maximum(0.0, _lidar_max_dist - dist_hazard) / _lidar_max_dist
            sensor_val_hazard = jp.where(dist_hazard > _lidar_max_dist, 0.0, sensor_val_hazard)

            # Only process if hazard ID is valid (>= 0)
            sensor_val_hazard = jp.where(hazard_mocap_id >= 0, sensor_val_hazard, 0.0)

            # Update the hazard Lidar observation for the primary bin
            hazard_lidar = hazard_lidar.at[bin_idx_hazard].set(
                jp.maximum(hazard_lidar[bin_idx_hazard], sensor_val_hazard)
            )

            if _lidar_alias:
                # Calculate alias interpolation factor for hazard
                alias_factor_hazard = bin_idx_float_hazard - bin_idx_hazard

                # Bin plus one (wraps around)
                bin_plus_idx_hazard = (bin_idx_hazard + 1) % _lidar_num_bins
                hazard_lidar = hazard_lidar.at[bin_plus_idx_hazard].set(
                    jp.maximum(hazard_lidar[bin_plus_idx_hazard], alias_factor_hazard * sensor_val_hazard)
                )

                # Bin minus one (wraps around)
                bin_minus_idx_hazard = (bin_idx_hazard - 1 + _lidar_num_bins) % _lidar_num_bins
                hazard_lidar = hazard_lidar.at[bin_minus_idx_hazard].set(
                    jp.maximum(hazard_lidar[bin_minus_idx_hazard], (1.0 - alias_factor_hazard) * sensor_val_hazard)
                )

            return (hazard_lidar, agent_pos, agent_z_angle, cos_a, sin_a), None

        # Process all hazards using scan to handle variable number of hazards
        hazard_mocap_ids_array = jp.array(self._hazard_mocap_ids, dtype=jp.int32)
        init_carry = (hazard_lidar_obs, agent_pos, agent_z_angle, cos_a, sin_a)
        (hazard_lidar_obs, _, _, _, _), _ = jax.lax.scan(
            process_hazard_lidar, init_carry, hazard_mocap_ids_array
        )

        # === HAZARD COMPASSES ===
        # Create individual compass observations for each hazard
        def compute_compass_for_hazard(mocap_idx):
            """Compute compass for a specific mocap index."""
            # Handle invalid mocap index
            hazard_pos_3d = jp.where(
                mocap_idx >= 0,
                data.mocap_pos[mocap_idx],
                jp.array([0.0, 0.0, 0.0])
            )

            # Calculate relative position to hazard (world frame)
            rel_hazard_pos_3d_world = hazard_pos_3d - agent_pos

            # Transform world-frame relative vector to agent's local frame
            world_dx_hazard = rel_hazard_pos_3d_world[0]
            world_dy_hazard = rel_hazard_pos_3d_world[1]

            agent_centric_dx_hazard = world_dx_hazard * cos_a + world_dy_hazard * sin_a
            agent_centric_dy_hazard = -world_dx_hazard * sin_a + world_dy_hazard * cos_a

            # Create normalized compass observation (agent-centric)
            rel_vec = jp.array([agent_centric_dx_hazard, agent_centric_dy_hazard])
            compass = rel_vec / (safe_norm(rel_vec) + 1e-8)

            # Return zero compass if invalid mocap index
            return jp.where(mocap_idx >= 0, compass, jp.zeros(2))

        # --- choose closest-k hazards for compasses (fixed-size) ---
        all_hz_ids = jp.array(self._hazard_mocap_ids, dtype=jp.int32)  # (H,)
        H = all_hz_ids.shape[0]

        k = int(self._hazard_compass_k)  # Python int => static
        k_eff = min(k, H)  # Python int => static

        def _pick_and_pad():
            all_hz_pos = data.mocap_pos[all_hz_ids]  # (H,3)
            rel_xy = all_hz_pos[:, :2] - agent_pos[:2]  # (H,2)
            d2 = jp.sum(rel_xy * rel_xy, axis=1)  # (H,)
            order = jp.argsort(d2)  # (H,)

            closest = all_hz_ids[order[:k_eff]]  # (k_eff,) static slice ✅

            if k_eff < k:
                pad = -jp.ones((k - k_eff,), dtype=jp.int32)
                closest = jp.concatenate([closest, pad], axis=0)  # (k,)
            return closest

        def _no_hazards():
            return -jp.ones((k,), dtype=jp.int32)

        closest_ids = jax.lax.cond(H > 0, lambda _: _pick_and_pad(), lambda _: _no_hazards(), operand=None)

        hazard_compasses = jax.vmap(compute_compass_for_hazard)(closest_ids)  # (k,2)
        hazard_compasses_flat = hazard_compasses.reshape((-1,))  # (2k,)

        # Build observation with separate goal and hazard lidars plus individual hazard compasses
        # Added: block observations for push task
        obs = jp.concatenate([
            accelerometer,  # (3,)
            velocimeter,  # (3,)
            gyro,  # (3,)
            magnetometer,  # (3,)
            goal_lidar_obs,  # (16,) - Goal Lidar
            hazard_lidar_obs,  # (16,) - Hazard Lidar
            goal_comp,  # (2,) - Goal compass (agent to goal)
            hazard_compasses_flat,  # (16,) - Individual hazard compasses (8 hazards * 2 each)
            # Block observations (new for push task)
            agent_to_block_comp,  # (2,) - Compass from agent to block
            block_to_goal_comp,  # (2,) - Compass from block to goal
            jp.array([agent_to_block_dist]),  # (1,) - Distance from agent to block (normalized)
            jp.array([block_to_goal_dist]),  # (1,) - Distance from block to goal (normalized)
        ])

        return obs

    def _get_metrics(self, data: mjx.Data, reward: jp.ndarray, cost: jp.ndarray,
                     dist_goal: jp.ndarray, last_dist_goal: jp.ndarray, ctrl_cost: jp.ndarray) -> Dict:
        """Get metrics dictionary."""
        agent_pos = data.xpos[self._agent_body]

        return {
            'reward': reward,
            'cost': cost,
            'x_position': agent_pos[0],
            'y_position': agent_pos[1],
            'distance_to_goal': dist_goal,
            'last_dist_goal': last_dist_goal,
            'ctrl_cost': ctrl_cost,
        }

    @property
    def observation_size(self) -> int:
        """Returns the size of the observation vector."""
        return (
                12 +  # Sensor data (3 each for accel, vel, gyro, mag)
                self._lidar_num_bins * 2 +  # Goal and hazard lidars
                2 +  # Goal compass (agent to goal)
                self._hazard_compass_k * 2 +  # Hazard compasses
                # Block observations (new for push task)
                2 +  # Agent-to-block compass
                2 +  # Block-to-goal compass
                1 +  # Agent-to-block distance
                1  # Block-to-goal distance
        )


class SafePushPoint(SafePush):
    """Point agent that pushes a block to a goal while avoiding hazards.

    Uses the point_push.xml agent model with accelerometer, velocimeter,
    gyro, and magnetometer sensors.
    """

    @property
    def agent_xml_file(self) -> str:
        return "point_push.xml"

    @property
    def agent_body_index(self) -> int:
        return 1

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
        """Get the point agent's current heading angle from z_hinge."""
        return data.qpos[2]


# Backward compatibility: SafePush used to be the point agent directly
# Now it's an abstract base class, so we provide SafePushPoint as default
