"""
SafePointGoal Environment

A cleaned-up point navigation environment with configurable circular hazards.
Features goal resetting, safety costs, and simplified lidar observations.
"""

import os
from typing import Dict

import jax
import mujoco
from jax import numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from brax.envs.base import PipelineEnv, State
from brax.envs.env_utils import create_hazard_manager_from_config, create_goal_manager_from_config, \
    generate_goal_xml_from_base, safe_norm, config_merge, expand_hazard_specs, sdf_cylinder, sdf_cube, place_objects, \
    sample_position_in_extents, base_xml_file_path, add_walls, choose_valid_position_shape_aware
from brax.envs.hazards import _type_defaults_from_registry
from brax.io import mjcf


def default_config() -> config_dict.ConfigDict:
    """Returns the default config for SafePointGoal environment."""
    return config_dict.create(
        # --- Physics settings ---
        physics=config_dict.create(
            backend='mjx',  # Use MJX backend for JAX-friendly physics
            n_frames=4,  # Physics steps per control step
            timestep=0.02,  # Simulation timestep
            terminate_when_unhealthy=True,  # End episode if agent leaves healthy z-range
            healthy_z_range=(0.05, 0.3),  # Min/Max healthy z
            reset_noise_scale=0.005,  # Small reset noise on qpos/qvel
            max_velocity=5.0,  # Optional velocity clamp for safety
        ),

        # --- Reward settings ---
        reward=config_dict.create(
            reward_goal=1.0,  # Sparse reward for reaching goal
            dense_scale=0.0,  # Dense reward scale (progress toward goal)
        ),

        # --- Cost (safety) settings ---
        cost=config_dict.create(
            scaler=1.0,  # Hazard cost scaler
            ctrl_cost_weight=0.001,  # Control effort cost
        ),

        # --- Lidar settings (simplified) ---
        lidar=config_dict.create(
            bins=16,  # Number of bins for goal and hazard lidars
            max_dist=3.0,  # Maximum detection distance
            alias=True,  # Bin aliasing for smoother readings
        ),

        # --- Placement constraints (Safety Gymnasium style) ---
        placement=config_dict.create(
            extents=(-2.5, -2.5, 2.5, 2.5),  # [min_x, min_y, max_x, max_y]
            agent_keepout=0.1,  # Keepout radius around agent
            margin=0.01,  # Additional spacing margin for placement
            attempts_pos=100,  # Max attempts to place a single object
            attempts_layout=1000,  # Max attempts to build a full valid layout
        ),

        # --- Goal settings ---
        goals=config_dict.create(
            type='cube',  # 'cube' or 'cylinder'
            count=1,  # Number of goals to instantiate
            size=0.2,  # Cube: (w,h,d); Cylinder: radius. If None, use goal own defaults.
            height=0.2,  # Cylinder/cube height
            positions=None,  # Optional explicit [(x,y,z), ...]; None => sampled
            collidable=False,  # Whether goal geoms participate in contact
        ),

        # --- Hazard settings (modular list) ---
        hazards=config_dict.create(
            type_defaults=_type_defaults_from_registry(),
            specs=[  # Each spec: {type, count, size, height, collidable, movable, density}
                dict(
                    type='cube',  # Currently 'cube' or 'cylinder'
                    count=8,  # Number of hazards of this type
                    size=0.2,  # Cube: half-size (uniform) or radius proxy; Cylinder: radius
                    height=0.2,  # Cylinder/cube height. If None, use hazard own defaults.
                    collidable=True,  # Whether hazards participate in contact
                    movable=False,  # Dynamic hazards? (WIP)
                    density=1.0,  # kg/m^3 (WIP)
                ),
            ],
        ),

        base_agent_file_name="point.xml",  # Name of the agent from the assets folder
        # --- Debugging ---
        debug=False,  # Print extra diagnostics during setup/reset
    )


class SafePointGoal(PipelineEnv):
    """
    Safe Point Goal Navigation Environment

    A point navigation environment with:
    - Configurable number of circular hazards (default: 8)
    - Smaller hazard sizes (0.3 radius) for more challenging navigation
    - Goal resetting mechanism when goal is reached
    - Safety costs for hazard collisions
    - Rich sensor suite (accelerometer, velocimeter, gyro, magnetometer)
    - Dual lidar system with separate goal and hazard detection
    - Agent-centric observations for better learning
    - Individual compass observations for goal and each hazard

    Default observation space (62 dimensions):
    - Sensor data: 12 values (3 each for accel, velocity, gyro, magnetometer)
    - Goal lidar: 16 bins
    - Hazard lidar: 16 bins  
    - Goal compass: 2 values
    - Hazard compasses: 16 values (8 hazards × 2 values each)
    """

    def __init__(self, config: config_dict.ConfigDict | dict = None):
        if not config:
            config = default_config()
        expand_hazard_specs(config)

        # Store debug flag early for use in initialization
        self._debug = config.debug

        # Add outer walls
        hazards_cfg = config.hazards
        hazards_cfg = add_walls(hazards_cfg, config.placement)

        # Build managers
        self._hazard_manager = create_hazard_manager_from_config(hazards_cfg)
        self._goal_manager = create_goal_manager_from_config(config.goals)

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
        base_file_name = config.base_agent_file_name
        xml_path = generate_goal_xml_from_base(base_file_name, self._goal_manager, self._hazard_manager)
        self._xml_base_file_path = base_xml_file_path(base_file_name)

        try:
            mj_model = mujoco.MjModel.from_xml_path(xml_path)
            mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
            mj_model.opt.timestep = config.physics.timestep
            mj_model.opt.iterations = 4
            mj_model.opt.ls_iterations = 4
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
        super().__init__(sys, backend=config.physics.backend, n_frames=config.physics.n_frames)

        # Get body IDs
        self._agent_body = 1  # agent body

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
        required_sensors = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
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

        # Store configuration
        self._config = config

        # Reward
        self._reward_goal = config.reward.reward_goal
        self._reward_distance = config.reward.dense_scale

        # Cost
        self._ctrl_cost_weight = config.cost.ctrl_cost_weight
        self._cost_scaler = config.cost.scaler

        # Physics
        self._terminate_when_unhealthy = config.physics.terminate_when_unhealthy
        self._healthy_z_range = config.physics.healthy_z_range
        self._reset_noise_scale = config.physics.reset_noise_scale
        self._max_velocity = config.physics.max_velocity

        # Lidar
        self._lidar_num_bins = config.lidar.bins
        self._lidar_max_dist = config.lidar.max_dist

        # Placement
        self._placement_extents = config.placement.extents
        self._agent_keepout = config.placement.agent_keepout
        self._placement_margin = config.placement.margin
        self._max_placement_attempts = config.placement.attempts_pos
        self._max_layout_attempts = config.placement.attempts_layout

        if self._debug:
            print(
                f"SafePointGoal initialized with {self._num_hazards} hazards and {self._num_goals} goals")
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

        # Calculate initial distance to nearest goal
        agent_pos = data.xpos[self._agent_body]
        goals_xy = goal_positions[:, :2]
        agent_xy = agent_pos[:2]
        initial_dist_goal = jp.min(jp.sqrt(jp.sum(jp.square(goals_xy - agent_xy[None, :]), axis=1) + 1e-8))

        info = {
            "goal_positions": goal_positions,
            "hazard_positions": hazard_positions,
            "step_count": 0,
            "last_dist_goal": initial_dist_goal,
            "goals_reached_count": 0,
            "goals_per_episode": 0,
            "cost": 0.0,
            "respawn_rng": rng_layout,
        }

        obs = self._get_obs(data)
        reward, cost, ctrl_cost, goals_reached, goals_per_ep, goals_per_step, done = jp.zeros(7)
        metrics = self._get_metrics(data, reward, cost, initial_dist_goal, initial_dist_goal, ctrl_cost,
                                    goals_reached, goals_per_ep, goals_per_step)

        return State(data, obs, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Execute one step in the environment."""

        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        # Get positions
        agent_pos = data.xpos[self._agent_body]
        hazard_positions = state.info['hazard_positions']
        goal_positions = state.info['goal_positions']
        last_dist_goal = state.info['last_dist_goal']

        # ============================== GOAL REWARDS ==============================

        # Distances to all goals (XY)
        agent_xy = agent_pos[:2]
        goals_xy = goal_positions[:, :2]

        is_cube = (self._goal_type_ids == 0)  # TODO extend for more types

        # vectorized SDFs
        sdf_cube_2d = jax.vmap(lambda c, he, y: sdf_cube(agent_xy, c, he, y))(
            goals_xy, self._goal_box_he, self._goal_yaws
        )
        sdf_cylinder_2d = jax.vmap(lambda c, r: sdf_cylinder(agent_xy, c, r))(
            goals_xy, self._goal_radii
        )

        # pick per-type
        sdf = jp.where(is_cube, sdf_cube_2d, sdf_cylinder_2d)

        reached_mask = (sdf <= 0.0)
        num_goals_reached = jp.sum(reached_mask.astype(jp.int32))

        # Dense reward: distance to nearest goal
        outside_dist = jp.maximum(sdf, 0.0)  # clamp negative (inside) to 0
        dist_goal = jp.min(outside_dist)  # distance to nearest goal boundary

        dist_reward = (last_dist_goal - dist_goal) * self._reward_distance
        goal_reward = self._reward_goal * num_goals_reached

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
            """
            rng_key, object_positions_xy, object_is_rect, object_half_extents, object_radii, new_goal_positions_out = carry
            object_slot = goal_span_start + goal_index  # where this goal sits in the object arrays

            def _place_new(_):
                # Temporarily disable this goal’s own keepout while sampling
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

                # Update arrays at this slot and restore the radius
                updated_positions_xy = object_positions_xy.at[object_slot].set(new_pos_xyz[:2])
                updated_goal_positions_out = new_goal_positions_out.at[goal_index].set(new_pos_xyz)

                updated_radii = object_radii.at[object_slot].set(goal_keepout_radius)

                return (next_rng, updated_positions_xy, object_is_rect, object_half_extents, updated_radii,
                        updated_goal_positions_out)

            def _keep_old(_):
                updated_goal_positions_out = new_goal_positions_out.at[goal_index].set(goal_positions[goal_index])
                next_rng, _ = jax.random.split(rng_key)
                return (next_rng, object_positions_xy, object_is_rect, object_half_extents, object_radii,
                        updated_goal_positions_out)

            return jax.lax.cond(reached_mask[goal_index], _place_new, _keep_old, operand=None)

        # Compute new positions for all goals (only those reached will move)
        new_goal_positions = jp.zeros_like(goal_positions)
        (rng_for_goal_respawn,
         object_positions_xy,
         object_is_rect,
         object_half_extents,
         object_radii,
         new_goal_positions) = jax.lax.fori_loop(
            0,
            self._num_goals,
            lambda i, carry: _place_or_keep_goal(carry, i),
            (rng_for_goal_respawn, object_positions_xy, object_is_rect, object_half_extents, object_radii,
             new_goal_positions),
        )

        # Scatter updated goal mocaps back into the physics state
        mocap_pos = data.mocap_pos
        mocap_pos = mocap_pos.at[jp.array(self._goal_mocap_ids)].set(new_goal_positions)
        data = data.replace(mocap_pos=mocap_pos)

        # ============================== METRICS AGGREGATION ==============================

        # Update counters
        updated_goals_reached = state.info['goals_reached_count'] + num_goals_reached
        updated_goals_per_episode = state.info['goals_per_episode'] + num_goals_reached
        goals_per_step = num_goals_reached.astype(jp.float32)

        # TODO control cost should be a separate cost component, not serve as a reward penalty
        ctrl_cost = jp.sum(jp.square(action)) * self._ctrl_cost_weight

        # Safety cost (distance-based penalty near hazards)
        cost = self._calculate_safety_cost(data, hazard_positions)

        # Total reward
        reward = dist_reward + goal_reward

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

        # Get observation and metrics
        obs = self._get_obs(data)
        metrics = self._get_metrics(data, reward, cost, dist_goal, last_dist_goal, ctrl_cost, updated_goals_reached,
                                    updated_goals_per_episode, goals_per_step)

        # Update info
        new_info = state.info.copy()
        new_info.update({
            "goal_positions": new_goal_positions,
            "step_count": state.info['step_count'] + 1,
            "last_dist_goal": dist_goal,
            "goals_reached_count": updated_goals_reached,
            "goals_per_episode": updated_goals_per_episode,
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
        agent_xy = data.xpos[self._agent_body][:2]

        # Contact buffers (valid inside step; at reset we'll pass None)
        ids1 = getattr(data.contact, "geom1", None)
        ids2 = getattr(data.contact, "geom2", None)
        dist = getattr(data.contact, "dist", None)
        ncon = getattr(data, "ncon", None)

        total = jp.array(0.0)
        # Unrolled small loop; JAX traces a static graph here
        for i, h in enumerate(self._hazard_manager.hazards):
            hz_xy = hazard_positions[i, :2]
            total = total + h.calculate_cost(
                agent_xy=agent_xy,
                hazard_xy=hz_xy,
                cost_scaler=self._cost_scaler,
                contact_geom1=ids1,
                contact_geom2=ids2,
                contact_dist=dist,
                ncon=ncon,
                agent_geom_ids=self._agent_geom_ids
            )
        return total

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
        agent_z_angle = data.qpos[2]  # z_hinge_angle
        cos_a = jp.cos(agent_z_angle)
        sin_a = jp.sin(agent_z_angle)

        # World-frame relative XY vector to goal
        world_dx_goal = rel_goal_pos_3d_world[0]
        world_dy_goal = rel_goal_pos_3d_world[1]

        # Transform world-frame relative vector to agent's local frame
        agent_centric_dx_goal = world_dx_goal * cos_a + world_dy_goal * sin_a
        agent_centric_dy_goal = -world_dx_goal * sin_a + world_dy_goal * cos_a

        # 3. Create compass observation (agent-centric)
        agent_centric_rel_goal_xy = jp.array([agent_centric_dx_goal, agent_centric_dy_goal])
        goal_comp = agent_centric_rel_goal_xy / (safe_norm(agent_centric_rel_goal_xy) + 1e-8)

        # 4. Create Safety-Gymnasium style Lidars with configurable bins
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
        # Pad hazard_mocap_ids to ensure we can process them all
        hazard_mocap_ids_array = jp.array(self._hazard_mocap_ids + [-1] * (8 - len(self._hazard_mocap_ids)))[:8]
        init_carry = (hazard_lidar_obs, agent_pos, agent_z_angle, cos_a, sin_a)
        (hazard_lidar_obs, _, _, _, _), _ = jax.lax.scan(process_hazard_lidar, init_carry, hazard_mocap_ids_array)

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

        # Compute compasses for all hazard mocap indices
        hazard_mocap_ids_for_compass = jp.array(self._hazard_mocap_ids + [-1] * (8 - len(self._hazard_mocap_ids)))[:8]
        hazard_compasses = jax.vmap(compute_compass_for_hazard)(hazard_mocap_ids_for_compass)

        # Flatten to get (16,) shape for 8 hazards
        hazard_compasses_flat = hazard_compasses.flatten()

        # Build observation with separate goal and hazard lidars plus individual hazard compasses
        obs = jp.concatenate([
            accelerometer,  # (3,)
            velocimeter,  # (3,)
            gyro,  # (3,)
            magnetometer,  # (3,)
            goal_lidar_obs,  # (16,) - Goal Lidar
            hazard_lidar_obs,  # (16,) - Hazard Lidar
            goal_comp,  # (2,) - Goal compass
            hazard_compasses_flat,  # (16,) - Individual hazard compasses (8 hazards * 2 each)
        ])

        return obs

    def _get_metrics(self, data: mjx.Data, reward: jp.ndarray, cost: jp.ndarray,
                     dist_goal: jp.ndarray, last_dist_goal: jp.ndarray, ctrl_cost: jp.ndarray,
                     goals_reached_count: jp.ndarray, goals_per_episode: jp.ndarray,
                     goals_per_step: jp.ndarray) -> Dict:
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
            'goals_reached_count': jp.float32(goals_reached_count),
            'goals_per_episode': jp.float32(goals_per_episode),
            'goals_per_step': jp.float32(goals_per_step),
        }

    @property
    def observation_size(self) -> int:
        """Returns the size of the observation vector."""
        return (
                12 +  # Sensor data (3 each for accel, vel, gyro, mag)
                self._lidar_num_bins * 2 +  # Goal and hazard lidars
                2 +  # Goal compass
                self._num_hazards * 2  # Hazard compasses
        )


def SafePointGoal_8Cubes(**kwargs):
    """SafePointGoal with 8 cube hazards."""
    return SafePointGoal(kwargs)


def SafePointGoal_12Cubes(**kwargs):
    """SafePointGoal with 12 cube hazards."""
    config = default_config()
    config.hazards.specs = [
        {"type": "cube", "count": 12},
    ]
    config = config_merge(config, kwargs)
    return SafePointGoal(config)


def SafePointGoal_12Cylinders(**kwargs):
    """SafePointGoal with 12 cylinder hazards."""
    config = default_config()
    config.goals.type = 'cylinder'
    config.goals.count = 2
    config.goals.size = 0.2
    config.goals.height = 0.2
    config.hazards.specs = [
        {"type": "cylinder", "count": 12, "size": 0.3, "height": 0.01, "collidable": False},
        {"type": "outer_wall", "offset": 0.5, "thickness": 0.06, "height": 0.1, "collidable": True, "fixed": True},
    ]
    config = config_merge(config, kwargs)
    return SafePointGoal(config)


def SafePointGoal_MixedHazards(**kwargs):
    """SafePointGoal with mixed hazard types: 5 cubes + 7 cylinders."""
    config = default_config()
    config.goals.type = 'cylinder'
    config.goals.count = 2
    config.goals.size = 0.2
    config.goals.height = 0.2
    config.hazards.specs = [
        {"type": "cube", "count": 3, "size": 0.3, "height": 0.01, "collidable": False},
        {"type": "cube", "count": 2, "size": 0.2, "height": 0.5, "collidable": True},
        {"type": "cylinder", "count": 4, "size": 0.4, "height": 0.01, "collidable": False},
        {"type": "cylinder", "count": 3, "size": 0.2, "height": 0.4, "collidable": True},
        {"type": "outer_wall", "offset": 0.5, "thickness": 0.06, "height": 0.1, "collidable": True, "fixed": True},
    ]
    config = config_merge(config, kwargs)
    return SafePointGoal(config)
