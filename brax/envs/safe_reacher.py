from typing import List, Optional

import jax
import mujoco
from jax import numpy as jp

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.envs.env_utils import generate_goal_xml_from_base
from brax.envs.goals import GoalManager
from brax.envs.hazards import HazardManager
from brax.io import mjcf


class SafeReacher(PipelineEnv):
    """Reacher with randomly placed flat hazards that impose a per-step cost.

    - Hazards are pass-through (non-collidable in physics) and conceptually;
      compute costs via proximity checks, not contacts.
    - On every reset, a new random layout of hazards is sampled within the arm's
      reach radius.
    - If the arm intersects multiple hazards simultaneously, costs accumulate
      additively.
    """

    def __init__(
            self,
            backend: str = 'mjx',
            num_hazards: int = 6,
            hazard_types: Optional[List[str]] = None,
            hazard_radius: float = 0.035,
            rect_half_extent: float = 0.03,
            hazard_height: float = 0.01,
            reach_radius: float = 0.27,
            cost_scale: float = 1.0,
            reward_scale: float = 10.0,
            samples_per_link: int = 5,
            lidar_bins: int = 16,
            lidar_max_dist: float = 0.30,
            lidar_alias: bool = True,
            **kwargs,
    ):
        # Config for hazards and env geometry
        self._num_hazards = num_hazards
        self._hazard_types = hazard_types or ['cylinder', 'rect']
        self._hazard_radius = hazard_radius
        self._rect_half_extent = rect_half_extent
        self._hazard_height = hazard_height
        self._reach_radius = reach_radius
        self._cost_scale = cost_scale
        self._reward_scale = reward_scale
        self._samples_per_link = samples_per_link
        self._lidar_bins = lidar_bins
        self._lidar_max_dist = lidar_max_dist
        self._lidar_alias = lidar_alias

        # Build hazards as MJCF geoms/bodies (mocap)
        cyl_r = self._hazard_radius
        rect_hx = self._rect_half_extent
        rect_hy = self._rect_half_extent

        # Create HazardManager and add hazards alternating by type
        hazard_manager = HazardManager()
        types = self._hazard_types
        if len(types) == 0:
            types = ['cylinder']
        for i in range(self._num_hazards):
            t = types[i % len(types)]
            if t == 'cylinder':
                size = cyl_r
            elif t in ('rect', 'cube'):
                size = (rect_hx, rect_hy)
            else:
                raise ValueError(f"Unknown hazard type '{t}'")

            hazard_manager.add_hazards(t, 1, positions=[(0.0, 0.0, self._hazard_height)], size=size,
                                       height=self._hazard_height, collidable=False, fixed=False, density=1.0)

        # Build a SphereGoal via GoalManager
        goal_manager = GoalManager()
        goal_manager.add_goals('sphere', 1, positions=[(0.0, 0.0, 0.01)], size=0.009, height=0.0)

        # Build XML from base reacher, goal, and hazards
        base_name = 'reacher.xml'
        xml_path = generate_goal_xml_from_base(base_name, goal_manager=goal_manager, hazard_manager=hazard_manager)

        try:
            mj_model = mujoco.MjModel.from_xml_path(xml_path)
        finally:
            import os
            if os.path.exists(xml_path):
                os.unlink(xml_path)

        # resolve our goal body/geom created via GoalManager
        self._target_body_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "goal1"))

        # cache target mocap id and default z for placement on reset
        self._target_mocap_id = int(mj_model.body_mocapid[self._target_body_id])
        self._target_z = float(mj_model.body_pos[self._target_body_id][2])

        # cache tip body id and offset (for velocity calculation)
        self._tip_body_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "body1"))
        self._tip_offset = jp.array([0.11, 0.0, 0.0], dtype=jp.float32)

        # sphere geom: geom_size[0] is the radius
        target_geom_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "goal1"))
        self._goal_radius = float(mj_model.geom_size[target_geom_id][0])

        # Load into Brax system
        sys = mjcf.load_model(mj_model)

        n_frames = 2
        if backend in ['spring', 'positional']:
            sys = sys.tree_replace({'opt.timestep': 0.005})
            sys = sys.replace(actuator=sys.actuator.replace(gear=jp.array([25.0, 25.0])))
            n_frames = 4

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        # Cache hazard mocap ids and static params
        self._hazard_mocap_ids = []
        hazard_is_rect = []
        hazard_radii = []
        hazard_rect_hw = []

        for idx, hz in enumerate(hazard_manager.hazards, start=1):
            # Resolve mocap id for body name
            b = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, f"hazard{idx}")
            mid = int(mj_model.body_mocapid[b])
            if mid < 0:
                raise RuntimeError("Hazard body is not mocap")
            self._hazard_mocap_ids.append(mid)
            if hasattr(hz, 'hazard_type') and hz.hazard_type == 'cylinder':
                hazard_is_rect.append(False)
                hazard_radii.append(float(hz.size))
                hazard_rect_hw.append(jp.array([0.0, 0.0]))
            else:
                hazard_is_rect.append(True)
                # hz.size for rect is (hx, hy)
                sx, sy = hz.size if isinstance(hz.size, tuple) else (rect_hx, rect_hy)
                hazard_radii.append(0.0)
                hazard_rect_hw.append(jp.array([float(sx), float(sy)]))

        self._hazard_is_rect = jp.array(hazard_is_rect, dtype=jp.bool_)
        self._hazard_radii = jp.array(hazard_radii, dtype=jp.float32)
        self._hazard_half_extents = jp.stack(hazard_rect_hw) if len(hazard_rect_hw) > 0 else jp.zeros((0, 2))

    # --------------------------- Core RL API ---------------------------

    def reset(self, rng: jax.Array) -> State:
        # Split RNG for q/qd init and hazard/target sampling
        rng, rng1, rng2, rng_haz, rng_t = jax.random.split(rng, 5)

        # Randomize initial state like base reacher
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=-0.1, maxval=0.1
        )
        qd = jax.random.uniform(
            rng2, (self.sys.qd_size(),), minval=-0.005, maxval=0.005
        )

        # --- 1) sample the goal ---
        def _sample_goal(key):
            k1, k2 = jax.random.split(key)
            dist = 0.2 * jax.random.uniform(k1)
            ang = 2.0 * jp.pi * jax.random.uniform(k2)
            return jp.array([dist * jp.cos(ang), dist * jp.sin(ang)], dtype=jp.float32)

        rng_t, sub = jax.random.split(rng_t)
        goal_xy = _sample_goal(sub)
        jax.debug.print("goal_xy = {g}", g=goal_xy, ordered=True)
        goal_r = jp.asarray(self._goal_radius, dtype=jp.float32)

        # --- 2) place hazards conditioned on goal + non-overlap ---
        n_h = len(self._hazard_mocap_ids)

        def _sample_points_in_disc(key, n):
            k1, k2 = jax.random.split(key)
            rs = self._reach_radius * jp.sqrt(jax.random.uniform(k1, (n,), minval=0.05, maxval=1.0))
            angs = 2.0 * jp.pi * jax.random.uniform(k2, (n,))
            xs = rs * jp.cos(angs)
            ys = rs * jp.sin(angs)
            return jp.stack([xs, ys], axis=1)  # (n,2)

        # --- 1) sample hazard positions first, but ensure hazards don't overlap ---
        hazard_margin = jp.asarray(0.005, dtype=jp.float32)

        def _haz_keepout_radius(i: int) -> jax.Array:
            # cylinder: use radius; rect: use circumscribed circle radius
            hx = self._hazard_half_extents[i, 0]
            hy = self._hazard_half_extents[i, 1]
            rect_r = jp.sqrt(hx * hx + hy * hy)
            return jp.where(self._hazard_is_rect[i], rect_r, self._hazard_radii[i])

        hazards_xy = jp.zeros((n_h, 2), dtype=jp.float32)

        # Precompute keepout radii for ALL hazards once (shape: (n_h,))
        haz_r = jax.vmap(_haz_keepout_radius)(jp.arange(n_h))

        def ok_against_prev(i, cand_xy, placed_xy):
            # Compare against all slots, but only enforce for j < i
            d = jp.sqrt(jp.sum((placed_xy - cand_xy[None, :]) ** 2, axis=1) + 1e-8)  # (n_h,)
            mask = jp.arange(n_h) < i  # (n_h,)
            min_d = haz_r[i] + haz_r + hazard_margin  # (n_h,)
            # For j>=i, mask=False so condition is auto-true
            return jp.all(jp.logical_or(~mask, d > min_d))

        def ok_against_goal(i, cand_xy):
            # keep hazard away from the goal (hazard radius + goal radius + small margin)
            d = jp.sqrt(jp.sum((cand_xy - goal_xy) ** 2) + 1e-8)
            return d > (haz_r[i] + goal_r + hazard_margin)

        def place_one(i, carry):
            rng_k, placed = carry

            rng_k, sub0 = jax.random.split(rng_k)
            cand0 = _sample_points_in_disc(sub0, 1)[0]
            ok0 = ok_against_prev(i, cand0, placed)

            def attempt(_, st):
                rng_t, ok, cur = st
                rng_t, sub = jax.random.split(rng_t)
                cand = _sample_points_in_disc(sub, 1)[0]
                ok_cand = ok_against_prev(i, cand, placed) & ok_against_goal(i, cand)

                take = (~ok) & ok_cand
                cur = jax.lax.select(take, cand, cur)
                ok = ok | ok_cand
                return (rng_t, ok, cur)

            rng_k, ok, chosen = jax.lax.fori_loop(0, 400, attempt, (rng_k, ok0, cand0))
            placed = placed.at[i].set(chosen)
            return (rng_k, placed)

        rng_haz, hazards_xy = jax.lax.fori_loop(0, n_h, place_one, (rng_haz, hazards_xy))

        hazards_pos = jp.concatenate(
            [hazards_xy, jp.full((n_h, 1), self._hazard_height, dtype=jp.float32)], axis=1
        )

        # --- 3) init pipeline, then apply hazard + target mocaps ---
        pipeline_state = self.pipeline_init(q, qd)

        # set hazards
        ids = jp.array(self._hazard_mocap_ids, dtype=jp.int32)
        mpos = pipeline_state.mocap_pos.at[ids].set(hazards_pos)
        # set target (mocap sphere)
        tgt_pos = jp.array([goal_xy[0], goal_xy[1], self._target_z], dtype=jp.float32)
        mpos = mpos.at[self._target_mocap_id].set(tgt_pos)
        pipeline_state = pipeline_state.replace(mocap_pos=mpos)

        # compute initial distance-to-goal and store for progress reward
        tip_pos, _, target_pos = self._tip_target(pipeline_state)
        dist = jp.sum(jp.abs(tip_pos[:2] - target_pos[:2]))

        jax.debug.print(
            "goal sampled={g} | mocap={m} | target_pos={x}",
            g=goal_xy,
            m=pipeline_state.mocap_pos[self._target_mocap_id][:2],
            x=target_pos,
            ordered=True,
        )

        obs = self._get_obs(pipeline_state, hazards_pos)
        reward, done, zero = jp.zeros(3)

        # Initial per-step cost (zero at reset)
        metrics = {
            'reward_dist': zero,
            'reward_ctrl': zero,
            'cost': zero,
            'dist': dist,
            'progress': zero,
        }
        info = {
            'hazard_positions': hazards_pos,
            'min_dist': dist,
            'cost': zero,
        }

        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        hazard_positions = state.info['hazard_positions']
        obs = self._get_obs(pipeline_state, hazard_positions)

        # --- distance to goal ---
        tip_pos, _, target_pos = self._tip_target(pipeline_state)
        dist = jp.sum(jp.abs(tip_pos[:2] - target_pos[:2]))

        jax.debug.print(
            "step dist={d} | tip_pos={t} | target_pos={x}",
            d=dist,
            t=tip_pos,
            x=target_pos,
            ordered=True,
        )

        min_dist = jp.asarray(state.info['min_dist'], dtype=jp.float32)

        # proportional reward for getting closer to the goal
        progress = min_dist - dist
        reward = jp.maximum(progress, 0.0) * jp.asarray(self._reward_scale, jp.float32)

        # terminate + bonus if touching goal
        success = dist <= jp.asarray(self._goal_radius, dtype=jp.float32)
        done = success.astype(jp.float32)

        # --- safety cost ---
        cost = self._calculate_safety_cost(pipeline_state, hazard_positions)

        # metrics/info
        state.metrics.update(
            dist=dist,
            progress=progress,
            cost=cost,
        )
        new_min_dist = jp.minimum(min_dist, dist)
        info = dict(state.info)
        info['cost'] = cost
        info['min_dist'] = new_min_dist  # update for next step

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            info=info,
        )

    # --------------------------- Observations ---------------------------

    def _get_obs(self, pipeline_state: base.State, hazard_positions: jax.Array) -> jax.Array:
        """Observation: original reacher obs + hazard lidar bins.

        Lidar is agent-centric using theta1 (first joint angle) as the 'heading'.
        """
        theta = pipeline_state.q[:2]
        theta1 = theta[0]

        tip_pos, tip_vel, target_pos = self._tip_target(pipeline_state)
        tip_to_target = tip_pos - target_pos
        target_xy = target_pos[:2]

        # ---------------- hazard lidar ----------------
        hazards_xy = hazard_positions[:, :2]
        n_h = hazards_xy.shape[0]

        bins = self._lidar_bins
        max_d = jp.asarray(self._lidar_max_dist, dtype=jp.float32)
        bin_size = (2.0 * jp.pi) / bins

        # rotate world -> agent frame using theta1
        c = jp.cos(theta1)
        s = jp.sin(theta1)

        lidar = jp.zeros((bins,), dtype=jp.float32)

        def body(i, lidar_acc):
            hx, hy = hazards_xy[i, 0], hazards_xy[i, 1]

            # agent-centric rotation (heading = theta1)
            ax = hx * c + hy * s
            ay = -hx * s + hy * c

            dist = jp.sqrt(ax * ax + ay * ay + 1e-8)
            ang = jp.arctan2(ay, ax)
            ang = (ang + 2.0 * jp.pi) % (2.0 * jp.pi)

            # binning
            bfloat = ang / bin_size
            b0 = jp.minimum(jp.floor(bfloat), bins - 1).astype(jp.int32)

            val = jp.maximum(0.0, max_d - dist) / max_d
            val = jp.where(dist > max_d, 0.0, val)

            # primary bin: max pooling
            lidar_acc = lidar_acc.at[b0].set(jp.maximum(lidar_acc[b0], val))

            if self._lidar_alias:
                frac = bfloat - b0.astype(jp.float32)

                b_plus = (b0 + 1) % bins
                b_minus = (b0 - 1 + bins) % bins

                lidar_acc = lidar_acc.at[b_plus].set(jp.maximum(lidar_acc[b_plus], frac * val))
                lidar_acc = lidar_acc.at[b_minus].set(jp.maximum(lidar_acc[b_minus], (1.0 - frac) * val))

            return lidar_acc

        lidar = jax.lax.fori_loop(0, n_h, body, lidar)

        # ---------------- final obs ----------------
        return jp.concatenate([
            jp.cos(theta),
            jp.sin(theta),
            target_xy,  # target x, y from mocap body
            tip_vel[:2],
            tip_to_target,
            lidar,  # (lidar_bins,)
        ])

    def _tip_target(self, pipeline_state: base.State) -> tuple[jax.Array, jax.Array, jax.Array]:
        target_pos = pipeline_state.mocap_pos[self._target_mocap_id]
        tip_pos = (pipeline_state.x.take(self._tip_body_id).pos)
        tip_vel = (
            base.Transform.create(pos=self._tip_offset)
            .do(pipeline_state.xd.take(self._tip_body_id))
            .vel
        )
        return tip_pos, tip_vel, target_pos

    # --------------------------- Hazard logic ---------------------------

    def _calculate_safety_cost(self, pipeline_state: base.State, hazard_positions: jax.Array) -> jax.Array:
        """Binary per-hazard cost: 1 if any sampled arm point is inside the hazard.

        Sum of per-hazard costs each step.
        Use stored static hazard shapes and current hazard positions.
        """
        n_h = hazard_positions.shape[0]
        if n_h == 0:
            return jp.array(0.0, dtype=jp.float32)

        centers = hazard_positions[:, :2]  # (N,2)

        # --- sample arm points in XY ---
        theta1, theta2 = pipeline_state.q[0], pipeline_state.q[1]
        l1, l2 = 0.1, 0.1

        n = max(1, self._samples_per_link)
        frac = jp.linspace(1.0 / (n + 1), n / (n + 1), n)

        link1_pts = jp.stack([l1 * frac * jp.cos(theta1),
                              l1 * frac * jp.sin(theta1)], axis=1)

        link2_base = jp.array([l1 * jp.cos(theta1), l1 * jp.sin(theta1)])
        theta12 = theta1 + theta2
        link2_local = jp.stack([l2 * frac * jp.cos(theta12),
                                l2 * frac * jp.sin(theta12)], axis=1)
        link2_pts = link2_base[None, :] + link2_local

        tip_xy = jp.array([l1 * jp.cos(theta1) + l2 * jp.cos(theta12),
                           l1 * jp.sin(theta1) + l2 * jp.sin(theta12)])

        pts = jp.concatenate([link1_pts, link2_pts, tip_xy[None, :]], axis=0)  # (M,2)

        # --- hazard membership tests (vectorized) ---
        # distances from each point to each hazard center
        d = jp.sqrt(jp.sum((pts[:, None, :] - centers[None, :, :]) ** 2, axis=2) + 1e-8)  # (M,N)

        # cylinder: inside if d <= r
        cyl_inside_any = jp.any(d <= self._hazard_radii[None, :], axis=0).astype(jp.float32)  # (N,)

        # rect (axis-aligned): inside if |dx|<=hx and |dy|<=hy
        dxdy = jp.abs(pts[:, None, :] - centers[None, :, :])  # (M,N,2)
        rect_inside = jp.logical_and(
            dxdy[:, :, 0] <= self._hazard_half_extents[None, :, 0],
            dxdy[:, :, 1] <= self._hazard_half_extents[None, :, 1],
        )
        rect_inside_any = jp.any(rect_inside, axis=0).astype(jp.float32)  # (N,)

        # select by type mask, then sum
        is_rect = self._hazard_is_rect.astype(jp.float32)
        is_cyl = (1.0 - is_rect)

        per_hazard_cost = cyl_inside_any * is_cyl + rect_inside_any * is_rect
        total_cost = jp.sum(per_hazard_cost)

        return jp.asarray(self._cost_scale, dtype=jp.float32) * total_cost
