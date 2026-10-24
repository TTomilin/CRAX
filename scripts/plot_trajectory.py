#!/usr/bin/env python3
"""Plot the trajectory of a trained policy over a top-down view of the arena.

Loads a saved policy, runs a single episode, records the agent's (x, y)
position every step and draws it as a line on top of an orthographic top-down
render of the environment. The render shows the scene as laid out at the
start of the episode (hazards, goals, walls), so hazard placement is the one
sampled for this particular episode. Hazards that move during the episode
(e.g. gremlins) get their paths overlaid, and every goal position the agent
was assigned is marked in order.

Usage:
    python scripts/plot_trajectory.py \
        --checkpoint safe_goal_point_Level_1_ppo_seed0_20260916_143548_544422 \
        --env safe_goal_point --level 1

    # Without a checkpoint the point agent drives in circles.
    python scripts/plot_trajectory.py --env safe_goal_point --level 2 --seed 3

`--checkpoint` may be a run directory (the latest step is used), a specific
step directory, or either of those relative to `models/`. Vision checkpoints
(pixel observations) are replayed with the same GPU-rendered camera, resolution
and frame stack they were trained with, inferred from the checkpoint config.
"""
import argparse
import importlib
import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

os.environ.setdefault("MUJOCO_GL", "egl")
# Leave GPU headroom for MJWarp when replaying vision checkpoints (must be set before JAX initialises).
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import jax
import jax.numpy as jnp
import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from crax import envs

ROOT_DIR = Path(__file__).resolve().parent.parent

# Checkpoint config file -> module providing `load_policy`.
_CHECKPOINT_LOADERS = {
    "ppo_network_config.json": "crax.training.agents.ppo.checkpoint",
    "sac_network_config.json": "crax.training.agents.sac.checkpoint",
}


# Fixed view half-extents (around the origin) for envs whose plots get compared side by side.
_DEFAULT_EXTENTS = {
    "safe_circle_point": 1.9,
}


def resolve_checkpoint(checkpoint: str) -> Path:
    """Returns the step directory for `checkpoint` (run dir -> latest step)."""
    path = Path(checkpoint)
    if not path.exists():
        path = ROOT_DIR / "models" / checkpoint
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    path = path.resolve()
    if any((path / f).exists() for f in _CHECKPOINT_LOADERS):
        return path
    steps = sorted(p for p in path.iterdir() if p.is_dir() and p.name.isdigit())
    if not steps:
        raise FileNotFoundError(f"No checkpoint steps found in {path}")
    return steps[-1]


def read_checkpoint_config(path: Path):
    """Returns (loader module, observation_size) for a checkpoint step directory."""
    for config_fname, module in _CHECKPOINT_LOADERS.items():
        if (path / config_fname).exists():
            return module, json.loads((path / config_fname).read_text())["observation_size"]
    raise ValueError(f"Unrecognised checkpoint format in {path} "
                     f"(expected one of {list(_CHECKPOINT_LOADERS)})")


def vision_settings(obs_size) -> Optional[Dict]:
    """Recovers the GpuPixelObservationWrapper kwargs from a checkpoint's observation_size (None if state-based)."""
    if not isinstance(obs_size, dict):
        return None
    pixel_keys = [k for k in obs_size if k.startswith("pixels/")]
    if not pixel_keys:
        return None
    height, width, channels = obs_size[pixel_keys[0]]
    return dict(cameras=[k.split("/", 1)[1] for k in pixel_keys], height=height, width=width,
                obs_mode="pixels+state" if "state" in obs_size else "pixels", frame_stack=channels // 3)


def algorithm_from_run_name(path: Path) -> str:
    """Parses the algorithm from a `<env>_Level_<n>_<alg>_seed<k>_<timestamp>` run directory name."""
    for name in (path.name, path.parent.name):
        m = re.search(r"_Level_\d+_(.+?)_seed\d+_", name)
        if m:
            return m.group(1)
    return "ppo"


def load_policy(path: Path, module: str, deterministic: bool, vision: Optional[Dict]) -> Callable:
    kwargs = {}
    if vision is not None:
        from crax.training.run_utils import make_vision_network_factory

        state_key = "state" if vision["obs_mode"] == "pixels+state" else ""
        kwargs["network_factory"] = make_vision_network_factory(
            algorithm_from_run_name(path), policy_obs_key=state_key, value_obs_key=state_key)
    print(f"Loading policy from {path}")
    return importlib.import_module(module).load_policy(path, deterministic=deterministic, **kwargs)


def make_circle_policy() -> Callable:
    """Point agent: full thrust with a constant turn, for testing without a checkpoint."""

    def circle_policy(obs, key):
        del obs, key
        return jnp.array([1.0, 0.6]), {}

    return circle_policy


def unwrap(env):
    """Strips wrappers (UnifiedEnvAdapter etc.) to get the task env."""
    while hasattr(env, "env"):
        env = env.env
    return env


def run_episode(env, policy: Callable, episode_length: int, seed: int, batched: bool) -> Dict[str, np.ndarray]:
    """Runs one episode and records agent/hazard/goal positions at every step.

    `batched` marks a single-world vectorised env (the pixel-obs wrapper needs one), whose state
    carries a leading batch dimension of 1 that is stripped from everything recorded.
    """
    base = unwrap(env)
    agent_body = getattr(base, "agent_body_index", 1)
    hazard_ids = np.asarray(getattr(base, "_hazard_mocap_ids", []), dtype=np.int32)
    goal_ids = np.asarray(getattr(base, "_goal_mocap_ids", []), dtype=np.int32)

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    policy_fn = jax.jit(policy)

    key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
    state = reset_fn(reset_key)

    def unbatch(tree):
        return jax.tree.map(lambda x: x[0], tree) if batched else tree

    def scalar(x):
        return float(np.asarray(x).reshape(-1)[0])

    first_pipeline_state = unbatch(state.pipeline_state)

    def positions(ps):
        ps = unbatch(ps)
        mocap = np.asarray(ps.mocap_pos) if ps.mocap_pos.size else np.zeros((0, 3))
        return (np.asarray(ps.xpos[agent_body, :2]),
                mocap[hazard_ids, :2] if hazard_ids.size else np.zeros((0, 2)),
                mocap[goal_ids, :2] if goal_ids.size else np.zeros((0, 2)))

    agent, hazards, goals, costs = [], [], [], [0.0]
    a, h, g = positions(state.pipeline_state)
    agent.append(a), hazards.append(h), goals.append(g)
    ep_return, ep_cost = 0.0, 0.0

    for _ in range(episode_length):
        key, action_key = jax.random.split(key)
        action, _ = policy_fn(state.obs, action_key)
        state = step_fn(state, action)
        a, h, g = positions(state.pipeline_state)
        agent.append(a), hazards.append(h), goals.append(g)
        cost = scalar(state.info.get("cost", state.metrics.get("cost", 0.0)))
        costs.append(cost)
        ep_return += scalar(state.reward)
        ep_cost += cost
        if scalar(state.done):
            break

    print(f"Episode: steps={len(agent) - 1} return={ep_return:.2f} cost={ep_cost:.2f}")
    return {
        "first_pipeline_state": first_pipeline_state,
        "agent": np.stack(agent),        # (T, 2)
        "hazards": np.stack(hazards),    # (T, H, 2)
        "goals": np.stack(goals),        # (T, G, 2)
        "costs": np.asarray(costs),      # (T,)
        "return": ep_return,
        "cost": ep_cost,
    }


def render_top_down(mj_model: mujoco.MjModel, pipeline_state, center: np.ndarray, view_height: float,
                    width: int, height: int, hide_agent: bool, agent_body: int) -> np.ndarray:
    """Orthographic render looking straight down; image x = world +x, image up = world +y.

    The image spans `view_height` world units vertically and
    `view_height * width / height` horizontally, centred at `center`.
    """
    mj_model.vis.global_.offwidth = max(mj_model.vis.global_.offwidth, width)
    mj_model.vis.global_.offheight = max(mj_model.vis.global_.offheight, height)

    data = mujoco.MjData(mj_model)
    data.qpos[:] = np.asarray(pipeline_state.qpos)
    data.qvel[:] = np.asarray(pipeline_state.qvel)
    if mj_model.nmocap > 0:
        data.mocap_pos[:] = np.asarray(pipeline_state.mocap_pos)
        data.mocap_quat[:] = np.asarray(pipeline_state.mocap_quat)
    mujoco.mj_forward(mj_model, data)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [center[0], center[1], 0.0]
    cam.distance = 10.0
    cam.azimuth = 90.0
    cam.elevation = -90.0
    cam.orthographic = 1

    # For an orthographic free camera, vis.global.fovy is the view height in world units.
    saved_fovy, saved_ortho = mj_model.vis.global_.fovy, mj_model.vis.global_.orthographic
    mj_model.vis.global_.fovy = view_height
    mj_model.vis.global_.orthographic = 1
    option = mujoco.MjvOption()
    if hide_agent:
        # Put the agent's geoms in a group that is not rendered.
        agent_geoms = np.where(mj_model.geom_bodyid == agent_body)[0]
        saved_groups = mj_model.geom_group[agent_geoms].copy()
        mj_model.geom_group[agent_geoms] = 5
        option.geomgroup[5] = 0
    try:
        with mujoco.Renderer(mj_model, height=height, width=width) as renderer:
            renderer.update_scene(data, camera=cam, scene_option=option)
            return renderer.render()
    finally:
        mj_model.vis.global_.fovy = saved_fovy
        mj_model.vis.global_.orthographic = saved_ortho
        if hide_agent:
            mj_model.geom_group[agent_geoms] = saved_groups


def view_bounds(episode: Dict, margin: float, extent: Optional[float]):
    """Returns (center, half_size) of the square region to render."""
    if extent is not None:
        return np.zeros(2), extent
    # Hazards include the outer walls (when present), so the whole arena stays in view.
    pts = [episode["agent"], episode["goals"].reshape(-1, 2), episode["hazards"].reshape(-1, 2)]
    pts = np.concatenate(pts)
    lo, hi = pts.min(axis=0) - margin, pts.max(axis=0) + margin
    center = (lo + hi) / 2
    half = (hi - lo).max() / 2
    return center, half


def plot(episode: Dict, image: np.ndarray, center: np.ndarray, half: float, output: Path, color_by: str,
         legend_fontsize: float = 8, legend_ncol: int = 1) -> None:
    """Draws the overlay on an axes that fills the whole figure: no axes, title or margins."""
    extent = [center[0] - half, center[0] + half, center[1] - half, center[1] + half]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, extent=extent, origin="upper", interpolation="bilinear")

    # Moving hazards: draw their paths.
    hazards = episode["hazards"]
    if hazards.size:
        moved = np.linalg.norm(hazards - hazards[:1], axis=-1).max(axis=0) > 1e-3
        for i in np.where(moved)[0]:
            ax.plot(hazards[:, i, 0], hazards[:, i, 1], ls="--", lw=1.2, color="#7a1fa2", alpha=0.8,
                    label="moving hazard path" if i == np.where(moved)[0][0] else None)

    # Agent trajectory, coloured by time or by incurred cost.
    xy = episode["agent"]
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    if color_by == "cost":
        # Cost segments are wider and drawn on top so they stay visible where the path crosses itself.
        # costs[t] is computed on the post-step state, i.e. at xy[t]; colour each segment xy[t] -> xy[t+1] by the
        # cost at its start point so red begins where the agent actually is in the cost region.
        in_cost = episode["costs"][:-1] > 0
        safe_color, cost_color, cost_width = (0.1, 0.35, 0.9), (0.85, 0.1, 0.1), 4.0
        ax.add_collection(LineCollection(segments[~in_cost], colors=[safe_color], linewidths=2.2, zorder=3))
        ax.add_collection(LineCollection(segments[in_cost], colors=[cost_color], linewidths=cost_width,
                                         capstyle="round", zorder=4))
        ax.plot([], [], color=safe_color, lw=2.2, label="trajectory")
        ax.plot([], [], color=cost_color, lw=cost_width, label="trajectory (cost > 0)")
    else:
        lc = LineCollection(segments, cmap="viridis", linewidths=2.2)
        lc.set_array(np.arange(len(segments)))
        ax.add_collection(lc)

    ax.scatter(*xy[0], s=90, marker="o", facecolor="white", edgecolor="black", zorder=5, label="start")
    ax.scatter(*xy[-1], s=110, marker="X", facecolor="black", edgecolor="white", zorder=5, label="end")

    # Goals, numbered in the order the agent reached them. A goal slot jumping to a new position means the
    # old position was reached at that step; goals still pending at the end are drawn hollow, unnumbered.
    goals = episode["goals"]
    # Goals that never move are scenery (e.g. the visual-only markers of the circle task): leave them to the render.
    if goals.size and np.abs(goals - goals[:1]).max() > 1e-3:
        reached = [(t, slot) for t in range(1, len(goals)) for slot in range(goals.shape[1])
                   if np.linalg.norm(goals[t, slot] - goals[t - 1, slot]) > 1e-3]
        badge = dict(ha="center", va="center", fontsize=9, fontweight="bold", zorder=6)
        for n, (t, slot) in enumerate(reached, 1):
            ax.annotate(str(n), goals[t - 1, slot], color="white",
                        bbox=dict(boxstyle="circle,pad=0.2", fc="#2e7d32", ec="white", lw=1), **badge)
        for g in goals[-1]:
            ax.annotate("  ", g, bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="#2e7d32", lw=1.5), **badge)
        ax.plot([], [], marker="o", ls="", color="#2e7d32", label=f"goal reached, in order ({len(reached)})")
        ax.plot([], [], marker="o", ls="", mfc="white", mec="#2e7d32", label="goal not reached")
        print(f"Goals reached: {len(reached)}")

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.legend(loc="upper right", fontsize=legend_fontsize, ncol=legend_ncol, framealpha=0.85, columnspacing=1.0,
              handletextpad=0.4, borderaxespad=0.3, markerscale=max(1.0, legend_fontsize / 8))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved trajectory plot to {output}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=None, help="Run or step directory (absolute or under models/)")
    p.add_argument("--env", required=True, help="Environment name, e.g. safe_goal_point")
    p.add_argument("--level", type=int, default=None, choices=[1, 2, 3], help="Difficulty level")
    p.add_argument("--seed", type=int, default=0, help="Seed for the episode (sets hazard layout)")
    p.add_argument("--episode_length", type=int, default=None, help="Max steps (default: env default)")
    p.add_argument("--stochastic", action="store_true", help="Sample actions instead of using the mean")
    p.add_argument("--extent", type=float, default=None,
                   help="Half-size of the rendered square around the origin (default: fixed per env where set in "
                        "_DEFAULT_EXTENTS, else fit to episode)")
    p.add_argument("--margin", type=float, default=0.5, help="Padding around the episode when auto-fitting")
    p.add_argument("--resolution", type=int, default=1024, help="Top-down render resolution in pixels")
    p.add_argument("--show_agent", action="store_true", help="Render the agent at its start pose")
    p.add_argument("--legend_fontsize", type=float, default=8, help="Legend font size")
    p.add_argument("--legend_ncol", type=int, default=1, help="Number of legend columns")
    p.add_argument("--color_by", choices=["time", "cost"], default="time",
                   help="Colour the trajectory by step or highlight steps that incurred cost")
    p.add_argument("--output", default=None, help="Output image path (default: visualizations/...)")
    args = p.parse_args()

    vision = None
    if args.checkpoint:
        ckpt_path = resolve_checkpoint(args.checkpoint)
        module, ckpt_obs_size = read_checkpoint_config(ckpt_path)
        vision = vision_settings(ckpt_obs_size)

    if vision is None:
        env = envs.get_environment(args.env, level=args.level)
        episode_length = args.episode_length or getattr(unwrap(env), "episode_length", 1000)
    else:
        print(f"Vision checkpoint: {vision}")
        episode_length = args.episode_length or getattr(
            envs.get_environment(args.env, level=args.level), "episode_length", 1000)
        # GpuPixelObservationWrapper needs the MJX backend and a (here single-world) batched env.
        env = envs.create(args.env, level=args.level, episode_length=episode_length, auto_reset=False,
                          batch_size=1, vision=True, vision_kwargs=dict(vision, num_envs=1), backend="mjx")
    base = unwrap(env)

    if args.checkpoint:
        if vision is None and list(np.atleast_1d(ckpt_obs_size)) != list(np.atleast_1d(env.observation_size)):
            raise ValueError(f"Checkpoint observation_size={ckpt_obs_size} does not match the environment's "
                             f"{env.observation_size}; check --env and --level")
        policy = load_policy(ckpt_path, module, deterministic=not args.stochastic, vision=vision)
        # A step directory is named after its run, so outputs from different steps don't collide.
        policy_name = (f"{ckpt_path.parent.name}_step{int(ckpt_path.name)}" if ckpt_path.name.isdigit()
                       and Path(args.checkpoint).name.isdigit() else Path(args.checkpoint).name)
    else:
        print("No checkpoint given, using a circle-driving policy.")
        policy = make_circle_policy()
        policy_name = f"{args.env}{'_Level_' + str(args.level) if args.level else ''}_circle_policy"

    episode = run_episode(env, policy, episode_length, args.seed, batched=vision is not None)

    center, half = view_bounds(episode, args.margin, args.extent or _DEFAULT_EXTENTS.get(args.env))
    image = render_top_down(env.sys.mj_model, episode["first_pipeline_state"], center, 2 * half,
                            args.resolution, args.resolution, hide_agent=not args.show_agent,
                            agent_body=getattr(base, "agent_body_index", 1))

    output = Path(args.output) if args.output else (
        ROOT_DIR / "visualizations" / args.env /
        f"{policy_name}_seed{args.seed}_trajectory_{args.color_by}.png")
    plot(episode, image, center, half, output, args.color_by, args.legend_fontsize, args.legend_ncol)


if __name__ == "__main__":
    main()
