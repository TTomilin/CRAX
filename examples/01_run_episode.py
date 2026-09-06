#!/usr/bin/env python3
"""Minimal CRAX example: run one episode with random actions and save an MP4.

Usage:
    python examples/01_run_episode.py
    python examples/01_run_episode.py --env_name safe_circle_point --level 2
    python examples/01_run_episode.py --steps 500 --out videos/rollout.mp4
"""
import argparse
import os
from pathlib import Path

# MuJoCo needs an OpenGL backend for offscreen rendering. EGL works headless;
# set MUJOCO_GL yourself (e.g. to "osmesa" or "glfw") to override.
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v3 as iio
import jax
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from crax import envs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env_name", type=str, default="safe_goal_point",
                   help="Environment name from the CRAX registry")
    p.add_argument("--level", type=int, default=1, choices=[1, 2, 3],
                   help="Difficulty level")
    p.add_argument("--steps", type=int, default=300, help="Max steps in the episode")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--camera", type=str, default=None,
                   help="Camera name to render from (default: the env's default camera)")
    p.add_argument("--width", type=int, default=640, help="Video width in pixels")
    p.add_argument("--height", type=int, default=480, help="Video height in pixels")
    p.add_argument("--fps", type=int, default=100,
                   help="Video frame rate (default: real time, i.e. 1 / env.dt)")
    p.add_argument("--out", type=str, default=None,
                   help="Output MP4 path (default: videos/<env_name>_level<level>_random.mp4)")
    p.add_argument("--no_overlay", action="store_true",
                   help="Do not draw the reward/cost overlay on the frames")
    return p.parse_args()


def overlay_metrics(frames, returns, costs):
    """Draws the running return and cost onto each rendered frame."""
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except (OSError, IOError):
        font = ImageFont.load_default()

    annotated = []
    for frame, ret, cost in zip(frames, returns, costs):
        img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        draw = ImageDraw.Draw(img)
        lines = [(f"Return: {ret:.2f}", (50, 220, 50)), (f"Cost: {cost:.2f}", (230, 60, 60))]
        for i, (text, color) in enumerate(lines):
            # Stroke keeps the text readable on both light and dark scenes.
            draw.text((10, 10 + 26 * i), text, font=font, fill=color,
                      stroke_width=2, stroke_fill=(0, 0, 0))
        annotated.append(np.array(img))
    return annotated


def main() -> None:
    args = parse_args()

    # 1. Create the environment. `level` picks the difficulty configuration.
    env = envs.get_environment(args.env_name, level=args.level)
    print(f"env={args.env_name} level={args.level} "
          f"obs_size={env.observation_size} action_size={env.action_size}")

    # 2. JIT the reset/step functions once; every call after the first is fast.
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    # 3. Roll out a single episode with uniform random actions in [-1, 1].
    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng = jax.random.split(rng)
    state = reset_fn(reset_rng)

    trajectory = [state.pipeline_state]
    total_reward, total_cost = 0.0, 0.0
    # Running totals per frame, so the overlay matches what is on screen.
    returns, costs = [total_reward], [total_cost]

    for _ in range(args.steps):
        rng, action_rng = jax.random.split(rng)
        action = jax.random.uniform(action_rng, (env.action_size,), minval=-1.0, maxval=1.0)
        state = step_fn(state, action)

        trajectory.append(state.pipeline_state)
        total_reward += float(state.reward)
        # Safety cost is reported in state.info (and mirrored in state.metrics).
        total_cost += float(state.info.get("cost", state.metrics.get("cost", 0.0)))
        returns.append(total_reward)
        costs.append(total_cost)

        if bool(state.done):
            break

    print(f"episode finished: steps={len(trajectory) - 1} "
          f"return={total_reward:.2f} cost={total_cost:.2f}")

    # 4. Render the trajectory and write it as an MP4 at real-time speed.
    frames = env.render(trajectory, height=args.height, width=args.width, camera=args.camera)
    if not args.no_overlay:
        frames = overlay_metrics(frames, returns, costs)
    fps = args.fps if args.fps is not None else int(round(1.0 / float(env.dt)))

    out_path = Path(args.out) if args.out else Path("videos") / f"{args.env_name}_level{args.level}_random.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_path, np.stack(frames).astype(np.uint8), fps=fps)
    print(f"saved video: {out_path} ({len(frames)} frames @ {fps} fps)")


if __name__ == "__main__":
    main()
