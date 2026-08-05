"""Visualise egocentric GPU pixel observations vs CPU (MuJoCo) rendering.

Records two videos from the same rollout:
  1. GPU render  — egocentric obs from GpuPixelObservationWrapper (pixelbrax)
  2. CPU render  — same 'vision' camera via MuJoCo's built-in renderer

A third side-by-side video concatenates them for easy comparison.

Run from the repo root:

    python scripts/visualise_ego_obs.py --env safe_goal_point
    python scripts/visualise_ego_obs.py --env safe_goal_point --camera vision_back

Requires imageio and imageio-ffmpeg:
    pip install imageio imageio-ffmpeg
"""

import argparse
import os
import sys

import imageio
import jax
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crax import envs


def parse_args():
    p = argparse.ArgumentParser(description="Visualise egocentric pixel observations")
    p.add_argument("--env", type=str, default="safe_goal_point",
                   help="Environment name (default: safe_goal_point)")
    p.add_argument("--level", type=int, default=0,
                   help="Difficulty level (default: 0)")
    p.add_argument("--camera", type=str, default="vision",
                   help="Camera name from the MuJoCo XML (default: 'vision')")
    p.add_argument("--num_steps", type=int, default=200,
                   help="Number of steps to record (default: 200)")
    p.add_argument("--height", type=int, default=128,
                   help="Render height in pixels (default: 128)")
    p.add_argument("--width", type=int, default=128,
                   help="Render width in pixels (default: 128)")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--fps", type=int, default=30, help="Output video FPS (default: 30)")
    p.add_argument("--output", type=str, default="videos",
                   help="Output directory (default: videos/)")
    return p.parse_args()


def cpu_render_frame(mj_model, cpu_renderer, pipeline_state, camera_name):
    """Render one frame with MuJoCo's CPU renderer from a given pipeline_state."""
    d = mujoco.MjData(mj_model)
    d.qpos[:] = np.asarray(pipeline_state.qpos)
    d.qvel[:] = np.asarray(pipeline_state.qvel)
    if mj_model.nmocap > 0:
        d.mocap_pos[:] = np.asarray(pipeline_state.mocap_pos)
        d.mocap_quat[:] = np.asarray(pipeline_state.mocap_quat)
    mujoco.mj_forward(mj_model, d)
    cpu_renderer.update_scene(d, camera=camera_name)
    return cpu_renderer.render().copy()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Build environments
    # ------------------------------------------------------------------ #
    print(f"Creating '{args.env}' (GPU vision, camera='{args.camera}', "
          f"{args.width}x{args.height}) ...")
    gpu_env = envs.get_environment(
        args.env,
        level=args.level,
        vision=True,
        vision_kwargs=dict(
            camera=args.camera,
            height=args.height,
            width=args.width,
            obs_mode="pixels",
        ),
    )

    # Raw env (no vision wrapper) – used only to access sys for CPU rendering.
    raw_env = envs.get_environment(args.env, level=args.level)

    obs_key = f"pixels/{args.camera}"
    action_size = gpu_env.action_size
    mj_model = raw_env.sys.mj_model

    # MuJoCo CPU renderer at the same resolution for fair comparison.
    cpu_renderer = mujoco.Renderer(mj_model, height=args.height, width=args.width)

    # ------------------------------------------------------------------ #
    # JIT-compile reset / step
    # ------------------------------------------------------------------ #
    rng = jax.random.PRNGKey(args.seed)
    jit_reset = jax.jit(gpu_env.reset)
    jit_step = jax.jit(gpu_env.step)

    print("Compiling (first call may be slow) ...")
    rng, reset_rng = jax.random.split(rng)
    state = jit_reset(reset_rng)

    # ------------------------------------------------------------------ #
    # Rollout
    # ------------------------------------------------------------------ #
    gpu_frames = []
    cpu_frames = []
    num_resets = 0

    print(f"Rolling out {args.num_steps} steps ...")
    for i in range(args.num_steps):
        # GPU egocentric frame
        gpu_frames.append(np.asarray(state.obs[obs_key]))

        # CPU frame from same pipeline_state
        try:
            cpu_frame = cpu_render_frame(
                mj_model, cpu_renderer, state.pipeline_state, args.camera
            )
            cpu_frames.append(cpu_frame)
        except Exception as e:
            # Fall back to a grey placeholder if CPU render fails
            cpu_frames.append(np.full((args.height, args.width, 3), 128, dtype=np.uint8))
            if i == 0:
                print(f"  [warning] CPU render failed: {e}")

        if state.done:
            rng, reset_rng = jax.random.split(rng)
            state = jit_reset(reset_rng)
            num_resets += 1
            continue

        rng, act_rng = jax.random.split(rng)
        action = jax.random.uniform(act_rng, (action_size,), minval=-1.0, maxval=1.0)
        state = jit_step(state, action)

        if (i + 1) % 50 == 0:
            print(f"  step {i + 1}/{args.num_steps}")

    cpu_renderer.close()

    # ------------------------------------------------------------------ #
    # Save videos
    # ------------------------------------------------------------------ #
    os.makedirs(args.output, exist_ok=True)
    stem = f"{args.env}_level{args.level}_{args.camera}"

    gpu_path = os.path.join(args.output, f"{stem}_gpu.mp4")
    cpu_path = os.path.join(args.output, f"{stem}_cpu.mp4")
    side_path = os.path.join(args.output, f"{stem}_comparison.mp4")

    print(f"Saving GPU video  → {gpu_path}")
    imageio.mimwrite(gpu_path, gpu_frames, fps=args.fps, macro_block_size=1)

    print(f"Saving CPU video  → {cpu_path}")
    imageio.mimwrite(cpu_path, cpu_frames, fps=args.fps, macro_block_size=1)

    # Side-by-side: GPU on the left, CPU on the right.
    # Resize CPU frames to match GPU dimensions if needed.
    side_frames = []
    for gf, cf in zip(gpu_frames, cpu_frames):
        # Ensure both are uint8 HxWx3
        if cf.shape != gf.shape:
            from PIL import Image as PILImage
            cf = np.array(
                PILImage.fromarray(cf).resize((gf.shape[1], gf.shape[0]))
            )
        side_frames.append(np.concatenate([gf, cf], axis=1))

    print(f"Saving comparison → {side_path}")
    imageio.mimwrite(side_path, side_frames, fps=args.fps, macro_block_size=1)

    print(f"\nDone. ({num_resets} episode resets, "
          f"frame size {gpu_frames[0].shape[1]}x{gpu_frames[0].shape[0]})")
    print("GPU = pixelbrax (left),  CPU = MuJoCo renderer (right).")


if __name__ == "__main__":
    main()
