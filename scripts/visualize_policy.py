#!/usr/bin/env python3
"""Visualize a trained policy by running a rollout and saving video + snapshots.

Usage:
    python scripts/visualize_policy.py --checkpoint path/to/checkpoint --env safe_point_goal

    # With custom options
    python scripts/visualize_policy.py \
        --checkpoint results/ppo_lag/checkpoint_10000000 \
        --env safe_point_goal \
        --difficulty 2 \
        --episode_length 1000 \
        --output_dir visualizations \
        --snapshot_interval 20 \
        --video_fps 30 \
        --deterministic
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import jax


# Select rendering backend before importing mujoco
def _select_backend() -> str:
    """Select a working MuJoCo rendering backend."""
    import os

    def _try_backend(gl_backend: str) -> bool:
        os.environ["MUJOCO_GL"] = gl_backend
        try:
            import mujoco
            return True
        except Exception:
            return False

    # Respect pre-set MUJOCO_GL if available
    preset = os.environ.get("MUJOCO_GL")
    if preset:
        return preset

    # Try common headless-capable backends
    for backend in ["egl", "osmesa", "glfw"]:
        if _try_backend(backend):
            return backend

    return ""


_select_backend()

from brax import envs
from brax.io import image as brax_image
from brax.training.agents.ppo import checkpoint as ppo_checkpoint


def load_policy(checkpoint_path: str, deterministic: bool = True):
    """Load a trained policy from a checkpoint."""
    root_dir = Path(__file__).parent.parent.resolve()
    path = root_dir / "models" / checkpoint_path
    return ppo_checkpoint.load_policy(path)


def run_rollout(env, policy, episode_length: int, seed: int = 0):
    """Run a single episode rollout and collect states."""
    jit_step = jax.jit(env.step)
    jit_reset = jax.jit(env.reset)
    jit_policy = jax.jit(policy)

    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)

    # Reset environment
    state = jit_reset(reset_key)

    # Collect trajectory
    states = [state]
    total_reward = 0.0
    total_cost = 0.0

    for step in range(episode_length):
        key, action_key = jax.random.split(key)

        # Get action from policy
        action, _ = jit_policy(state.obs, action_key)

        # Step environment
        state = jit_step(state, action)
        states.append(state)

        # Accumulate metrics
        total_reward += float(state.reward)
        if hasattr(state, 'info') and 'cost' in state.info:
            total_cost += float(state.info['cost'])

        # Check for done
        if state.done:
            break

    return states, total_reward, total_cost


def save_video(
        sys,
        states,
        output_path: str,
        fps: int = 100,
        width: int = 640,
        height: int = 480,
        camera: Optional[str] = None,
):
    """Save a video of the trajectory."""
    try:
        import imageio
    except ImportError:
        print("Warning: imageio not installed. Skipping video save.")
        print("Install with: pip install imageio imageio-ffmpeg")
        return False

    # Get pipeline states for rendering
    pipeline_states = [s.pipeline_state for s in states]

    # Render all frames
    print(f"Rendering {len(pipeline_states)} frames...")
    frames = brax_image.render_array(sys, pipeline_states, height=height, width=width, camera=camera)

    # Save as mp4
    print(f"Saving video to {output_path}...")
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    return True


def save_snapshots(
        sys,
        states,
        output_dir: str,
        interval: int = 20,
        width: int = 640,
        height: int = 480,
        camera: Optional[str] = None,
):
    """Save snapshot images at regular intervals."""
    from PIL import Image

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Select frames at interval
    snapshot_indices = list(range(0, len(states), interval))
    if len(states) - 1 not in snapshot_indices:
        snapshot_indices.append(len(states) - 1)  # Always include last frame

    print(f"Saving {len(snapshot_indices)} snapshots to {output_dir}...")

    for idx in snapshot_indices:
        state = states[idx]

        # Render single frame
        frame = brax_image.render_array(sys, state.pipeline_state, height=height, width=width, camera=camera)

        # Save as PNG
        img = Image.fromarray(frame)
        img.save(output_path / f"frame_{idx:05d}.png")

    print(f"Saved snapshots: {snapshot_indices}")


def main():
    parser = argparse.ArgumentParser(description="Visualize a trained CRAX policy")

    # Required arguments
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--env", required=True, help="Environment name (e.g., safe_point_goal)")

    # Environment options
    parser.add_argument("--difficulty", type=int, default=None, help="Environment difficulty level")
    parser.add_argument("--episode_length", type=int, default=100, help="Episode length")
    parser.add_argument("--env_kwargs", type=str, default=None, help="JSON string of extra env kwargs")

    # Output options
    parser.add_argument("--output_dir", default="visualizations", help="Output directory")
    parser.add_argument("--name", default=None, help="Name for output files (default: env_checkpoint)")

    # Rendering options
    parser.add_argument("--width", type=int, default=640, help="Video/image width")
    parser.add_argument("--height", type=int, default=480, help="Video/image height")
    parser.add_argument("--video_fps", type=int, default=100, help="Video frames per second")
    parser.add_argument("--snapshot_interval", type=int, default=20, help="Steps between snapshots")
    parser.add_argument("--camera", default="fixedfar", help="Camera name for rendering")

    # Policy options
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    # Output control
    parser.add_argument("--no_video", action="store_true", help="Skip video generation")
    parser.add_argument("--no_snapshots", action="store_true", help="Skip snapshot generation")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output name
    if args.name:
        name = args.name
    else:
        checkpoint_name = Path(args.checkpoint).name
        name = f"{args.env}_{checkpoint_name}"

    # Build environment kwargs
    env_kwargs = {}
    if args.difficulty is not None:
        env_kwargs["difficulty"] = args.difficulty
    if args.env_kwargs:
        env_kwargs.update(json.loads(args.env_kwargs))

    print(f"Creating environment: {args.env}")
    if env_kwargs:
        print(f"  kwargs: {env_kwargs}")

    # Create environment
    env = envs.get_environment(args.env, **env_kwargs)

    # Load policy
    print(f"Loading policy from: {args.checkpoint}")
    policy = load_policy(args.checkpoint, deterministic=args.deterministic)

    # Run rollout
    print(f"Running rollout for {args.episode_length} steps...")
    states, total_reward, total_cost = run_rollout(
        env, policy, args.episode_length, seed=args.seed
    )

    print(f"Rollout complete:")
    print(f"  Steps: {len(states)}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Total cost: {total_cost:.2f}")

    # Save video
    if not args.no_video:
        video_path = output_dir / f"{name}.mp4"
        save_video(
            env.sys, states, str(video_path),
            fps=args.video_fps, width=args.width, height=args.height,
            camera=args.camera,
        )

    # Save snapshots
    if not args.no_snapshots:
        snapshots_dir = output_dir / f"{name}_snapshots"
        save_snapshots(
            env.sys, states, str(snapshots_dir),
            interval=args.snapshot_interval, width=args.width, height=args.height,
            camera=args.camera,
        )

    # Save metadata
    metadata = {
        "checkpoint": args.checkpoint,
        "env": args.env,
        "env_kwargs": env_kwargs,
        "episode_length": args.episode_length,
        "actual_steps": len(states),
        "total_reward": total_reward,
        "total_cost": total_cost,
        "deterministic": args.deterministic,
        "seed": args.seed,
    }

    metadata_path = output_dir / f"{name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
