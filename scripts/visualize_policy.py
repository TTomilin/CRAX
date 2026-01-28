#!/usr/bin/env python3
"""Visualize a trained policy by running rollouts and saving video + snapshots.

Usage:
    python scripts/visualize_policy.py --checkpoint path/to/checkpoint --env safe_point_goal

    # With custom options
    python scripts/visualize_policy.py \
        --checkpoint results/ppo_lag/checkpoint_10000000 \
        --env safe_point_goal \
        --level 2 \
        --episode_length 1000 \
        --num_episodes 3 \
        --output_dir visualizations \
        --snapshot_interval 20 \
        --video_fps 30 \
        --deterministic
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List, Tuple

import imageio.v3 as iio
import jax
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def make_random_policy(action_size: int):
    """Creates a policy that returns random actions."""
    def random_policy(obs, key):
        del obs
        action = jax.random.uniform(key, shape=(action_size,), minval=-1.0, maxval=1.0)
        return action, {}
    return random_policy


def run_rollout(
        env,
        policy,
        num_episodes: int,
        episode_length: int,
        seed: int = 0
) -> Tuple[List, List[dict], List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Run multiple episode rollouts and collect states.

    Args:
        env: The environment
        policy: The policy function
        num_episodes: Number of episodes to run
        episode_length: Maximum steps per episode
        seed: Random seed

    Returns:
        Tuple of (all_states, episode_metrics, all_rewards, all_costs) where episode_metrics is a list
        of dicts with per-episode reward/cost/length, and all_rewards/all_costs are lists of (step_val, cumulative_val)
        for every step across all episodes.
    """
    jit_step = jax.jit(env.step)
    jit_reset = jax.jit(env.reset)
    jit_policy = jax.jit(policy)

    key = jax.random.PRNGKey(seed)

    all_states = []
    episode_metrics = []
    all_rewards_per_step: List[Tuple[float, float]] = []
    all_costs_per_step: List[Tuple[float, float]] = []

    for ep in range(num_episodes):
        key, reset_key = jax.random.split(key)

        # Reset environment for each episode
        state = jit_reset(reset_key)
        episode_states = [state]
        episode_reward = 0.0
        episode_cost = 0.0
        episode_steps = 0

        # Store the initial state's reward/cost (always 0 at reset)
        all_rewards_per_step.append((0.0, 0.0))
        all_costs_per_step.append((0.0, 0.0))

        for step in range(episode_length):
            key, action_key = jax.random.split(key)

            # Get action from policy
            action, _ = jit_policy(state.obs, action_key)

            # Step environment
            state = jit_step(state, action)
            episode_states.append(state)
            episode_steps += 1

            # Accumulate metrics
            current_reward = float(state.reward)
            episode_reward += current_reward

            current_cost = 0.0
            if hasattr(state, 'info') and 'cost' in state.info:
                current_cost = float(state.info['cost'])
            # Brax uses metrics for costs (e.g. episodic/cost)
            elif hasattr(state, 'metrics') and 'cost' in state.metrics:
                current_cost = float(state.metrics['cost'])

            episode_cost += current_cost

            all_rewards_per_step.append((current_reward, episode_reward))
            all_costs_per_step.append((current_cost, episode_cost))

            # Check for done
            if state.done:
                break

        # Store episode data
        all_states.extend(episode_states)
        episode_metrics.append({
            'episode': ep + 1,
            'reward': episode_reward,
            'cost': episode_cost,
            'length': episode_steps,
        })

        print(f"  Episode {ep + 1}/{num_episodes}: "
              f"reward={episode_reward:.2f}, cost={episode_cost:.2f}, steps={episode_steps}")

    return all_states, episode_metrics, all_rewards_per_step, all_costs_per_step


def save_video(
        env,
        states,
        rewards_per_step: List[Tuple[float, float]],
        costs_per_step: List[Tuple[float, float]],
        output_path: str,
        fps: int = 100,
        width: int = 640,
        height: int = 480,
        camera: Optional[str] = None,
        show_metrics: bool = True,
        font: str = "DejaVuSans-Bold",
        font_size: int = 20,
):
    """Save a video of the trajectory with optional metric overlay."""
    # Get pipeline states for rendering
    pipeline_states = [s.pipeline_state for s in states]

    # Render all frames
    print(f"Rendering {len(pipeline_states)} frames...")
    if hasattr(env, 'render') and callable(getattr(env, 'render')):
        frames_np = env.render(pipeline_states, height=height, width=width, camera=camera)
    else:
        frames_np = brax_image.render_array(env.sys, pipeline_states, height=height, width=width, camera=camera)

    frames_to_write = []

    if show_metrics:
        try:
            # Try to load specified font
            font_obj = ImageFont.truetype(f"{font}.ttf", font_size)
        except (OSError, IOError):
            # Fallback to default font if not found
            font_obj = ImageFont.load_default()
            print(f"Warning: Could not load font '{font}'. Using default font.")

        for i, frame_array in enumerate(frames_np):
            img = Image.fromarray(frame_array.astype(np.uint8))
            draw = ImageDraw.Draw(img)

            # Metrics for current frame (skip first state which is reset state)
            if i > 0 and i - 1 < len(rewards_per_step):
                # rewards_per_step and costs_per_step are 0-indexed for steps,
                # but states is 0-indexed for states (includes initial state), so offset
                _, r_cum = rewards_per_step[i - 1]
                _, c_cum = costs_per_step[i - 1]
            else:
                r_cum = 0.0
                c_cum = 0.0

            reward_text = f"Reward: {r_cum:.2f}"
            cost_text = f"Cost: {c_cum:.2f}"

            draw.text((10, 10), reward_text, font=font_obj, fill=(50, 220, 50),
                      stroke_width=2, stroke_fill=(0, 0, 0))
            draw.text((10, 40), cost_text, font=font_obj, fill=(230, 60, 60),
                      stroke_width=2, stroke_fill=(0, 0, 0))
            frames_to_write.append(np.array(img))
    else:
        frames_to_write = frames_np

    # Save as mp4
    print(f"Saving video to {output_path}...")
    iio.imwrite(output_path, np.stack(frames_to_write), fps=fps)

    return True


def save_snapshots(
        env,
        states,
        output_dir: str,
        rewards_per_step: Optional[List[Tuple[float, float]]] = None,
        costs_per_step: Optional[List[Tuple[float, float]]] = None,
        interval: int = 20,
        width: int = 640,
        height: int = 480,
        camera: Optional[str] = None,
        show_metrics: bool = False,
        font: str = "DejaVuSans-Bold",
        font_size: int = 20,
):
    """Save snapshot images at regular intervals with optional metric overlay."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Select frames at interval
    snapshot_indices = list(range(0, len(states), interval))
    if len(states) - 1 not in snapshot_indices:
        snapshot_indices.append(len(states) - 1)  # Always include last frame

    print(f"Saving {len(snapshot_indices)} snapshots to {output_dir}...")

    # Load font if showing metrics
    font_obj = None
    if show_metrics:
        try:
            font_obj = ImageFont.truetype(f"{font}.ttf", font_size)
        except (OSError, IOError):
            font_obj = ImageFont.load_default()
            print(f"Warning: Could not load font '{font}'. Using default font.")

    for idx in snapshot_indices:
        state = states[idx]

        # Render single frame
        if hasattr(env, 'render') and callable(getattr(env, 'render')):
            frame = env.render(state.pipeline_state, height=height, width=width, camera=camera)
        else:
            frame = brax_image.render_array(env.sys, state.pipeline_state, height=height, width=width, camera=camera)

        # Create image from frame
        img = Image.fromarray(frame)

        # Add metric overlay if requested
        if show_metrics and font_obj is not None and rewards_per_step is not None and costs_per_step is not None:
            draw = ImageDraw.Draw(img)

            # Get cumulative metrics for this frame
            if idx > 0 and idx - 1 < len(rewards_per_step):
                _, r_cum = rewards_per_step[idx - 1]
                _, c_cum = costs_per_step[idx - 1]
            else:
                r_cum = 0.0
                c_cum = 0.0

            reward_text = f"Reward: {r_cum:.2f}"
            cost_text = f"Cost: {c_cum:.2f}"
            step_text = f"Step: {idx}"

            draw.text((10, 10), reward_text, font=font_obj, fill=(50, 220, 50),
                      stroke_width=2, stroke_fill=(0, 0, 0))
            draw.text((10, 40), cost_text, font=font_obj, fill=(230, 60, 60),
                      stroke_width=2, stroke_fill=(0, 0, 0))
            draw.text((10, 70), step_text, font=font_obj, fill=(200, 200, 200),
                      stroke_width=2, stroke_fill=(0, 0, 0))

        # Save as PNG
        img.save(output_path / f"frame_{idx:05d}.png")

    print(f"Saved snapshots: {snapshot_indices}")


def main():
    import time
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Visualize a trained CRAX policy")

    # Required arguments
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint directory")
    parser.add_argument("--env", required=True, help="Environment name (e.g., safe_point_goal)")

    # Environment options
    parser.add_argument("--level", type=int, default=None, help="Environment difficulty level (1, 2, or 3)")
    parser.add_argument("--episode_length", type=int, default=None, help="Episode length (default: use env default)")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run")
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
    parser.add_argument("--show_metrics", action="store_true", help="Overlay reward and cost on video frames")
    parser.add_argument("--font", type=str, default="DejaVuSans-Bold", help="Font for metric overlay")
    parser.add_argument("--font_size", type=int, default=20, help="Font size for metric overlay")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir) / args.env
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output name
    if args.name:
        name = args.name
    else:
        if args.checkpoint:
            checkpoint_name = Path(args.checkpoint).name
        else:
            checkpoint_name = 'random_policy'
        level_str = f"_level_{args.level}" if args.level else ""
        name = f"{args.env}{level_str}_{checkpoint_name}"

    # Build environment kwargs
    env_kwargs = {}
    if args.env_kwargs:
        env_kwargs.update(json.loads(args.env_kwargs))

    print(f"Creating environment: {args.env}")
    if args.level:
        print(f"  Difficulty level: {args.level}")
    if env_kwargs:
        print(f"  Extra kwargs: {env_kwargs}")

    # Create environment with difficulty level
    env = envs.get_environment(args.env, level=args.level, **env_kwargs)

    # Determine episode length
    episode_length = args.episode_length
    if episode_length is None:
        episode_length = getattr(env, 'episode_length', 1000)
    print(f"  Episode length: {episode_length}")

    # Load policy or create a random one
    if args.checkpoint:
        print(f"Loading policy from: {args.checkpoint}")
        policy = load_policy(args.checkpoint, deterministic=args.deterministic)
    else:
        print("No checkpoint provided, using random policy.")
        policy = make_random_policy(env.action_size)

    # Run rollout
    print(f"Running {args.num_episodes} episode(s) for up to {episode_length} steps each...")
    states, episode_metrics, all_rewards_per_step, all_costs_per_step = run_rollout(
        env, policy,
        num_episodes=args.num_episodes,
        episode_length=episode_length,
        seed=args.seed
    )

    # Compute aggregate metrics
    total_reward = sum(ep['reward'] for ep in episode_metrics)
    total_cost = sum(ep['cost'] for ep in episode_metrics)
    avg_reward = total_reward / len(episode_metrics)
    avg_cost = total_cost / len(episode_metrics)
    avg_length = sum(ep['length'] for ep in episode_metrics) / len(episode_metrics)

    print(f"\nRollout complete:")
    print(f"  Total frames: {len(states)}")
    print(f"  Episodes: {len(episode_metrics)}")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average cost: {avg_cost:.2f}")
    print(f"  Average length: {avg_length:.1f}")

    # Save video
    if not args.no_video:
        video_path = output_dir / f"{name}.mp4"
        save_video(
            env, states, all_rewards_per_step, all_costs_per_step, str(video_path), fps=args.video_fps,
            width=args.width, height=args.height, camera=args.camera, show_metrics=args.show_metrics, font=args.font,
            font_size=args.font_size
        )

    # Save snapshots
    if not args.no_snapshots:
        snapshots_dir = output_dir / f"{name}_snapshots"
        save_snapshots(
            env, states, str(snapshots_dir), rewards_per_step=all_rewards_per_step, costs_per_step=all_costs_per_step,
            interval=args.snapshot_interval, width=args.width, height=args.height, camera=args.camera,
            show_metrics=args.show_metrics, font=args.font, font_size=args.font_size
        )

    # Save metadata
    metadata = {
        "checkpoint": args.checkpoint,
        "env": args.env,
        "level": args.level,
        "env_kwargs": env_kwargs,
        "episode_length": episode_length,
        "num_episodes": args.num_episodes,
        "total_frames": len(states),
        "total_reward": total_reward,
        "total_cost": total_cost,
        "avg_reward": avg_reward,
        "avg_cost": avg_cost,
        "avg_length": avg_length,
        "episode_metrics": episode_metrics,
        "deterministic": args.deterministic,
        "seed": args.seed,
    }

    metadata_path = output_dir / f"{name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")

    print("\nDone!")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Script finished in {duration:.2f} seconds.")


if __name__ == "__main__":
    main()
