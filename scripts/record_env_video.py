#!/usr/bin/env python3
"""
Record video of environment rollouts with optional trained policy.

This script unifies the video recording functionality from various test files.
Each environment can have specific default configurations.

Usage examples:
  python scripts/record_env_video.py --env safe_ant
  python scripts/record_env_video.py --env safe_point_goal --level 2 --steps 500
  python scripts/record_env_video.py --env safe_walker --model my_checkpoint
  python scripts/record_env_video.py --env safe_velocity --agent ant --level 1
"""
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Environment-specific default configurations
ENV_DEFAULTS = {
    "safe_ant": {
        "steps": 300,
        "episodes": 2,
        "fps": 50,
        "action_mode": "periodic",
        "extra_metrics": ["feet_on_ground", "x_velocity"],
        "cameras": [0],
        "env_class": "SafeAnt",
        "env_module": "brax.envs.safe_ant",
    },
    "safe_reacher": {
        "steps": 100,
        "episodes": 5,
        "fps": 25,
        "action_mode": "periodic",
        "extra_metrics": ["dist"],
        "cameras": [0],
        "env_class": "SafeReacher",
        "env_module": "brax.envs.safe_reacher",
    },
    "safe_walker": {
        "steps": 1000,
        "episodes": 1,
        "fps": 100,
        "action_mode": "random",
        "extra_metrics": [],
        "cameras": ["fixedfar"],
        "env_kwargs": {"cost": {"scaler": 0.1}, "reward": {"scaler": 0.01}},
    },
    "safe_height": {
        "steps": 300,
        "episodes": 1,
        "fps": 50,
        "action_mode": "random",
        "extra_metrics": [],
        "cameras": ["track"],
    },
    "safe_point_goal": {
        "steps": 300,
        "episodes": 1,
        "fps": 100,
        "action_mode": "random",
        "extra_metrics": [],
        "cameras": [0],
    },
    "safe_point_circle": {
        "steps": 300,
        "episodes": 1,
        "fps": 100,
        "action_mode": "circular",  # Special case for circular policy
        "extra_metrics": [],
        "cameras": ["fixedfar"],
    },
    "safe_circle": {
        "steps": 300,
        "episodes": 1,
        "fps": 100,
        "action_mode": "circular",
        "extra_metrics": [],
        "cameras": ["fixedfar"],
    },
    "safe_spider": {
        "steps": 300,
        "episodes": 1,
        "fps": 50,
        "action_mode": "periodic",
        "extra_metrics": ["feet_on_ground"],
        "cameras": ["track"],
        "width": 640,
        "height": 480,
    },
    "safe_velocity": {
        "steps": 300,
        "episodes": 1,
        "fps": 50,
        "action_mode": "periodic",
        "extra_metrics": ["velocity_value", "velocity_violation"],
        "cameras": [0],
    },
}


def load_policy_from_checkpoint(model_path: str):
    """Load a policy from a checkpoint path."""
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    candidates = [model_path]
    if not os.path.isabs(model_path):
        root_dir = Path(__file__).parent.parent.resolve()
        candidates.append(str(root_dir / "models" / model_path))

    for c in candidates:
        if os.path.exists(c):
            try:
                return ppo_checkpoint.load_policy(c, deterministic=True)
            except Exception as e:
                print(f"Failed to load policy from '{c}': {e}")
                return None

    print(f"Model path not found. Tried: {candidates}.")
    return None


def create_environment(env_name: str, level: int = None, agent: str = None, seed: int = None, **extra_kwargs):
    """Create an environment with optional level and agent parameters."""
    from brax import envs

    defaults = ENV_DEFAULTS.get(env_name, {})

    # Check if we should use direct class instantiation
    if "env_class" in defaults:
        import importlib
        module = importlib.import_module(defaults["env_module"])
        env_cls = getattr(module, defaults["env_class"])

        kwargs = {}
        if level is not None:
            kwargs["difficulty"] = level
        kwargs.update(defaults.get("env_kwargs", {}))
        kwargs.update(extra_kwargs)
        return env_cls(**kwargs)

    # Use registry-based creation
    kwargs = {}
    if level is not None:
        kwargs["level"] = level
    if agent is not None:
        kwargs["agent"] = agent
    if seed is not None:
        kwargs["seed"] = seed
    kwargs.update(defaults.get("env_kwargs", {}))
    kwargs.update(extra_kwargs)

    print(f"Creating {env_name} environment (level={level}, agent={agent})")
    return envs.get_environment(env_name, **kwargs)


def make_circular_policy(action_dim: int, thrust: float = 1.0, yaw_rate: float = 0.0,
                         thrust_idx: int = 0, yaw_idx: int = 1):
    """Create a circular policy for point-mass circle environments."""
    import jax.numpy as jp

    base = jp.zeros((action_dim,), dtype=jp.float32)
    base = base.at[thrust_idx].set(thrust)
    base = base.at[yaw_idx].set(yaw_rate)

    def policy(obs, rng):
        return base, {}

    return policy


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Record video of environment rollouts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment-specific defaults are applied automatically. Override with flags.

Examples:
  %(prog)s --env safe_ant
  %(prog)s --env safe_point_goal --level 2 --steps 500
  %(prog)s --env safe_walker --model checkpoint_name
  %(prog)s --env safe_velocity --agent halfcheetah --level 3
""")
    p.add_argument("--env", type=str, required=True, help="Environment name")
    p.add_argument("--level", type=int, default=None, help="Difficulty level")
    p.add_argument("--agent", type=str, default=None,
                   help="Agent type (for multi-agent envs like safe_velocity)")
    p.add_argument("--steps", type=int, default=None, help="Steps per episode")
    p.add_argument("--episodes", type=int, default=None, help="Number of episodes")
    p.add_argument("--fps", type=int, default=None, help="Video frames per second")
    p.add_argument("--camera", type=str, default=None, help="Camera name")
    p.add_argument("--action-mode", type=str, default=None,
                   choices=["random", "zero", "periodic", "circular"],
                   help="Action generation mode (ignored if --model is provided)")
    p.add_argument("--model", type=str, default=None,
                   help="Path to checkpoint for trained policy")
    p.add_argument("--seed", type=int, default=1, help="Random seed")
    p.add_argument("--out", type=str, default=None,
                   help="Output video name (default: env name)")
    p.add_argument("--width", type=int, default=None, help="Video width")
    p.add_argument("--height", type=int, default=None, help="Video height")
    p.add_argument("--no-metrics", action="store_true",
                   help="Disable metric overlay on video")
    p.add_argument("--cpu", action="store_true", help="Force CPU execution")
    return p


def main(args: argparse.Namespace) -> None:
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"

    os.environ.setdefault("MUJOCO_GL", "egl")

    from run_utils import record_episode_video_simple, record_episode_video

    # Get environment defaults
    defaults = ENV_DEFAULTS.get(args.env, {})

    # Apply defaults with CLI overrides
    steps = args.steps if args.steps is not None else defaults.get("steps", 300)
    episodes = args.episodes if args.episodes is not None else defaults.get("episodes", 1)
    fps = args.fps if args.fps is not None else defaults.get("fps", 50)
    action_mode = args.action_mode if args.action_mode is not None else defaults.get("action_mode", "random")
    extra_metrics = defaults.get("extra_metrics", [])
    width = args.width if args.width is not None else defaults.get("width", 320)
    height = args.height if args.height is not None else defaults.get("height", 240)

    if args.camera is not None:
        cameras = [args.camera]
    else:
        cameras = defaults.get("cameras", [0])

    out_name = args.out if args.out is not None else args.env
    if args.level is not None:
        out_name = f"{out_name}_level{args.level}"
    if args.agent is not None:
        out_name = f"{out_name}_{args.agent}"

    # Create environment
    env = create_environment(
        args.env,
        level=args.level,
        agent=args.agent,
        seed=args.seed,
    )

    # Handle policy loading or action mode
    policy = None
    if args.model:
        policy = load_policy_from_checkpoint(args.model)
        if policy is None:
            print("Warning: Could not load policy, falling back to action mode")

    # Handle circular policy special case
    if action_mode == "circular" and policy is None:
        print("Using circular policy for orbit behavior")
        policy = make_circular_policy(
            action_dim=env.action_size,
            thrust=1.0,
            yaw_rate=0.0,
            thrust_idx=0,
            yaw_idx=1,
        )
        # Use record_episode_video with make_inference_fn for circular policy
        def make_infer(_params):
            return policy
        record_episode_video(
            env,
            make_inference_fn=make_infer,
            params=None,
            steps=steps,
            out_name=out_name,
            cameras=cameras,
            width=width,
            height=height,
            show_metrics=not args.no_metrics,
            num_episodes=episodes,
            fps=fps,
            seed=args.seed,
            log_to_wandb=False,
        )
        return

    # Use simple recording
    record_episode_video_simple(
        env,
        steps=steps,
        policy=policy,
        action_mode=action_mode if policy is None else None,
        cameras=cameras,
        width=width,
        height=height,
        out_name=out_name,
        seed=args.seed,
        show_metrics=not args.no_metrics,
        num_episodes=episodes,
        fps=fps,
        extra_metrics=extra_metrics if extra_metrics else None,
    )


if __name__ == "__main__":
    main(build_args().parse_args())
