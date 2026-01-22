"""Tests for the SafeWalker environment."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from brax.envs.safe_walker import SafeWalker
from run_utils import record_episode_video_simple


def _load_policy_from_model_arg(model_arg: str):
    """Load a policy from a checkpoint path."""
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    candidates = []
    if model_arg:
        candidates.append(model_arg)
    candidates.append(os.path.join("models", model_arg))

    ckpt = None
    for c in candidates:
        if os.path.exists(c):
            ckpt = c
            break

    if ckpt is None:
        print(f"Model path not found. Tried: {candidates}.")
        return None

    try:
        return ppo_checkpoint.load_policy(ckpt, deterministic=True)
    except Exception as e:
        print(f"Failed to load policy from '{ckpt}': {e}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test SafeWalker environment")
    parser.add_argument("--model", type=str, default=None, help="Checkpoint path or name under models/")
    parser.add_argument("--steps", type=int, default=1000, help="Steps per episode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--camera", type=str, default="fixedfar", help="Camera name")
    parser.add_argument("--out", type=str, default="safe_walker", help="Output video name")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"

    import jax

    kwargs = {
        "cost": {"scaler": 0.1},
        "reward": {"scaler": 0.01}
    }
    env = SafeWalker(**kwargs)

    policy = None
    action_mode = "random"
    if args.model:
        root_dir = Path(__file__).parent.parent.resolve()
        model_path = root_dir / "models" / args.model
        policy = _load_policy_from_model_arg(str(model_path))
        if policy is not None:
            action_mode = None  # Use provided policy

    record_episode_video_simple(
        env,
        steps=args.steps,
        policy=policy,
        action_mode=action_mode if policy is None else "random",
        cameras=[args.camera],
        out_name=args.out,
        seed=args.seed,
        show_metrics=True,
        num_episodes=args.episodes,
        fps=100,
    )
