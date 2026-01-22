"""Tests for the SafeReacher environment."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from brax.envs.safe_reacher import SafeReacher
from run_utils import record_episode_video_simple


def _init_safe_reacher():
    env = SafeReacher()
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test SafeReacher environment")
    parser.add_argument("--steps", type=int, default=100, help="Steps per episode")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    args = parser.parse_args()

    env = _init_safe_reacher()
    record_episode_video_simple(
        env,
        steps=args.steps,
        action_mode="periodic",
        out_name="safe_reacher",
        show_metrics=True,
        extra_metrics=["dist"],
        num_episodes=args.episodes,
        fps=25,
    )
