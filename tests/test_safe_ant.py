"""Tests for the SafeAnt environment."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from brax.envs.safe_ant import SafeAnt
from run_utils import record_episode_video_simple


def _init_safe_ant(difficulty: int = 1):
    env = SafeAnt(difficulty=difficulty)
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test SafeAnt environment")
    parser.add_argument("--difficulty", type=int, default=1, help="Difficulty level")
    parser.add_argument("--steps", type=int, default=300, help="Steps per episode")
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes")
    args = parser.parse_args()

    env = _init_safe_ant(difficulty=args.difficulty)
    record_episode_video_simple(
        env,
        steps=args.steps,
        action_mode="periodic",
        out_name="safe_ant",
        show_metrics=True,
        extra_metrics=["feet_on_ground", "x_velocity"],
        num_episodes=args.episodes,
        fps=50,
    )
