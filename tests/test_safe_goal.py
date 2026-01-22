"""Tests for the SafePointGoal environment."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from brax import envs
from run_utils import record_episode_video_simple


def _init_safe_goal(level):
    print(f"Creating SafePointGoal environment (level={level})")
    env = envs.get_environment('safe_point_goal', level=level)
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test SafePointGoal environment")
    parser.add_argument("--steps", type=int, default=300, help="Steps per episode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--level", type=int, default=1, help="Difficulty level")
    args = parser.parse_args()

    env = _init_safe_goal(args.level)
    record_episode_video_simple(
        env,
        steps=args.steps,
        action_mode="random",
        out_name="safe_point_goal",
        show_metrics=True,
        num_episodes=args.episodes,
        fps=100,
    )
