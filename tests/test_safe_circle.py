"""Tests for the SafeHeightHumanoid environment."""

import argparse
import sys
from pathlib import Path

from utils import make_circular_policy

sys.path.insert(0, str(Path(__file__).parent.parent))

from brax import envs
from run_utils import record_episode_video_simple, record_episode_video


def _init_safe_circle(level):
    print(f"Creating SafeHeightHumanoid environment (level={level})")
    env = envs.get_environment('safe_point_circle', level=level)
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test SafeHeightHumanoid environment")
    parser.add_argument("--steps", type=int, default=300, help="Steps per episode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--level", type=int, default=1, help="Difficulty level")
    args = parser.parse_args()

    env = _init_safe_circle(args.level)

    # Make the constant-action circular policy
    make_infer = make_circular_policy(
        action_dim=env.action_size,
        thrust=1.0,  # forward push along agent x
        yaw_rate=0.0,  # steady left turn -> circle
        thrust_idx=0,  # actuator 0 = site motor along x
        yaw_idx=1,  # actuator 1 = velocity actuator on hinge 'z'
    )

    record_episode_video(
        env,
        make_inference_fn=make_infer,
        params=None,  # policy ignores params
        steps=args.steps,
        out_name="safe_circle",
        cameras=["fixedfar"],
        show_metrics=True,
        num_episodes=args.episodes,
        fps=100,
    )
