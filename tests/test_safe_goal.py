#!/usr/bin/env python3
"""Test script for SafePointGoal environment.

Renders a random policy rollout and saves as a video with accumulated
reward and cost overlays.

Usage:
    python tests/test_safe_goal.py
    python tests/test_safe_goal.py --level 2 --episode_length 500 --num_episodes 2
"""

import argparse

from brax import envs
from run_utils import record_episode_video
from utils import make_random_policy


def main():
    parser = argparse.ArgumentParser(description="Test SafePointGoal environment with random actions")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3],
                        help="Difficulty level (1=easy, 2=medium, 3=hard)")
    parser.add_argument("--episode_length", type=int, default=300,
                        help="Maximum steps per episode")
    parser.add_argument("--num_episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--output_dir", type=str, default="videos",
                        help="Output directory for videos")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--width", type=int, default=640,
                        help="Video width")
    parser.add_argument("--height", type=int, default=480,
                        help="Video height")
    parser.add_argument("--fps", type=int, default=50,
                        help="Video frames per second")
    parser.add_argument("--camera", type=str, default="fixedfar",
                        help="Camera name for rendering")
    parser.add_argument("--no_metrics", action="store_true",
                        help="Disable reward/cost overlay on video")
    args = parser.parse_args()

    print(f"Creating SafePointGoal environment (level={args.level})")
    env = envs.get_environment('safe_point_goal', level=args.level)

    # Create random policy
    make_policy = make_random_policy(action_dim=env.action_size)

    # Record video using existing utility
    print(f"Recording {args.num_episodes} episode(s), {args.episode_length} steps each")
    record_episode_video(
        env=env,
        make_inference_fn=make_policy,
        params=None,  # random policy doesn't use params
        steps=args.episode_length,
        cameras=[args.camera],
        width=args.width,
        height=args.height,
        fps=args.fps,
        out_name=f"{args.output_dir}/safe_point_goal_level{args.level}",
        log_to_wandb=False,
        seed=args.seed,
        show_metrics=not args.no_metrics,
        num_episodes=args.num_episodes,
    )

    print(f"\nDone! Video saved to: {args.output_dir}/safe_point_goal_level{args.level}_{args.camera}.mp4")


if __name__ == '__main__':
    main()
