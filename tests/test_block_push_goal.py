#!/usr/bin/env python3
"""Test script for BlockPushGoal environment.

Renders a circular motion rollout and saves as a video with accumulated
reward and cost overlays.

Usage:
    python tests/test_block_push_goal.py
    python tests/test_block_push_goal.py --episode_length 500 --num_episodes 2
    python tests/test_block_push_goal.py --thrust 0.6 --yaw_rate 0.4
"""

import argparse

from brax import envs
from run_utils import record_episode_video
from utils import make_circular_policy


def main():
    parser = argparse.ArgumentParser(
        description="Test BlockPushGoal environment with circular motion actions"
    )
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
    parser.add_argument("--thrust", type=float, default=1.0,
                        help="Forward velocity for circular motion (constant)")
    parser.add_argument("--yaw_rate", type=float, default=0.3,
                        help="Turning rate for circular motion (constant, positive = left)")

    # Environment configuration
    parser.add_argument("--num_hazards", type=int, default=8,
                        help="Number of hazards in the environment")
    parser.add_argument("--goal_size", type=float, default=0.2,
                        help="Size of the goal")
    args = parser.parse_args()

    print("Creating BlockPushGoal environment")
    print(f"  num_hazards: {args.num_hazards}, goal_size: {args.goal_size}")

    env = envs.get_environment(
        'block_push_goal',
        hazard_specs=[
            dict(type='cube', count=args.num_hazards, size=0.2, height=0.2,
                 collidable=True, movable=False, density=1.0),
        ],
        goal_size=args.goal_size,
    )

    # Create circular policy using the helper
    print(f"Creating circular policy: thrust={args.thrust}, yaw_rate={args.yaw_rate}")
    make_policy = make_circular_policy(
        action_dim=env.action_size,
        thrust=args.thrust,
        yaw_rate=args.yaw_rate,
        thrust_idx=0,
        yaw_idx=1,
    )

    # Record video using existing utility
    print(f"Recording {args.num_episodes} episode(s), {args.episode_length} steps each")
    record_episode_video(
        env=env,
        make_inference_fn=make_policy,
        params=None,  # circular policy doesn't use params
        steps=args.episode_length,
        cameras=[args.camera],
        width=args.width,
        height=args.height,
        fps=args.fps,
        out_name="block_push_goal_circular",
        log_to_wandb=False,
        seed=args.seed,
        show_metrics=not args.no_metrics,
        num_episodes=args.num_episodes,
    )

    print(f"\nDone! Video saved to: {args.output_dir}/block_push_goal_circular_{args.camera}.mp4")


if __name__ == '__main__':
    main()
