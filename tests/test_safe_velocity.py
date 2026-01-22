"""Tests for the unified safe_velocity environment."""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from brax import envs
from brax.envs.safe_velocity import SafeVelocity, DEFAULT_THRESHOLDS, create_safe_velocity_env
from run_utils import record_episode_video_simple


AGENTS = ["ant", "halfcheetah", "hopper", "humanoid", "swimmer", "walker2d"]


def _run_video_for_agent(agent: str, steps: int = 300, num_episodes: int = 1):
    """Helper to record video for a specific agent."""
    env = SafeVelocity(agent=agent)
    return record_episode_video_simple(
        env,
        steps=steps,
        action_mode="periodic",
        out_name=f"safe_velocity_{agent}",
        show_metrics=True,
        extra_metrics=["velocity_value", "velocity_violation"],
        num_episodes=num_episodes,
        fps=50,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test safe_velocity environments")
    parser.add_argument("--agent", type=str, default="ant",
                        choices=AGENTS + ["all"],
                        help="Agent to test (or 'all' for all agents)")
    parser.add_argument("--steps", type=int, default=300, help="Steps per episode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    args = parser.parse_args()

    agents_to_test = AGENTS if args.agent == "all" else [args.agent]

    for agent in agents_to_test:
        print(f"\n{'='*50}")
        print(f"Testing safe_velocity with agent: {agent}")
        print(f"{'='*50}")

        env = SafeVelocity(agent=agent)
        print(f"Action size: {env.action_size}")
        print(f"Observation size: {env.observation_size}")
        print(f"Default threshold: {DEFAULT_THRESHOLDS[agent]}")

        key = jax.random.PRNGKey(42)
        state = env.reset(key)
        print(f"Initial state obs shape: {state.obs.shape}")

        print(f"\nRecording video for {agent}...")
        video_path = _run_video_for_agent(agent, steps=args.steps, num_episodes=args.episodes)
        print(f"Video saved to: {video_path}")
