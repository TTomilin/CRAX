# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Curriculum training across multiple environments or difficulty levels.

This module provides utilities for:
1. Sequential training across difficulty levels (e.g., safe_ant_level_1 -> level_2 -> level_3)
2. Continuing training with warm-started parameters

Example usage:
    from brax.training import curriculum
    from brax.training.agents import ppo_lag

    # Define curriculum stages
    stages = [
        curriculum.Stage(env_name='safe_ant_level_1', num_steps=1_000_000),
        curriculum.Stage(env_name='safe_ant_level_2', num_steps=1_000_000),
        curriculum.Stage(env_name='safe_ant_level_3', num_steps=1_000_000),
    ]

    # Train with curriculum
    final_params, all_metrics = curriculum.train_curriculum(
        stages=stages,
        train_fn=ppo_lag.train,
        train_kwargs={'episode_length': 1000, 'num_envs': 2048},
    )
"""

import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from brax import envs
from brax.envs.difficulty import apply_difficulty


class Stage(NamedTuple):
    """A single stage in the curriculum."""
    env_name: str
    level: int
    num_steps: int
    # Optional stage-specific overrides for train_kwargs
    kwargs_override: Optional[Dict[str, Any]] = None


class StageResult(NamedTuple):
    """Results from a single curriculum stage."""
    stage_idx: int
    env_name: str
    num_steps: int
    final_metrics: Dict[str, Any]
    training_time: float
    # All metrics collected during this stage
    metrics_history: List[Dict[str, Any]]


def train_curriculum(
        stages: List[Stage],
        train_fn: Callable,
        train_kwargs: Dict[str, Any],
        progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        seed: int = 0,
) -> Tuple[Any, List[StageResult]]:
    """Train sequentially across multiple curriculum stages.

    Args:
        stages: List of Stage objects defining the curriculum.
        train_fn: Training function (e.g., ppo_lag.train, focops.train).
            Must accept 'environment' or 'env_name' and return (make_policy, params, metrics).
        train_kwargs: Base kwargs passed to train_fn (e.g., episode_length, num_envs).
        progress_fn: Optional callback called with (step, metrics) during training.
            The step is the global step across all stages.
        seed: Random seed for the first stage. Incremented for each stage.

    Returns:
        Tuple of (final_params, list of StageResult for each stage).
    """
    all_results = []
    current_params = None
    global_step = 0

    print(f"Starting curriculum training with {len(stages)} stages")
    print("=" * 60)

    for stage_idx, stage in enumerate(stages):
        print(f"\n[Stage {stage_idx + 1}/{len(stages)}] {stage.env_name}")
        print(f"  Training for {stage.num_steps:,} steps")
        print("-" * 40)

        # Merge base kwargs with stage-specific overrides
        stage_kwargs = dict(train_kwargs)
        if stage.kwargs_override:
            stage_kwargs.update(stage.kwargs_override)

        # Create environment from name, supporting difficulty levels via apply_difficulty
        # Extract env creation kwargs (e.g., backend) that should not be merged into config
        env_creation_kwargs = {k: v for k, v in stage_kwargs.items() if k in ['backend']}

        env_name = stage.env_name
        level = stage.level
        level_kwargs: Dict[str, Any] = {}
        level_kwargs = apply_difficulty(env_name, level_kwargs, level)

        # Create the environment with base name and merged kwargs
        environment = envs.get_environment(env_name, **env_creation_kwargs, **level_kwargs)

        # Set environment and steps for this stage
        stage_kwargs['environment'] = environment
        stage_kwargs['num_timesteps'] = stage.num_steps
        stage_kwargs['seed'] = seed + stage_idx

        # Remove env_name if present (use 'environment' instead)
        stage_kwargs.pop('env_name', None)

        # Warm-start from previous stage's parameters
        if current_params is not None:
            stage_kwargs['restore_checkpoint_path'] = None  # Don't load from file
            stage_kwargs['pretrained_params'] = current_params

        # Collect metrics during training
        stage_metrics = []
        stage_start_step = global_step

        def stage_progress_fn(step: int, metrics: Dict[str, Any]):
            # Add stage info to metrics
            metrics_with_stage = dict(metrics)
            metrics_with_stage['curriculum_stage'] = stage_idx
            metrics_with_stage['curriculum_env'] = stage.env_name
            metrics_with_stage['global_step'] = stage_start_step + step

            stage_metrics.append(metrics_with_stage)

            if progress_fn:
                progress_fn(stage_start_step + step, metrics_with_stage)

        stage_kwargs['progress_fn'] = stage_progress_fn

        # Run training for this stage
        start_time = time.time()
        make_policy, params, final_metrics, eval_env = train_fn(**stage_kwargs)
        training_time = time.time() - start_time

        # Update for next stage
        current_params = params
        global_step += stage.num_steps

        # Store results
        result = StageResult(
            stage_idx=stage_idx,
            env_name=stage.env_name,
            num_steps=stage.num_steps,
            final_metrics=final_metrics,
            training_time=training_time,
            metrics_history=stage_metrics,
        )
        all_results.append(result)

        # Print summary
        print(f"\n  Stage {stage_idx + 1} complete:")
        print(f"    Training time: {training_time:.1f}s")
        if 'eval/episode_reward' in final_metrics:
            print(f"    Final reward: {final_metrics['eval/episode_reward']:.2f}")
        if 'eval/episode_cost' in final_metrics:
            print(f"    Final cost: {final_metrics['eval/episode_cost']:.2f}")

    print("\n" + "=" * 60)
    print("Curriculum training complete!")
    print(f"Total stages: {len(stages)}")
    print(f"Total steps: {global_step:,}")

    return current_params, all_results


def create_difficulty_curriculum(
        env_name: str,
        levels: List[int],
        steps_per_level: int,
) -> List[Stage]:
    """Create a curriculum for environments with difficulty levels.

    Args:
        env_name: Environment name (e.g., 'safe_reacher', 'safe_point_goal').
        levels: List of difficulty levels to train on.
        steps_per_level: Number of training steps per level.

    Returns:
        List of Stage objects.
    """
    stages = []
    for level in levels:
        stages.append(Stage(env_name, level, steps_per_level))
    return stages


def split_budget_curriculum(
        env_names: List[str],
        total_steps: int,
        weights: Optional[List[float]] = None,
) -> List[Stage]:
    """Split a total training budget across multiple environments.

    Args:
        env_names: List of environment names.
        total_steps: Total training budget to split.
        weights: Optional weights for each environment. If None, split equally.

    Returns:
        List of Stage objects.
    """
    if weights is None:
        weights = [1.0] * len(env_names)

    total_weight = sum(weights)
    stages = []
    for env_name, weight in zip(env_names, weights):
        steps = int(total_steps * weight / total_weight)
        stages.append(Stage(env_name=env_name, num_steps=steps))
    return stages
