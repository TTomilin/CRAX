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

"""Safety transfer utilities for transferring unsafe policies to safe RL algorithms.

This module provides utilities for:
1. Transferring PPO-trained policies to safe RL algorithms (PPO-Lag, PPO-PID, P3O, FOCOPS)
2. Initializing missing networks (cost_value) when transferring

The key insight is that safe RL algorithms have the same policy and value networks
as PPO, plus an additional cost_value network. We can:
- Copy policy and value parameters directly
- Initialize cost_value from scratch (random) or by copying value network params

Example usage:
    from brax.training import transfer
    from brax.training.agents import ppo, ppo_lag

    # Train unsafe PPO first
    make_policy, ppo_params, _ = ppo.train(
        env_name='safe_ant',
        num_timesteps=5_000_000,
    )

    # Transfer to PPO-Lag and continue training for safety
    make_safe_policy, safe_params, _ = ppo_lag.train(
        env_name='safe_ant',
        num_timesteps=5_000_000,
        pretrained_params=ppo_params,
        init_cost_value_from='value',  # or 'random'
    )

For a complete safety transfer benchmark:
    results = transfer.benchmark_safety_transfer(
        env_name='safe_ant',
        unsafe_train_fn=ppo.train,
        safe_train_fns={
            'ppo_lag': ppo_lag.train,
            'ppo_pid': ppo_pid.train,
            'focops': focops.train,
            'p3o': p3o.train,
        },
        unsafe_steps=5_000_000,
        safe_steps=5_000_000,
    )
"""

import copy
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from brax import envs
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from run_utils import filter_kwargs_for_fn, record_episode_video

# Optional wandb import - only used if wandb_config is provided
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class TransferResult(NamedTuple):
    """Results from a safety transfer experiment."""
    algorithm: str
    unsafe_final_metrics: Dict[str, Any]
    safe_final_metrics: Dict[str, Any]
    unsafe_training_time: float
    safe_training_time: float
    # Metrics history during safe training (to see adaptation curve)
    safe_metrics_history: List[Dict[str, Any]]


def convert_ppo_to_safe_params(
        ppo_params: Any,
        cost_value_init_fn: Callable,
        init_cost_value_from: str = 'value',
) -> Any:
    """Convert PPO parameters to safe RL parameters.

    Safe RL algorithms (PPO-Lag, PPO-PID, P3O, FOCOPS) have the structure:
        PPONetworkParams(policy=..., value=..., cost_value=...)

    While PPO has:
        PPONetworkParams(policy=..., value=...)

    This function adds the cost_value network.

    Args:
        ppo_params: Parameters from PPO training.
        cost_value_init_fn: Function to initialize cost_value network.
        init_cost_value_from: How to initialize cost_value:
            - 'value': Copy from value network (often works well)
            - 'random': Random initialization

    Returns:
        Parameters with cost_value network added.
    """
    # Check if params already have cost_value
    if hasattr(ppo_params, 'cost_value') and ppo_params.cost_value is not None:
        return ppo_params

    # Initialize cost_value network
    if init_cost_value_from == 'value':
        # Copy value network parameters to cost_value
        cost_value_params = copy.deepcopy(ppo_params.value)
    elif init_cost_value_from == 'random':
        # Use the provided initialization function
        cost_value_params = cost_value_init_fn()
    else:
        raise ValueError(f"Unknown init_cost_value_from: {init_cost_value_from}")

    # Create new params with cost_value
    # The exact structure depends on the flax dataclass used
    # We'll handle the common case where it's a PPONetworkParams dataclass
    try:
        return ppo_params.replace(cost_value=cost_value_params)
    except AttributeError:
        # Fallback: create a dict-like structure
        return type(ppo_params)(
            policy=ppo_params.policy,
            value=ppo_params.value,
            cost_value=cost_value_params,
        )


def benchmark_safety_transfer(
        env_name: str,
        unsafe_train_fn: Callable,
        safe_train_fns: Dict[str, Callable],
        unsafe_steps: int,
        safe_steps: int,
        train_kwargs: Optional[Dict[str, Any]] = None,
        unsafe_train_kwargs: Optional[Dict[str, Any]] = None,
        progress_fn: Optional[Callable] = None,
        seed: int = 0,
        wandb_config: Optional[Dict[str, Any]] = None,
        use_checkpoint_transfer: bool = True,
        checkpoint_dir: Optional[str] = None,
) -> Tuple[Any, Dict[str, TransferResult]]:
    """Benchmark safety transfer across multiple safe RL algorithms.

    This trains an unsafe policy (e.g., PPO) first, then transfers it to
    multiple safe RL algorithms to compare their adaptation performance.

    Args:
        env_name: Environment name.
        unsafe_train_fn: Training function for unsafe algorithm (e.g., ppo.train).
        safe_train_fns: Dict mapping algorithm names to training functions.
        unsafe_steps: Number of steps for unsafe pre-training.
        safe_steps: Number of steps for safe fine-tuning.
        train_kwargs: Common kwargs for all training functions.
        unsafe_train_kwargs: Additional kwargs only for unsafe training.
        progress_fn: Optional progress callback.
        seed: Random seed.
        wandb_config: Optional wandb configuration dict with keys:
            - enabled: bool, whether to use wandb
            - project: str, wandb project name
            - group: str, wandb group name (algorithms will be grouped together)
            - tags: list, wandb tags
            - base_name: str, base run name (algorithm name will be appended)
            - config: dict, config to log to wandb
        use_checkpoint_transfer: If True (default), save the unsafe model to a
            checkpoint file and load it using restore_checkpoint_path for safe
            fine-tuning. If False, forward params directly via pretrained_params.
        checkpoint_dir: Directory to save checkpoint files. If None, uses a
            temporary directory (cleaned up after transfer) or the directory
            from train_kwargs['save_checkpoint_path'] if provided.

    Returns:
        Tuple of (unsafe_params, dict mapping algorithm names to TransferResult).
    """
    train_kwargs = train_kwargs or {}
    unsafe_train_kwargs = unsafe_train_kwargs or {}

    # Setup checkpoint directory for transfer
    temp_checkpoint_dir = None
    if use_checkpoint_transfer:
        if checkpoint_dir is not None:
            transfer_checkpoint_dir = Path(checkpoint_dir).resolve()
        elif train_kwargs.get('save_checkpoint_path'):
            transfer_checkpoint_dir = Path(train_kwargs['save_checkpoint_path']).resolve() / 'transfer'
        else:
            # Create a temporary directory that will be cleaned up after transfer
            temp_checkpoint_dir = tempfile.mkdtemp(prefix='transfer_checkpoint_')
            transfer_checkpoint_dir = Path(temp_checkpoint_dir).resolve()
        transfer_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"Transfer checkpoint directory: {transfer_checkpoint_dir}")

    # Determine if wandb is enabled
    use_wandb = (wandb_config is not None and
                 wandb_config.get('enabled', False) and
                 _WANDB_AVAILABLE)

    print("=" * 70)
    print("SAFETY TRANSFER BENCHMARK")
    print("=" * 70)
    print(f"Environment: {env_name}")
    print(f"Unsafe pre-training: {unsafe_steps:,} steps")
    print(f"Safe fine-tuning: {safe_steps:,} steps per algorithm")
    print(f"Safe algorithms to test: {list(safe_train_fns.keys())}")
    print()

    # Create environment using difficulty level if provided
    env_kwargs = train_kwargs.get('env_kwargs', None) or {}
    difficulty = train_kwargs.get('difficulty', None)
    env = envs.get_environment(env_name, level=difficulty, **env_kwargs)
    eval_env = envs.get_environment(env_name, level=difficulty, **env_kwargs)

    # Determine the episode length
    episode_length = train_kwargs.get('episode_length') or getattr(env, 'episode_length', None)

    # Phase 1: Train unsafe policy
    print("[Phase 1] Training unsafe baseline policy")
    print("-" * 50)

    # Initialize wandb run for unsafe training
    if use_wandb:
        base_name = wandb_config.get('base_name', f'{env_name}_transfer')
        run_config = wandb_config.get('config', {})
        run_config['algorithm'] = 'ppo'
        run_config['phase'] = 'unsafe'
        wandb.init(
            project=wandb_config.get('project', 'safety-transfer'),
            name=f"{base_name}_ppo",
            id=f"{base_name}_ppo",
            group=wandb_config.get('group', env_name),
            tags=wandb_config.get('tags', []),
            config=run_config,
            job_type='unsafe_pretraining',
        )

    # Build kwargs for unsafe train function, filtering to its signature
    base_unsafe_kwargs = dict(train_kwargs)
    base_unsafe_kwargs.update(unsafe_train_kwargs)
    base_unsafe_kwargs['environment'] = env
    base_unsafe_kwargs['eval_env'] = eval_env
    base_unsafe_kwargs['num_timesteps'] = unsafe_steps
    base_unsafe_kwargs['seed'] = seed
    base_unsafe_kwargs['episode_length'] = episode_length
    unsafe_kwargs = filter_kwargs_for_fn(unsafe_train_fn, base_unsafe_kwargs)

    unsafe_metrics_history = []

    def unsafe_progress_fn(step, metrics):
        metrics['phase'] = 'unsafe'
        unsafe_metrics_history.append(metrics)
        if use_wandb and wandb.run is not None:
            wandb.log(metrics, step=step)
        if progress_fn:
            progress_fn(step, metrics)

    unsafe_kwargs['progress_fn'] = unsafe_progress_fn

    start_time = time.time()
    unsafe_make_policy, unsafe_params, unsafe_final_metrics, returned_eval_env = unsafe_train_fn(**unsafe_kwargs)
    unsafe_training_time = time.time() - start_time

    print(f"\nUnsafe training complete:")
    print(f"  Time: {unsafe_training_time:.1f}s")
    if 'eval/episode_reward' in unsafe_final_metrics:
        print(f"  Final reward: {unsafe_final_metrics['eval/episode_reward']:.2f}")
    if 'eval/episode_cost' in unsafe_final_metrics:
        print(f"  Final cost: {unsafe_final_metrics['eval/episode_cost']:.2f}")

    # Save checkpoint for transfer if using checkpoint-based transfer
    unsafe_checkpoint_path = None
    if use_checkpoint_transfer:
        from brax.training.agents.ppo import networks as ppo_networks
        unsafe_checkpoint_path = transfer_checkpoint_dir / 'unsafe_ppo'
        # Get observation and action sizes from environment
        obs_size = env.observation_size
        action_size = env.action_size
        # Create checkpoint config
        ckpt_config = ppo_checkpoint.network_config(
            observation_size=obs_size,
            action_size=action_size,
            normalize_observations=train_kwargs.get('normalize_observations', True),
            network_factory=ppo_networks.make_ppo_networks,
        )
        # Save the checkpoint
        ppo_checkpoint.save(
            path=str(unsafe_checkpoint_path),
            step=unsafe_steps,
            params=unsafe_params,
            config=ckpt_config,
        )
        print(f"  Saved unsafe checkpoint to: {unsafe_checkpoint_path}")

    # Record unsafe rollout video if requested
    if not train_kwargs.get('skip_video', False):
        video_env = returned_eval_env or eval_env
        video_length = train_kwargs.get('video_length') or train_kwargs.get('episode_length')
        if video_length is None:
            video_length = getattr(video_env, 'episode_length', None) or getattr(video_env, 'default_episode_length', None)
        out_name = (wandb_config.get('base_name', f'{env_name}_transfer') + '_ppo') if use_wandb else f'{env_name}_ppo_transfer'
        record_episode_video(
            env=video_env,
            make_inference_fn=unsafe_make_policy,
            params=unsafe_params,
            steps=video_length,
            cameras=train_kwargs.get('cameras'),
            width=train_kwargs.get('video_width'),
            height=train_kwargs.get('video_height'),
            fps=train_kwargs.get('video_fps'),
            frame_stride=train_kwargs.get('video_frame_stride'),
            out_name=out_name,
            log_to_wandb=use_wandb and wandb.run is not None,
            seed=seed,
            num_episodes=train_kwargs.get('num_video_episodes', 1),
        )

    # Finish wandb run for unsafe training
    if use_wandb and wandb.run is not None:
        wandb.finish()

    # Phase 2: Transfer to each safe algorithm
    print("\n[Phase 2] Safety Transfer Experiments")
    print("-" * 50)

    results = {}
    for algo_name, safe_train_fn in safe_train_fns.items():
        print(f"\n  Testing {algo_name}...")

        # Initialize wandb run for this safe algorithm
        if use_wandb:
            base_name = wandb_config.get('base_name', f'{env_name}_transfer')
            run_config = wandb_config.get('config', {}).copy()
            run_config['algorithm'] = algo_name
            run_config['phase'] = 'safe'
            run_config['pretrained_from'] = 'ppo'
            wandb.init(
                project=wandb_config.get('project', 'safety-transfer'),
                name=f"{base_name}_{algo_name}",
                id=f"{base_name}_{algo_name}",
                group=wandb_config.get('group', env_name),
                tags=wandb_config.get('tags', []),
                config=run_config,
                job_type='safety_finetuning',
            )

        base_safe_kwargs = dict(train_kwargs)
        base_safe_kwargs['environment'] = env
        base_safe_kwargs['eval_env'] = eval_env
        base_safe_kwargs['num_timesteps'] = safe_steps
        base_safe_kwargs['seed'] = seed
        base_safe_kwargs['episode_length'] = episode_length

        # Transfer parameters: either via checkpoint file or direct params
        if use_checkpoint_transfer and unsafe_checkpoint_path is not None:
            # Use checkpoint-based transfer (default): load from saved file
            # Find the checkpoint subdirectory (orbax saves with step number)
            ckpt_subdirs = list(unsafe_checkpoint_path.glob('*'))
            if ckpt_subdirs:
                # Use the checkpoint with the highest step number
                latest_ckpt = max(ckpt_subdirs, key=lambda p: int(p.name) if p.name.isdigit() else 0)
                base_safe_kwargs['restore_checkpoint_path'] = str(latest_ckpt)
                print(f"    Using checkpoint transfer from: {latest_ckpt}")
            else:
                # Fallback to direct params if checkpoint not found
                print(f"    Warning: Checkpoint not found at {unsafe_checkpoint_path}, using direct params")
                base_safe_kwargs['pretrained_params'] = unsafe_params
        else:
            # Direct parameter forwarding (legacy mode)
            base_safe_kwargs['pretrained_params'] = unsafe_params

        # Filter kwargs for the specific safe algorithm signature
        safe_kwargs = filter_kwargs_for_fn(safe_train_fn, base_safe_kwargs)

        safe_metrics_history = []

        # Use closure to capture algo_name correctly for each iteration
        def make_safe_progress_fn(alg_name, metrics_history):
            def safe_progress_fn(step, metrics):
                metrics['phase'] = 'safe'
                metrics['algorithm'] = alg_name
                metrics_history.append(dict(metrics))
                if use_wandb and wandb.run is not None:
                    wandb.log(metrics, step=step)
                if progress_fn:
                    progress_fn(step, metrics)
            return safe_progress_fn

        safe_kwargs['progress_fn'] = make_safe_progress_fn(algo_name, safe_metrics_history)

        start_time = time.time()
        safe_make_policy, safe_params, safe_final_metrics, returned_eval_env_safe = safe_train_fn(**safe_kwargs)
        safe_training_time = time.time() - start_time

        print(f"    {algo_name} complete:")
        print(f"      Time: {safe_training_time:.1f}s")
        if 'eval/episode_reward' in safe_final_metrics:
            print(f"      Final reward: {safe_final_metrics['eval/episode_reward']:.2f}")
        if 'eval/episode_cost' in safe_final_metrics:
            print(f"      Final cost: {safe_final_metrics['eval/episode_cost']:.2f}")

        # Record safe-phase rollout video if requested
        if not train_kwargs.get('skip_video', False):
            video_env = returned_eval_env_safe or eval_env
            video_length = train_kwargs.get('video_length') or train_kwargs.get('episode_length')
            if video_length is None:
                video_length = getattr(video_env, 'episode_length', None) or getattr(video_env, 'default_episode_length', None)
            out_name = (wandb_config.get('base_name', f'{env_name}_transfer') + f'_{algo_name}') if use_wandb else f'{env_name}_{algo_name}_transfer'
            record_episode_video(
                env=video_env,
                make_inference_fn=safe_make_policy,
                params=safe_params,
                steps=video_length,
                cameras=train_kwargs.get('cameras'),
                width=train_kwargs.get('video_width'),
                height=train_kwargs.get('video_height'),
                fps=train_kwargs.get('video_fps'),
                frame_stride=train_kwargs.get('video_frame_stride'),
                out_name=out_name,
                log_to_wandb=use_wandb and wandb.run is not None,
                seed=seed,
                num_episodes=train_kwargs.get('num_video_episodes', 1),
            )

        # Finish wandb run for this safe algorithm
        if use_wandb and wandb.run is not None:
            wandb.finish()

        results[algo_name] = TransferResult(
            algorithm=algo_name,
            unsafe_final_metrics=unsafe_final_metrics,
            safe_final_metrics=safe_final_metrics,
            unsafe_training_time=unsafe_training_time,
            safe_training_time=safe_training_time,
            safe_metrics_history=safe_metrics_history,
        )

    # Print summary
    print("\n" + "=" * 70)
    print("SAFETY TRANSFER RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nUnsafe baseline (after {unsafe_steps:,} steps):")
    if 'eval/episode_reward' in unsafe_final_metrics:
        print(f"  Reward: {unsafe_final_metrics['eval/episode_reward']:.2f}")
    if 'eval/episode_cost' in unsafe_final_metrics:
        print(f"  Cost: {unsafe_final_metrics['eval/episode_cost']:.2f}")

    print(f"\nSafe fine-tuning results (after {safe_steps:,} additional steps):")
    for algo_name, result in results.items():
        if result is None:
            print(f"  {algo_name}: FAILED")
            continue
        reward = result.safe_final_metrics.get('eval/episode_reward', float('nan'))
        cost = result.safe_final_metrics.get('eval/episode_cost', float('nan'))
        print(f"  {algo_name}: reward={reward:.2f}, cost={cost:.2f}")

    # Clean up temporary checkpoint directory if it was created
    if temp_checkpoint_dir is not None:
        try:
            shutil.rmtree(temp_checkpoint_dir)
            print(f"\nCleaned up temporary checkpoint directory: {temp_checkpoint_dir}")
        except Exception as e:
            print(f"\nWarning: Failed to clean up temporary directory {temp_checkpoint_dir}: {e}")

    return unsafe_params, results


def quick_transfer_comparison(
        env_name: str,
        total_budget: int = 10_000_000,
        unsafe_fraction: float = 0.5,
        seed: int = 0,
        **train_kwargs,
) -> Dict[str, TransferResult]:
    """Quick comparison of safety transfer across all available safe RL algorithms.

    Args:
        env_name: Environment name.
        total_budget: Total training budget (split between unsafe and safe phases).
        unsafe_fraction: Fraction of budget for unsafe pre-training.
        seed: Random seed.
        **train_kwargs: Additional kwargs passed to all training functions.

    Returns:
        Dict mapping algorithm names to TransferResult.
    """
    # Import training functions
    from brax.training.agents import ppo
    from brax.training.agents import ppo_lag
    from brax.training.agents import ppo_pid
    from brax.training.agents import focops
    from brax.training.agents import p3o

    unsafe_steps = int(total_budget * unsafe_fraction)
    safe_steps = total_budget - unsafe_steps

    _, results = benchmark_safety_transfer(
        env_name=env_name,
        unsafe_train_fn=ppo.train,
        safe_train_fns={
            'ppo_lag': ppo_lag.train,
            'ppo_pid': ppo_pid.train,
            'focops': focops.train,
            'p3o': p3o.train,
        },
        unsafe_steps=unsafe_steps,
        safe_steps=safe_steps,
        train_kwargs=train_kwargs,
        seed=seed,
    )

    return results
