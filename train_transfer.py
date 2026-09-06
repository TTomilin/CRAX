#!/usr/bin/env python
"""Example: Safety transfer from PPO to safe RL algorithms.

This script demonstrates how to:
1. Train an unsafe PPO policy
2. Transfer it to multiple safe RL algorithms (PPO-Lag, PPO-PID, FOCOPS, P3O)
3. Compare how well each algorithm adapts the unsafe policy to satisfy safety constraints

Example usage:
    python train_transfer.py --env_name safe_velocity_ant --unsafe_steps 2000000 --safe_steps 2000000
"""

import functools
import os
from datetime import datetime
from pathlib import Path

from crax.training import transfer
from configs.training_config import bool_type, build_base_parser
from run_utils import (
    setup_gpu_environment, get_algorithm_train_fn, custom_progress_fn,
    make_vision_network_factory,
)
import wandb


def main():
    parser = build_base_parser(description='Safety transfer benchmark')
    parser.add_argument('--unsafe_steps', type=float, default=1e8, help='Steps for unsafe pre-training')
    parser.add_argument('--safe_steps', type=float, default=1e8, help='Steps for safe fine-tuning per algorithm')
    parser.add_argument('--algorithms', type=str, nargs='+', default=['ppo_lag', 'ppo_pid', 'focops', 'p3o', 'crpo'],
                        help='Safe algorithms to test')
    parser.add_argument('--use_checkpoint_transfer', type=bool_type, nargs='?', const=True, default=True,
                        help='If True, save unsafe model to checkpoint and load via restore_checkpoint_path. '
                             'If False, forward params directly (legacy mode).')
    parser.add_argument('--no_checkpoint_transfer', dest='use_checkpoint_transfer', action='store_false',
                        help='Disable checkpoint-based transfer (use direct param forwarding)')
    parser.add_argument('--transfer_checkpoint_dir', type=str, default="models",
                        help='Directory to save transfer checkpoint')
    args = parser.parse_args()

    # Setup GPU environment
    setup_gpu_environment()

    # Build safe algorithm train fns via utility, and unsafe as base PPO
    safe_train_fns = {name: get_algorithm_train_fn(name) for name in args.algorithms}
    unsafe_train_fn = get_algorithm_train_fn('ppo')

    # Run experiments per seed
    for seed in args.seeds:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        base_run_name = f"{args.env_name}_transfer_seed{seed}_{timestamp}"

        # Prepare full config dict
        cli_cfg = vars(args)
        runtime_cfg = {"seed": seed, "timestamp": timestamp}
        cfg = {**cli_cfg, **runtime_cfg}

        # Optional checkpoint dir
        if args.store_model:
            root_dir = Path(__file__).parent.resolve()
            ckpt_root = root_dir / args.model_dir / base_run_name
            os.makedirs(ckpt_root, exist_ok=True)
            cfg["save_checkpoint_path"] = ckpt_root

        # Add vision support
        if args.vision:
            vision_kwargs = dict(
                cameras=args.vision_cameras,
                height=args.vision_height,
                width=args.vision_width,
                obs_mode=args.vision_obs_mode,
                frame_stack=args.vision_frame_stack,
                grayscale=args.vision_grayscale,
                num_render_workers=args.vision_render_workers,
            )
            cfg['vision'] = True
            cfg['vision_kwargs'] = vision_kwargs
            state_obs_key = 'state' if args.vision_obs_mode == 'pixels+state' else ''
            # Network factory for unsafe (PPO) and safe algorithms
            cfg['network_factory'] = make_vision_network_factory(
                'ppo',  # unsafe phase uses base PPO
                policy_obs_key=state_obs_key,
                value_obs_key=state_obs_key,
            )
            cfg['augment_pixels'] = True

        # Build wandb config for per-algorithm runs
        wandb_config = None
        if args.use_wandb:
            wandb_config = {
                'enabled': True,
                'project': args.wandb_project,
                'group': args.wandb_group if args.wandb_group else f"{args.env_name}_transfer",
                'tags': args.wandb_tags or [],
                'base_name': base_run_name,
                'config': cfg,
            }

        # Progress logger (no wandb logging here - handled per-algorithm in transfer module)
        progress = functools.partial(custom_progress_fn, use_wandb=False, verbose=not args.quiet)

        # Execute benchmark (transfer module will manage wandb runs per algorithm)
        unsafe_params, results = transfer.benchmark_safety_transfer(
            env_name=args.env_name,
            unsafe_train_fn=unsafe_train_fn,
            safe_train_fns=safe_train_fns,
            unsafe_steps=int(args.unsafe_steps),
            safe_steps=int(args.safe_steps),
            train_kwargs=cfg,
            progress_fn=progress,
            seed=seed,
            wandb_config=wandb_config,
            use_checkpoint_transfer=args.use_checkpoint_transfer,
            checkpoint_dir=args.transfer_checkpoint_dir,
        )

        # Detailed results
        print("\n" + "=" * 70)
        print("DETAILED RESULTS")
        print("=" * 70)

        for algo_name, result in results.items():
            if result is None:
                print(f"\n{algo_name}: FAILED")
                continue

            print(f"\n{algo_name}:")
            print(f"  Unsafe baseline:")
            print(f"    Reward: {result.unsafe_final_metrics.get('eval/episode_reward', 'N/A')}")
            print(f"    Cost: {result.unsafe_final_metrics.get('eval/episode_cost', 'N/A')}")
            print(f"  After safety fine-tuning:")
            print(f"    Reward: {result.safe_final_metrics.get('eval/episode_reward', 'N/A')}")
            print(f"    Cost: {result.safe_final_metrics.get('eval/episode_cost', 'N/A')}")
            print(f"  Training time: {result.safe_training_time:.1f}s")

            # Check if algorithm achieved constraint satisfaction
            safe_cost = result.safe_final_metrics.get('eval/episode_cost', float('inf'))
            if safe_cost <= cfg.get('safety_bound', float('inf')):
                print(f"  Status: SAFE (cost {safe_cost:.1f} <= bound {cfg.get('safety_bound')})")
            else:
                print(f"  Status: UNSAFE (cost {safe_cost:.1f} > bound {cfg.get('safety_bound')})")


        # Finish wandb run if active (should already be finished inside transfer module)
        if wandb.run is not None:
            wandb.finish()


if __name__ == '__main__':
    main()
