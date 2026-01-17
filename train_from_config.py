"""
Training script for Safe-Brax experiments with configs.
Based on mourad_lag.ipynb training approach.
"""

import argparse
import functools
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import wandb

from brax import envs
from brax.envs import Env
from brax.envs.difficulty import apply_difficulty
from configs.training_config import build_base_parser
from run_utils import collect_rollout_metrics, record_episode_video, setup_gpu_environment, get_algorithm_train_fn, \
    filter_kwargs_for_fn, custom_progress_fn


def main():
    """Main function to run training from command line."""
    parser = build_base_parser(description='Train Safe-Brax agents from config files')
    config = parser.parse_args()

    env_name = config.env_name
    alg_name = config.alg
    difficulty = config.difficulty
    use_wandb = config.use_wandb

    # Setup GPU environment
    setup_gpu_environment()

    # Run training for each seed
    for seed in config.seeds:
        print(f"\n{'=' * 50}")
        print(f"Running experiment with seed {seed}")
        print(f"{'=' * 50}\n")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        run_name = f"{env_name}_Level_{difficulty}_{alg_name}_seed{seed}_{timestamp}"

        # Create environments
        adjusted_env_kwargs = apply_difficulty(env_name, config.env_kwargs, difficulty)
        env = envs.get_environment(env_name, **adjusted_env_kwargs)
        eval_env = envs.get_environment(env_name, **adjusted_env_kwargs)

        # Determine the episode length
        episode_length = adjusted_env_kwargs.get('episode_length') or getattr(env, 'episode_length', None)

        print(f"Training environment '{env_name}' instantiated with difficulty {difficulty}.")
        print(f"Evaluation environment '{env_name}' instantiated with difficulty {difficulty}.")

        cli_cfg = vars(config)
        runtime_cfg = {"seed": seed, "timestamp": timestamp, "episode_length": episode_length}
        cfg = {**cli_cfg, **runtime_cfg}

        if use_wandb:
            # Prepare wandb config
            wandb_config = cfg.copy()
            wandb_project = config.wandb_project
            wandb_group = config.wandb_group if config.wandb_group else env_name
            wandb_tags = config.wandb_tags

            # Initialize wandb
            wandb.init(
                project=wandb_project,
                name=run_name,
                id=run_name,
                config=wandb_config,
                group=wandb_group,
                job_type=alg_name,
                tags=wandb_tags,
            )

        if config.store_model:
            root_dir = Path(__file__).parent.resolve()
            ckpt_root = root_dir / config.model_dir / run_name
            os.makedirs(ckpt_root, exist_ok=True)
            cfg["save_checkpoint_path"] = ckpt_root

        # Setup metrics collection
        progress_fn = functools.partial(custom_progress_fn, use_wandb=use_wandb, verbose=not config.quiet)

        # Get the appropriate training function
        train_fn_base = get_algorithm_train_fn(alg_name)
        train_kwargs = filter_kwargs_for_fn(train_fn_base, cfg)

        # Create the training function
        train_fn = functools.partial(train_fn_base, **train_kwargs)

        # Train the agent
        print(f"Starting {alg_name} training for {env_name}...")
        make_inference_fn, params, final_metrics, eval_env = train_fn(
            environment=env,
            eval_env=eval_env,
            progress_fn=progress_fn
        )
        print("Training finished.")

        # Log final metrics to wandb
        if use_wandb and wandb.run is not None and final_metrics:
            final_log_data = {}
            for key, value in final_metrics.items():
                if value is not None:
                    if isinstance(value, (np.ndarray,)) and value.ndim > 0:
                        value = value.mean()
                    final_log_data[key] = value
            if final_log_data:
                wandb.log(final_log_data, step=int(config.num_timesteps))

        if not config.skip_rollout:
            print(f"\nPerforming rollout evaluation...")
            rollout_metrics = collect_rollout_metrics(
                env_name=env_name,
                make_inference_fn=make_inference_fn,
                params=params,
                num_steps=config.rollout_steps,
                seed=seed,
                save_trajectory=True,
                save_plots=True,
                env_kwargs=apply_difficulty(env_name, config.env_kwargs, config.difficulty)
            )

        if not config.skip_video:
            video_length = config.video_length if config.video_length else config.episode_length
            if video_length is None:
                video_length = getattr(eval_env, 'default_episode_length', None)
            record_episode_video(
                env=eval_env,
                make_inference_fn=make_inference_fn,
                params=params,
                steps=video_length,
                cameras=config.cameras,
                width=config.video_width,
                height=config.video_height,
                fps=config.video_fps,
                frame_stride=config.video_frame_stride,
                out_name=run_name,
                log_to_wandb=config.use_wandb,
                seed=seed,
                num_episodes=config.num_video_episodes,
            )

        # Finish wandb run if active
        if config.use_wandb and wandb.run is not None:
            wandb.finish()

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
