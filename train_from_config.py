"""
Training script for Safe-Brax experiments with configs.
Based on mourad_lag.ipynb training approach.
"""

import argparse
import functools
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import wandb

from brax import envs
from brax.envs import Env
from brax.envs.difficulty import apply_difficulty
from configs.training_config import build_base_parser
from run_utils import collect_rollout_metrics, record_episode_video, setup_gpu_environment, get_algorithm_train_fn, \
    filter_kwargs_for_fn

# Global metrics buffer instance
metrics_buffer = []


def custom_progress_fn(num_steps: int, metrics: Dict[str, Any], use_wandb: bool = False, verbose: bool = True) -> None:
    """
    Progress function to print metrics and log to Weights & Biases.

    Args:
        num_steps: Current training step
        metrics: Metrics dictionary
        use_wandb: Whether to use wandb logging
        verbose: Whether to print metrics to console
    """
    global metrics_buffer

    if verbose:
        print(f"Step {num_steps}:")

    log_data = {}
    for key, value in metrics.items():
        if verbose and any(tok in key for tok in ("lambda", "cost", "constraint", "reward")):
            print(f"  {key}: {value}")
        log_data[key] = value

    metrics_buffer.append({"step": num_steps, **log_data})

    if use_wandb and wandb.run is not None and log_data:
        for row in metrics_buffer:
            wandb.log(log_data, step=row["step"])

        # clear the logged history from the buffer
        metrics_buffer.clear()


def train_from_config(config: argparse.Namespace, seed: int, use_wandb: bool = True,
                      verbose: bool = True) -> tuple[Any, Any, Any, Env]:
    """
    Train an agent using the provided configuration.

    Returns:
        Tuple of (make_inference_fn, params, final_eval_metrics)
    """
    env_name = config.env_name
    alg_name = config.alg
    difficulty = config.difficulty
    run_name = f"{env_name}_Level_{difficulty}_{alg_name}_seed{seed}_{config.timestamp}"

    # Create environments
    adjusted_env_kwargs = apply_difficulty(env_name, config.env_kwargs, difficulty)
    train_environment = envs.get_environment(env_name, **adjusted_env_kwargs)
    eval_env = envs.get_environment(env_name, **adjusted_env_kwargs)

    print(f"Training environment '{env_name}' instantiated with difficulty {difficulty}.")
    print(f"Evaluation environment '{env_name}' instantiated with difficulty {difficulty}.")

    cli_cfg = vars(config)
    runtime_cfg = {"seed": seed, "timestamp": config.timestamp}
    cfg = {**cli_cfg, **runtime_cfg}

    cfg = vars(config)
    cfg['seed'] = seed
    if use_wandb:
        # Prepare wandb config
        wandb_config = cfg.copy()
        wandb_config['seed'] = seed

        # Initialize wandb
        wandb_project = config.wandb_project
        wandb_group = config.wandb_group if config.wandb_group else env_name
        wandb_tags = config.wandb_tags

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
    progress_fn = functools.partial(custom_progress_fn, use_wandb=use_wandb, verbose=verbose)

    # Get the appropriate training function
    train_fn_base = get_algorithm_train_fn(alg_name)
    train_kwargs = filter_kwargs_for_fn(train_fn_base, cfg)

    # Create the training function
    train_fn = functools.partial(train_fn_base, **train_kwargs)

    # Train the agent
    print(f"Starting {alg_name} training for {env_name}...")
    make_inference_fn, params, final_eval_metrics, eval_env = train_fn(
        environment=train_environment,
        eval_env=eval_env,
        progress_fn=progress_fn
    )
    print("Training finished.")

    # Log final metrics to wandb
    if use_wandb and wandb.run is not None and final_eval_metrics:
        final_log_data = {}
        for key, value in final_eval_metrics.items():
            if value is not None:
                final_log_data[key] = value
        if final_log_data:
            wandb.log(final_log_data, step=int(float(np.asarray(config.num_timesteps).reshape(()))))

    return make_inference_fn, params, final_eval_metrics, eval_env


def main():
    """Main function to run training from command line."""
    parser = build_base_parser(description='Train Safe-Brax agents from config files')
    config = parser.parse_args()

    # Setup GPU environment
    setup_gpu_environment()

    # Run training for each seed
    for seed in config.seeds:
        print(f"\n{'=' * 50}")
        print(f"Running experiment with seed {seed}")
        print(f"{'=' * 50}\n")

        # Train the agent
        make_inference_fn, params, final_metrics, eval_env = train_from_config(
            config=config,
            seed=seed,
            use_wandb=config.use_wandb,
            verbose=not config.quiet
        )

        # Perform rollout evaluation if not skipped
        if not config.skip_rollout:
            print(f"\nPerforming rollout evaluation...")
            rollout_env_name = config.env_name
            rollout_metrics = collect_rollout_metrics(
                env_name=rollout_env_name,
                make_inference_fn=make_inference_fn,
                params=params,
                num_steps=config.rollout_steps,
                seed=seed,
                save_trajectory=True,
                save_plots=True,
                env_kwargs=apply_difficulty(rollout_env_name, config.env_kwargs, config.difficulty)
            )

        if not config.skip_video:
            video_length = config.video_length if config.video_length else config.episode_length
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
                out_name=f"{config.env_name}_{config.alg}_seed{seed}",
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
