"""
Training script for Safe-Brax experiments with configs.
Based on mourad_lag.ipynb training approach.
"""

import functools
import os
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import wandb
from crax import envs
from configs.training_config import build_base_parser
from run_utils import (
    collect_rollout_metrics, record_episode_video, setup_gpu_environment,
    get_algorithm_train_fn, filter_kwargs_for_fn, custom_progress_fn,
    make_vision_network_factory, morphology_override, VISION_CAMERA_OVERRIDES,
)


def main():
    """Main function to run training from command line."""
    parser = build_base_parser(description='Train Safe-Brax agents from config files')
    config = parser.parse_args()

    env_name = config.env_name
    alg_name = config.alg
    difficulty = config.difficulty
    use_wandb = config.use_wandb

    # Fill in the morphology-specific pixel-obs training camera, but only if
    # the user didn't explicitly pass --vision_camera
    if config.vision_camera is None:
        config.vision_camera = morphology_override(env_name, VISION_CAMERA_OVERRIDES) or 'vision'

    # Setup GPU environment
    setup_gpu_environment(vision=config.vision)

    # Run training for each seed
    for seed in config.seeds:
        print(f"\n{'=' * 50}")
        print(f"Running experiment with seed {seed}")
        print(f"{'=' * 50}\n")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        run_name = f"{env_name}_Level_{difficulty}_{alg_name}_seed{seed}_{timestamp}"

        # Build vision kwargs if vision mode is enabled. Pixel-obs wrapping
        # happens inside the training function (GpuPixelObservationWrapper
        # must be applied after env vmapping, not here at env construction time.
        vision_kwargs = None
        if config.vision:
            vision_kwargs = dict(
                camera=config.vision_camera,
                height=config.vision_height,
                width=config.vision_width,
                obs_mode=config.vision_obs_mode,
                frame_stack=config.vision_frame_stack,
            )
            print(
                f"Vision mode: GPU rendering (MJWarp), "
                f"camera='{config.vision_camera}', "
                f"{config.vision_width}x{config.vision_height}"
            )

        # Create environments with difficulty level
        env_kwargs = config.env_kwargs or {}
        if env_name == 'safe_velocity':
            env_kwargs['agent'] = config.agent
        if config.vision:
            # GpuPixelObservationWrapper reads geom_xpos/cam_xpos, which only
            # the MJX pipeline populates.
            env_kwargs.setdefault('backend', 'mjx')
        env = envs.get_environment(env_name, level=difficulty, **env_kwargs)
        eval_env = envs.get_environment(env_name, level=difficulty, **env_kwargs)

        # Determine the episode length
        episode_length = config.episode_length or env_kwargs.get('episode_length') or getattr(env, 'episode_length', None)

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

        # Inject vision network factory + pixel-obs wrapping kwargs if vision
        # mode is enabled. 'vision_kwargs' is only accepted by train_fns that
        # support GpuPixelObservationWrapper (ppo and its pass-throughs). For
        # others, it's silently dropped by `filter_kwargs_for_fn` below if the
        # target signature doesn't declare it, so re-filter after adding it.
        if config.vision:
            state_obs_key = 'state' if config.vision_obs_mode == 'pixels+state' else ''
            train_kwargs['network_factory'] = make_vision_network_factory(
                alg_name,
                policy_obs_key=state_obs_key,
                value_obs_key=state_obs_key,
            )
            train_kwargs['augment_pixels'] = True
            train_kwargs['vision_kwargs'] = vision_kwargs
            train_kwargs = filter_kwargs_for_fn(train_fn_base, train_kwargs)
            if 'vision_kwargs' not in train_kwargs:
                raise ValueError(
                    f"--vision was set but algorithm '{alg_name}' does not "
                    f"support pixel observations (its train() has no "
                    f"'vision_kwargs' parameter)."
                )

        # Create the training function
        train_fn = functools.partial(train_fn_base, **train_kwargs)

        # Train the agent
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
                level=config.difficulty,
                env_kwargs=config.env_kwargs,
            )

        if not config.skip_video:
            video_length = config.video_length if config.video_length else config.episode_length
            if video_length is None:
                video_length = getattr(eval_env, 'episode_length', None) or getattr(eval_env, 'default_episode_length', None)
            # Use a non-vision env for video recording. Video frames are
            # rendered from pipeline_state via brax.io.image (main thread),
            # avoiding EGL threading issues from pure_callback.
            # The inference fn is wrapped to feed the state-only obs through
            # the vision policy with zeroed-out pixel channels.
            video_env = envs.get_environment(
                env_name, level=difficulty, **env_kwargs,
            )
            video_inference_fn = make_inference_fn
            if config.vision:
                _raw_make_policy = make_inference_fn
                # GpuPixelObservationWrapper is single-camera (config.vision_camera)
                # and always RGB (see GpuPixelObservationWrapper._channels).
                _pixel_keys = [f'pixels/{config.vision_camera}']
                _pixel_shape = (config.vision_height, config.vision_width, 3)

                def _vision_inference_fn(params, **kwargs):
                    inner_policy = _raw_make_policy(params, **kwargs)
                    def _wrapped_policy(obs, key):
                        # Build dict obs with zeroed pixels + real state
                        dict_obs = {k: jnp.zeros(_pixel_shape, dtype=jnp.uint8)
                                    for k in _pixel_keys}
                        dict_obs['state'] = obs
                        return inner_policy(dict_obs, key)
                    return _wrapped_policy
                video_inference_fn = _vision_inference_fn

            record_episode_video(
                env=video_env,
                make_inference_fn=video_inference_fn,
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
