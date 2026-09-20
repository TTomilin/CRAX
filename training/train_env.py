"""
Training script for Safe-Brax experiments with configs.
Based on mourad_lag.ipynb training approach.
"""

import functools
import os
from datetime import datetime
from pathlib import Path

import numpy as np

import wandb
from crax import envs
from training.config import build_base_parser
from training.run_utils import (
    collect_rollout_metrics, record_episode_video, setup_gpu_environment,
    get_algorithm_train_fn, filter_kwargs_for_fn, custom_progress_fn,
    make_vision_network_factory, morphology_override, VISION_CAMERA_OVERRIDES,
    make_periodic_vision_video_fn,
)
from crax.envs.limb_colors import colorize_env_limbs


# Algorithms whose vision networks own a cost value head (the only ones the privileged-camera conditions can be run with)
COST_VALUE_ALGS = {'ppo_lag', 'ppo_pid', 'focops', 'p3o', 'crpo'}


def resolve_privilege_routing(config, alg_name):
    """
    Returns (cameras, routing): `cameras` is the tuple of MJ cameras to render and `routing` is dict of per-head `pixels/<camera>` key tuples + state_obs_key for state-oracle.
    """
    mode = config.vision_privilege_mode
    ego_camera = config.vision_camera
    priv_camera = config.vision_privileged_camera

    if mode == 'none':
        if not getattr(config, 'vision_independent_encoders', False):
            return (ego_camera,), {}
        ego_key = f'pixels/{ego_camera}'
        return (ego_camera,), dict(
            policy_pixel_keys=(ego_key,),
            value_pixel_keys=(ego_key,),
            cost_value_pixel_keys=(ego_key,),
            share_encoder=False,
        )

    if not config.vision:
        raise ValueError(
            f"--vision_privilege_mode '{mode}' requires --vision: the "
            f"privileged-camera conditions only exist for pixel observations."
        )
    if alg_name not in COST_VALUE_ALGS:
        raise ValueError(
            f"--vision_privilege_mode '{mode}' requires an algorithm with a "
            f"cost value network (one of {sorted(COST_VALUE_ALGS)}), but the "
            f"selected algorithm is '{alg_name}'."
        )

    if mode == 'state_oracle':
        if config.vision_obs_mode != 'pixels+state':
            raise ValueError(
                "--vision_privilege_mode 'state_oracle' gives the cost critic "
                "the state vector instead of pixels, so it requires "
                f"--vision_obs_mode pixels+state (got "
                f"'{config.vision_obs_mode}')."
            )
    else:
        if config.vision_obs_mode != 'pixels':
            raise ValueError(
                f"--vision_privilege_mode '{mode}' requires "
                "--vision_obs_mode pixels so simulator state cannot leak into "
                "the actor or critics and confound the camera ablation."
            )
        if not priv_camera:
            raise ValueError(
                f"--vision_privilege_mode '{mode}' requires "
                f"--vision_privileged_camera to name the extra camera to "
                f"render for the critic(s)."
            )
        if priv_camera == ego_camera:
            raise ValueError(
                f"--vision_privileged_camera '{priv_camera}' is the same as "
                f"--vision_camera, so the critic would gain no extra "
                f"information; pick a different camera."
            )

    ego_key = f'pixels/{ego_camera}'
    priv_key = f'pixels/{priv_camera}'

    if mode == 'state_oracle':
        cameras = (ego_camera,)
        routing = dict(
            policy_obs_key='',
            value_obs_key='',
            policy_pixel_keys=(ego_key,),
            value_pixel_keys=(ego_key,),
            cost_value_pixel_keys=(),
            cost_value_obs_key='state',
        )
    else:
        cameras = (ego_camera, priv_camera)
        per_mode = {
            'cost': ((ego_key,), (ego_key,), (ego_key, priv_key)),
            'all_critics': ((ego_key,), (ego_key, priv_key), (ego_key, priv_key)),
            'reward': ((ego_key,), (ego_key, priv_key), (ego_key,)),
        }
        policy_keys, value_keys, cost_value_keys = per_mode[mode]
        routing = dict(
            policy_pixel_keys=policy_keys,
            value_pixel_keys=value_keys,
            cost_value_pixel_keys=cost_value_keys,
        )

    # This is required, but can fuse those that only receive egocentric obs
    routing['share_encoder'] = False
    return cameras, routing


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

    # Resolve priviledged camera
    privilege_mode = config.vision_privilege_mode
    cameras, pixel_routing = resolve_privilege_routing(config, alg_name)

    # Setup GPU environment
    setup_gpu_environment(vision=config.vision)

    # Run training for each seed
    for seed in config.seeds:
        print(f"\n{'=' * 50}")
        print(f"Running experiment with seed {seed}")
        print(f"{'=' * 50}\n")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        run_name = f"{env_name}_Level_{difficulty}_{alg_name}_seed{seed}_{timestamp}"

        # Build vision kwargs. Pixel-obs wrapping happens inside the training function
        vision_kwargs = None
        if config.vision:
            vision_kwargs = dict(
                camera=config.vision_camera,
                height=config.vision_height,
                width=config.vision_width,
                obs_mode=config.vision_obs_mode,
                frame_stack=config.vision_frame_stack,
            )
            if privilege_mode != 'none':
                # For multi-camera rendering
                vision_kwargs['cameras'] = cameras
            print(
                f"Vision mode: GPU rendering (MJWarp), "
                f"camera='{config.vision_camera}', "
                f"{config.vision_width}x{config.vision_height}"
            )
            #Debug
            if pixel_routing:
                print(
                    f"Vision mode: privilege='{privilege_mode}', "
                    f"cameras={cameras}, per-head encoders (share_encoder=False), "
                    f"policy={pixel_routing['policy_pixel_keys']}, "
                    f"reward_value={pixel_routing['value_pixel_keys']}, "
                    f"cost_value={pixel_routing['cost_value_pixel_keys']}, "
                    f"policy_state={pixel_routing.get('policy_obs_key', '')!r}, "
                    f"reward_state={pixel_routing.get('value_obs_key', '')!r}, "
                    f"cost_state={pixel_routing.get('cost_value_obs_key', '')!r}"
                )

        # Create environments with a difficulty level
        env_kwargs = config.env_kwargs or {}
        if env_name == 'safe_velocity':
            env_kwargs['agent'] = config.agent
        if config.vision:
            # GpuPixelObservationWrapper reads geom_xpos/cam_xpos, which only the MJX pipeline populates.
            env_kwargs.setdefault('backend', 'mjx')
        env = envs.get_environment(env_name, level=difficulty, **env_kwargs)
        eval_env = envs.get_environment(env_name, level=difficulty, **env_kwargs)

        # Distinct per-limb colours in the pixel observations
        if config.vision and config.vision_limb_colors:
            n_colored = colorize_env_limbs(env, env_name)
            colorize_env_limbs(eval_env, env_name)
            if n_colored:
                print(f"Vision mode: recoloured {n_colored} limb geoms for pixel observations.")
            else:
                print(f"Vision mode: --vision_limb_colors set but '{env_name}' has no limb "
                      f"colour scheme (see crax/envs/limb_colors.py); left unchanged.")

        # Determine the episode length
        episode_length = config.episode_length or env_kwargs.get('episode_length') or getattr(env, 'episode_length', None)

        # Periodic mid-training video, --vision only: a dedicated single-env vision-wrapped
        # rollout env, reading frames straight off its own GPU (MJWarp) pixel observations.
        video_fn = None
        if config.vision and not config.skip_video:
            video_env_kwargs = {k: v for k, v in env_kwargs.items() if k != 'episode_length'}
            # Re-colour inside `pre_vision_fn`, not on the returned env
            use_limb_colors = config.vision and config.vision_limb_colors
            periodic_video_env = envs.create(
                env_name, level=difficulty,
                episode_length=episode_length,
                auto_reset=True,
                batch_size=1,
                vision=True,
                vision_kwargs=dict(**vision_kwargs, num_envs=1),
                pre_vision_fn=(lambda e: colorize_env_limbs(e, env_name)) if use_limb_colors else None,
                **video_env_kwargs,
            )
            video_fn = make_periodic_vision_video_fn(
                periodic_video_env,
                every_steps=config.video_every_steps,
                steps=config.periodic_video_steps,
                num_episodes=config.num_video_episodes,
                pixel_camera=config.vision_camera,
                # Same extra-camera set vector-obs training's end-of-run video
                # uses (config.cameras, default ["fixedfar", "vision"]) minus
                # whichever one is already pixel_camera's own clip.
                extra_cameras=[c for c in config.cameras if c != config.vision_camera],
                frame_stack=config.vision_frame_stack,
                width=config.video_width,
                height=config.video_height,
                fps=config.video_fps,
                run_name=run_name,
                deterministic=config.deterministic_eval,
                log_to_wandb=config.use_wandb,
                seed=seed,
            )

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
            root_dir = Path(__file__).parent.parent.resolve()  # repo root, not training/
            ckpt_root = root_dir / config.model_dir / run_name
            os.makedirs(ckpt_root, exist_ok=True)
            cfg["save_checkpoint_path"] = ckpt_root

        # Setup metrics collection
        progress_fn = functools.partial(custom_progress_fn, use_wandb=use_wandb, verbose=not config.quiet)

        # Get the appropriate training function
        train_fn_base = get_algorithm_train_fn(alg_name)
        train_kwargs = filter_kwargs_for_fn(train_fn_base, cfg)

        # Inject vision network factory + pixel-obs wrapping kwargs if vision mode is enabled
        if config.vision:
            state_obs_key = 'state' if config.vision_obs_mode == 'pixels+state' else ''
            network_routing = dict(pixel_routing)
            network_routing.setdefault('policy_obs_key', state_obs_key)
            network_routing.setdefault('value_obs_key', state_obs_key)
            train_kwargs['network_factory'] = make_vision_network_factory(
                alg_name,
                **network_routing,
            )
            train_kwargs['augment_pixels'] = config.vision_augment
            train_kwargs['vision_kwargs'] = vision_kwargs
            if video_fn is not None:
                train_kwargs['policy_params_fn'] = video_fn
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
            if config.vision:
                # Force one last clip from the actual final params
                video_fn(int(config.num_timesteps), make_inference_fn, params, force=True)
            else:
                video_length = config.video_length if config.video_length else config.episode_length
                if video_length is None:
                    video_length = getattr(eval_env, 'episode_length', None) or getattr(eval_env, 'default_episode_length', None)
                video_env = envs.get_environment(
                    env_name, level=difficulty, **env_kwargs,
                )
                record_episode_video(
                    env=video_env,
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
