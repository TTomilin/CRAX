"""
Shared training configuration and CLI argument definitions.

This module centralizes the common configuration/arguments that were
previously defined in `train_from_config.py` so other scripts can import
and reuse them consistently (e.g., examples/curriculum_training.py and
examples/safety_transfer.py), and extend with script‑specific flags.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict


def _json_type(s):
    if s is None or isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON string for env_kwargs: {e}")


DEFAULT_ENV_KWARGS: Dict[str, Any] = {
    "physics": {
        "backend": "mjx",
        "timestep": 0.02,
        "n_frames": 4,
    },
    "cost": {
        "scaler": 1.0,
        "collision": 3.0,
        "ctrl_cost_weight": 0.001,
    },
    "reward": {
        "reward_goal": 1.0,
        "scaler": 0.0,
    },
}


def add_shared_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds shared/common CLI arguments used across training scripts.

    Scripts may call this and then add their own specific arguments.
    """
    # --- Core ---
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Random seeds")

    # --- Experiment Control ---
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    parser.add_argument("--skip_rollout", action="store_true", help="Skip rollout evaluation after training")
    parser.add_argument("--skip_video", action="store_true", help="Skip video recording after training")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save model parameters")
    parser.add_argument(
        "--out_dir", type=str, default="runs/experimental_results", help="Directory for metrics/outputs"
    )
    parser.add_argument("--store_model", type=bool, default=True, help="Store model checkpoint after training")

    # --- Environment ---
    parser.add_argument("--env_name", type=str, default="safe_goal_point", help="Env name")
    parser.add_argument(
        "--agent",
        type=str,
        default="ant",
        choices=['ant', 'halfcheetah', 'hopper', 'humanoid', 'swimmer', 'walker2d'],
        help="Agent for safe_velocity env",
    )
    parser.add_argument(
        "--difficulty", type=int, choices=[1, 2, 3], default=1, help="Difficulty level (1=easy, 2=medium, 3=hard)"
    )
    parser.add_argument(
        "--env_kwargs",
        type=_json_type,
        default=None,
        help="JSON dict string for env kwargs to override env defaults (if not specified, env uses its own defaults)",
    )

    # --- Algorithm ---
    parser.add_argument("--alg", type=str, default="ppo_lag", help="Algorithm name (e.g., ppo, ppo_lag)")
    parser.add_argument("--max_devices_per_host", type=int, default=None, help="Limit devices per host")

    # --- Training Scale / Rollout ---
    parser.add_argument("--num_timesteps", type=float, default=100_000_000, help="Total training timesteps")
    parser.add_argument("--episode_length", type=int, default=None, help="Episode length; if None, use the task's default")
    parser.add_argument("--num_envs", type=int, default=2048, help="Number of parallel envs")
    parser.add_argument("--unroll_length", type=int, default=8, help="Unroll length")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--num_minibatches", type=int, default=32, help="Number of minibatches")
    parser.add_argument("--num_updates_per_batch", type=int, default=6, help="SGD updates per batch")
    parser.add_argument(
        "--rollout-steps", dest="rollout_steps", type=int, default=2000, help="Steps for post-training rollout"
    )

    # --- Optimization / PPO Core ---
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--entropy_cost", type=float, default=5e-3, help="Entropy coefficient")
    parser.add_argument("--discounting", type=float, default=0.99, help="Discount factor (gamma)")
    parser.add_argument("--reward_scaling", type=float, default=0.1, help="Reward scaling")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clipping_epsilon", type=float, default=0.3, help="PPO clipping epsilon")
    parser.add_argument(
        "--normalize_observations", type=bool, default=True, help="Normalize observations (true/false)"
    )

    # --- Evaluation / Logging cadence ---
    parser.add_argument("--num_evals", type=int, default=5, help="Number of eval passes during training")
    parser.add_argument("--num_eval_envs", type=int, default=128, help="Parallel envs during eval")
    parser.add_argument("--deterministic_eval", type=bool, default=False, help="Deterministic eval policy")
    parser.add_argument(
        "--training_metrics_steps", type=float, default=1e6, help="Env steps between training metrics logs"
    )

    # --- PPO-Lagrange ---
    parser.add_argument("--safety_bound", type=float, default=25.0, help="Episodic safety constraint bound")
    parser.add_argument("--lagrangian_coef_rate", type=float, default=10.0, help="Lagrange multiplier LR")
    parser.add_argument("--initial_lambda_lagr", type=float, default=0.0, help="Initial lambda value")

    # --- PPO-PID Lagrange ---
    parser.add_argument("--pid_kp", type=float, default=10.0, help="PID: proportional gain")
    parser.add_argument("--pid_ki", type=float, default=0.01, help="PID: integral gain")
    parser.add_argument("--pid_kd", type=float, default=0.01, help="PID: derivative gain")
    parser.add_argument("--pid_integral_clip", type=float, default=1.0, help="PID: anti-windup cap for integral term")
    parser.add_argument("--pid_lambda_clip", type=float, default=1e6, help="PID: clamp for lambda")
    parser.add_argument("--pid_deriv_ema_beta", type=float, default=0.95, help="PID: derivative EMA smoothing")

    # --- PPO-Saute ---
    parser.add_argument(
        "--saute-gamma-budget", dest="gamma_budget", type=float, default=None,
        help="Budget discount factor; defaults to --discounting if None",
    )
    parser.add_argument(
        "--saute-violation-penalty", dest="violation_penalty", type=float, default=-1.0,
        help="Terminal penalty added on violation step",
    )
    parser.add_argument(
        "--saute-normalize-budget-obs", dest="normalize_budget_obs", type=int, default=1,
        help="Normalize budget observation by initial budget",
    )

    # --- PPO-Cost ---
    parser.add_argument("--cost-weight", type=float, default=1.0, help="Cost weight used for PPO-C verify logging")

    # --- FOCOPS ---
    parser.add_argument("--initial_nu", type=float, default=0.1, help="FOCOPS: initial value of nu (constraint multiplier)")
    parser.add_argument("--nu_lr", type=float, default=1.0, help="FOCOPS: learning rate for nu updates")
    parser.add_argument("--nu_max", type=float, default=100.0, help="FOCOPS: maximum value for nu")
    parser.add_argument("--focops_lam", type=float, default=1.5, help="FOCOPS: KL penalty coefficient lambda")
    parser.add_argument("--focops_eta", type=float, default=0.02, help="FOCOPS: advantage normalization temperature eta")

    # --- P3O ---
    parser.add_argument("--initial_kappa", type=float, default=0.01, help="P3O: initial kappa (cost penalty)")
    parser.add_argument("--kappa_increase_factor", type=float, default=1.1, help="P3O: multiplicative factor for kappa when constraint violated")
    parser.add_argument("--kappa_max", type=float, default=50.0, help="P3O: maximum kappa value")

    # --- CRPO ---
    parser.add_argument("--crpo_eta", type=float, default=0.0,
                        help="CRPO: tolerance added to safety_bound when deciding whether to optimize "
                             "reward or cost for a training step")

    # --- SAC-Lag (off-policy) ---
    parser.add_argument("--lagrangian_lr", type=float, default=0.01,
                        help="SAC-Lag: Lagrange multiplier LR (per-episode scale, comparable to PPO-Lag lagrangian_coef_rate; normalized internally by episode_length)")
    parser.add_argument("--initial_lambda", type=float, default=0.0, help="SAC-Lag: initial Lagrange multiplier value")
    parser.add_argument("--lambda_max", type=float, default=2.0, help="SAC-Lag: upper bound on Lagrange multiplier (prevents runaway)")
    parser.add_argument("--tau", type=float, default=0.005, help="SAC-Lag: soft target network update coefficient")
    parser.add_argument("--min_replay_size", type=int, default=0, help="SAC-Lag: minimum replay buffer size before training starts")
    parser.add_argument("--max_replay_size", type=float, default=None, help="SAC-Lag: maximum replay buffer size before training starts")
    parser.add_argument("--grad_updates_per_step", type=int, default=1, help="SAC-Lag: gradient updates per environment step")

    # --- WandB ---
    parser.add_argument("--use_wandb", type=bool, default=True, help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="crax", help="W&B project")
    parser.add_argument("--wandb_group", type=str, default=None, help="W&B group")
    parser.add_argument("--wandb_tags", type=str, nargs='+', help="JSON list or path of tags")

    # --- Vision (Pixel Observation) ---
    parser.add_argument("--vision", action="store_true", help="Use egocentric pixel observations (GPU via MJWarp)")
    parser.add_argument(
        "--vision_camera", type=str, default=None,
        help="Name of the MuJoCo camera to render pixel observations from "
             "(must exist in the XML). Default: 'vision' (egocentric), except "
             "'track' (external side/chase view) for humanoid/ant/cheetah/"
             "walker2d/spider morphologies, and 'fixedfar' for reacher.",
    )
    parser.add_argument("--vision_height", type=int, default=64, help="Render height in pixels")
    parser.add_argument("--vision_width", type=int, default=64, help="Render width in pixels")
    parser.add_argument(
        "--vision_obs_mode", type=str, choices=["pixels", "pixels+state"], default="pixels",
        help="'pixels' (pixels only) or 'pixels+state' (pixels + state vector)",
    )
    parser.add_argument("--vision_frame_stack", type=int, default=3, help="Number of frames to stack channel-wise (3 is standard for RGB dm_control-style pixel RL, e.g. DrQ/RAD)")

    # --- Video Recording ---
    parser.add_argument("--cameras", type=str, nargs="+", default=["fixedfar", "vision"], help="Camera names/ids")
    parser.add_argument("--video_width", type=int, default=320, help="Output video width")
    parser.add_argument("--video_height", type=int, default=240, help="Output video height")
    parser.add_argument("--video_length", type=int, default=None, help="Number of frames in the video")
    parser.add_argument("--video_fps", type=int, default=100, help="Output video FPS")
    parser.add_argument("--video_frame_stride", type=int, default=1, help="Output video frame stride")
    parser.add_argument(
        "--num_video_episodes", type=int, default=5, help="Number of episodes to record per evaluation",
    )

    return parser


def build_base_parser(description: str | None = None) -> argparse.ArgumentParser:
    """Builds and returns a parser with all shared training arguments added."""
    parser = argparse.ArgumentParser(description=description)
    return add_shared_training_args(parser)
