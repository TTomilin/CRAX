#!/usr/bin/env python3
"""Training-performance benchmark: OmniSafe PPOLag on SafetyAntVelocity-v1.

Companion to CRAX's `train_env.py --alg ppo_lag --env_name safe_velocity_ant`
run. Produces one per-seed CSV of (step, reward, cost) so
`results/plotting_results/omnisafe_vs_crax_ant_velocity.py` can overlay both
platforms' training curves (reward-vs-steps, cost-vs-steps), matching the
reviewer-requested Brax-paper-Fig-4-style comparison.

cost_limit (25.0) and episode length (1000) are OmniSafe/Safety-Gymnasium
defaults and already match CRAX's `--safety_bound 25.0` default and
`safe_velocity_ant`'s fixed 1000-step episode, so no config surgery needed
for a fair comparison beyond total_steps and seeds.

Usage (in the `omnisafe` conda env, NOT `crax`):
    conda run -n omnisafe python scripts/benchmark_omnisafe_ant_velocity.py \\
        --seeds 1 2 3 --num-timesteps 1000000

Runs sequentially, one seed at a time (avoid this if submitting all seeds as
separate sbatch jobs instead -- see scripts/run_omnisafe_ant_velocity_benchmark.sh).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Fix broken TF namespace package that causes tensorboard import to fail.
# Same patch as benchmark_omnisafe_training.py -- the omnisafe env's stray
# ~/.local tensorflow namespace package satisfies `import tensorflow` but has
# no attributes, which breaks torch's tensorboard writer import.
# ---------------------------------------------------------------------------
import os
import sys
import types

import tensorflow as _tf_ns  # loads the (empty) namespace package

_tf_io = types.ModuleType("tensorflow.io")
_tf_gfile = types.SimpleNamespace(join=os.path.join, get_filesystem=None)
_tf_io.gfile = _tf_gfile
_tf_ns.io = _tf_io
sys.modules["tensorflow.io"] = _tf_io
# ---------------------------------------------------------------------------

import argparse
import glob
import shutil
import time
from pathlib import Path
from typing import List

import pandas as pd

ENV_NAME = "SafetyAntVelocity-v1"


def record_and_upload_video(
        run_log_dir: Path,
        num_epochs: int,
        num_episodes: int,
        width: int,
        height: int,
) -> None:
    """Roll out the just-trained policy and log the replay video to the active wandb run.

    Reuses the wandb run omnisafe's Logger already opened for this seed
    (logger_cfgs.use_wandb=True in run_seed) instead of starting a new one.
    """
    import wandb
    from omnisafe import Evaluator

    model_name = f"epoch-{num_epochs}.pt"
    evaluator = Evaluator()
    evaluator.load_saved(
        save_dir=str(run_log_dir),
        model_name=model_name,
        render_mode="rgb_array",
        width=width,
        height=height,
    )

    video_dir = run_log_dir / "video"
    evaluator.render(num_episodes=num_episodes, save_replay_path=str(video_dir))

    videos = sorted(glob.glob(str(video_dir / "**" / "*.mp4"), recursive=True))
    if not videos:
        print(f"WARNING: no video found under {video_dir} after render()")
        return

    if wandb.run is not None:
        wandb.log({"eval/video": wandb.Video(videos[0], fps=evaluator.fps, format="mp4")})
        print(f"Uploaded video to wandb: {videos[0]}")
    else:
        print(f"WARNING: no active wandb run, video saved locally only: {videos[0]}")


def run_seed(
        seed: int,
        num_timesteps: int,
        num_envs: int,
        steps_per_env: int,
        cost_limit: float,
        device: str,
        output_dir: Path,
        record_video: bool = True,
        video_episodes: int = 1,
        video_width: int = 256,
        video_height: int = 256,
) -> Path:
    """Train one PPOLag run on SafetyAntVelocity-v1 and dump a (step, reward, cost) CSV."""
    import omnisafe  # deferred so the TF patch above is already applied
    import wandb

    steps_per_epoch = steps_per_env * num_envs
    num_epochs = max(num_timesteps // steps_per_epoch, 1)
    total_steps = steps_per_epoch * num_epochs

    log_dir = output_dir / f"_runs_seed{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}\nSeed {seed}: {ENV_NAME}  total_steps={total_steps:,}  "
          f"steps_per_epoch={steps_per_epoch:,}  num_epochs={num_epochs}\n{'=' * 60}")

    t0 = time.time()

    agent = omnisafe.Agent(
        "PPOLag",
        ENV_NAME,
        custom_cfgs={
            "seed": seed,
            "train_cfgs": {
                "vector_env_nums": num_envs,
                "total_steps": total_steps,
                "device": device,
            },
            "algo_cfgs": {
                "steps_per_epoch": steps_per_epoch,
            },
            "lagrange_cfgs": {
                "cost_limit": cost_limit,
            },
            "logger_cfgs": {
                "use_tensorboard": False,
                "use_wandb": True,
                "log_dir": str(log_dir),
            },
        },
    )
    agent.learn()

    wall_time = time.time() - t0
    print(f"Seed {seed} done in {wall_time:.0f}s")

    progress_files = list(log_dir.rglob("progress.csv"))
    if not progress_files:
        raise RuntimeError(f"No progress.csv found under {log_dir}")

    run_log_dir = progress_files[0].parent

    df = pd.read_csv(progress_files[0])
    out = pd.DataFrame({
        "step": df["TotalEnvSteps"],
        "reward": df["Metrics/EpRet"],
        "cost": df["Metrics/EpCost"],
        "seed": seed,
    })

    out_path = output_dir / f"seed_{seed}.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    if record_video:
        try:
            record_and_upload_video(
                run_log_dir=run_log_dir,
                num_epochs=num_epochs,
                num_episodes=video_episodes,
                width=video_width,
                height=video_height,
            )
        except Exception as e:
            print(f"WARNING: video recording failed for seed={seed}: {e}")
            import traceback
            traceback.print_exc()

    if wandb.run is not None:
        wandb.finish()

    shutil.rmtree(log_dir, ignore_errors=True)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OmniSafe PPOLag training-performance benchmark on SafetyAntVelocity-v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--num-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=4, help="Parallel envs")
    parser.add_argument("--steps-per-env", type=int, default=1000, help="Per-env steps per epoch")
    parser.add_argument("--cost-limit", type=float, default=25.0, help="Lagrangian cost budget")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None, help="Output dir (default: omnisafe_ant_velocity_benchmark_<timestamp>)")
    parser.add_argument("--skip-video", action="store_true", help="Skip end-of-training video recording/upload")
    parser.add_argument("--video-episodes", type=int, default=1, help="Number of eval episodes to record")
    parser.add_argument("--video-width", type=int, default=256)
    parser.add_argument("--video-height", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output) if args.output else Path(f"omnisafe_ant_velocity_benchmark_{time.strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_paths: List[Path] = []
    for seed in args.seeds:
        try:
            seed_paths.append(
                run_seed(
                    seed=seed,
                    num_timesteps=args.num_timesteps,
                    num_envs=args.num_envs,
                    steps_per_env=args.steps_per_env,
                    cost_limit=args.cost_limit,
                    device=args.device,
                    output_dir=output_dir,
                    record_video=not args.skip_video,
                    video_episodes=args.video_episodes,
                    video_width=args.video_width,
                    video_height=args.video_height,
                )
            )
        except Exception as e:
            print(f"FAILED for seed={seed}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone. {len(seed_paths)}/{len(args.seeds)} seeds succeeded. Results in: {output_dir}/")


if __name__ == "__main__":
    main()
