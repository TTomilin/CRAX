"""Training-performance comparison: CRAX vs OmniSafe+Safety-Gymnasium.

Reviewer-requested comparison (cf. Brax paper Fig. 4): reward-vs-steps and
cost-vs-steps training curves for PPOLag on the ant-velocity task, run on
both platforms, mean +/- 95% CI over seeds.

Inputs:
- CRAX side: parquet files at
  results/data/safe_velocity_ant/level_1/ppo_lag/seed_{seed}.parquet
  (pull with `python -m results.download --envs safe_velocity_ant --algos ppo_lag
  --wandb_tags OMNISAFE_COMPARISON
  --metrics episodic/reward_unscaled episodic/cost` first).
- OmniSafe side: CSV files at results/data/omnisafe_ant_velocity/seed_{seed}.csv
  (columns: step, reward, cost, seed), pulled from WandB with
  `python -m results.download_omnisafe_ant_velocity --project omnisafe --seeds 1 2 3`.
  Both --crax-input/--omnisafe-input default to these downloaders' --output paths,
  so no need to pass them explicitly unless you used a custom --output.

Reward parity: CRAX's `episodic/sum_reward` already nets ctrl_cost (same
formula both platforms use: forward_reward + healthy_reward - ctrl_cost), but
it also carries SafeVelocityBase's `reward_scaler` (default 0.01), which
OmniSafe/Safety-Gymnasium's raw `Metrics/EpRet` does not apply. We use
`episodic/reward_unscaled` (added in brax/envs/velocity_constraints.py) --
the same net reward logged *before* that scaling -- so both curves are on
directly comparable scales without a manual /100 correction.

Episode-length parity: `train_env.py --episode_length` now correctly overrides
the env's ctor default (previously silently clobbered), so `safe_velocity_ant`
runs at the intended 1000-step episodes, matching OmniSafe's fixed 1000-step
`SafetyAntVelocity-v1` episodes -- no manual reward/cost scaling needed here.

Initial-reward gap (why OmniSafe starts more negative): both platforms reach
the same ~3000 final reward with cost settling near the 25 bound, but CRAX
gets there in fewer steps and starts less negative. The gap traces to how
each platform's *untrained* policy turns network noise into actions, which
feeds directly into `ctrl_cost = ctrl_cost_weight * sum(action**2)` (same
formula both sides use -- see gymnasium's `AntEnv.control_cost`, inherited by
`safety_gymnasium`'s `SafetyAntVelocityEnv`, and CRAX's own `ant.py`):
  - CRAX's PPO actor (`NormalTanhDistribution` in
    brax/training/distribution.py) samples a Gaussian in *pre-tanh* space and
    squashes it through `tanh`. The action passed to `env.step` is therefore
    always in (-1, 1) by construction, regardless of how large the pre-tanh
    noise is -- capping ctrl_cost at `0.5 * 8 = 4` for Ant's 8 actuators from
    the very first step.
  - OmniSafe's PPOLag actor (`GaussianLearningActor` in
    omnisafe/models/actor/gaussian_learning_actor.py) samples a *raw,
    unsquashed* `Normal(mean, exp(log_std))`, with `log_std` initialized to
    zero -> std=1 directly in action space, no tanh. `control_cost()` is
    computed on this raw action *before* MuJoCo's actuator clipping (which
    only happens later, inside `do_simulation`) -- so early samples with
    |action| > 1 aren't capped, and ctrl_cost is both larger on average and
    far more variable right at initialization.
As training reduces `log_std` and reshapes the policy mean, this gap closes,
consistent with both curves converging to the same final reward/cost.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results.common import DEFAULT_METRIC_COLS as METRIC_COLS, get_series, set_mpl_style
from results.plotting_results.seed_variance import ci95, format_table

CRAX_ENV = "safe_velocity_ant"
CRAX_ALGO = "ppo_lag"


def load_crax_curves(base: Path, level: int, seeds: List[int]) -> List[pd.DataFrame]:
    """Load per-seed (step, reward, cost) frames from CRAX's wandb-downloaded parquets.

    `base` is results/download.py's --output base dir (default "results/data"); this
    rebuilds the same env/level/algo tree it stores into, so callers don't have to
    spell out the full leaf path (matches results/plotting_results/stage1_sweep.py etc).
    """
    curves = []
    for seed in seeds:
        fp = base / CRAX_ENV / f"level_{level}" / CRAX_ALGO / f"seed_{seed}.parquet"
        if not fp.exists():
            print(f"Skipping (missing): {fp}")
            continue
        df = pd.read_parquet(fp, engine="pyarrow")
        if "_step" not in df.columns:
            continue
        df = df.sort_values("_step", kind="mergesort")
        # Not get_series(metric="reward", ...): that resolves through
        # REWARD_METRIC_MAP to episodic/forward_reward for safe_velocity_ant
        # (forward-velocity only, no ctrl_cost) -- wrong for an OmniSafe-
        # comparable net-reward plot. Read the unscaled net reward directly.
        if "episodic/reward_unscaled" not in df.columns:
            print(f"Skipping (missing episodic/reward_unscaled -- re-download with "
                  f"--metrics episodic/reward_unscaled episodic/cost): {fp}")
            continue
        reward = df["episodic/reward_unscaled"].astype(np.float64)
        cost = get_series(df, algo=CRAX_ALGO, metric="cost", metric_cols=METRIC_COLS, env_name=CRAX_ENV)
        if reward is None or cost is None:
            print(f"Skipping (missing reward/cost columns): {fp}")
            continue
        curves.append(pd.DataFrame({
            "step": df["_step"].to_numpy(dtype=np.float64),
            "reward": reward.to_numpy(dtype=np.float64),
            "cost": cost.to_numpy(dtype=np.float64),
        }))
    return curves


def load_omnisafe_curves(base: Path, seeds: List[int]) -> List[pd.DataFrame]:
    """Load per-seed (step, reward, cost) frames from benchmark_omnisafe_ant_velocity.py output."""
    curves = []
    for seed in seeds:
        fp = base / f"seed_{seed}.csv"
        if not fp.exists():
            print(f"Skipping (missing): {fp}")
            continue
        df = pd.read_csv(fp).sort_values("step", kind="mergesort")
        curves.append(pd.DataFrame({
            "step": df["step"].to_numpy(dtype=np.float64),
            "reward": df["reward"].to_numpy(dtype=np.float64),
            "cost": df["cost"].to_numpy(dtype=np.float64),
        }))
    return curves


def interpolate_to_common_grid(curves: List[pd.DataFrame], num_points: int) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate each seed's (step, reward, cost) onto a shared step grid.

    Needed because CRAX (JAX-batched evals) and OmniSafe (per-epoch logging)
    report at different, non-aligned step counts. Returns (grid, reward[R,T], cost[R,T]).
    """
    if not curves:
        return np.array([]), np.array([[]]), np.array([[]])
    min_max_step = min(c["step"].max() for c in curves)
    max_min_step = max(c["step"].min() for c in curves)
    grid = np.linspace(max_min_step, min_max_step, num_points)
    rewards = np.stack([np.interp(grid, c["step"], c["reward"]) for c in curves], axis=0)
    costs = np.stack([np.interp(grid, c["step"], c["cost"]) for c in curves], axis=0)
    return grid, rewards, costs


def mean_ci_band(values: np.ndarray, ci_method: str) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column mean and 95% CI half-width across the seed axis (axis=0)."""
    means = np.empty(values.shape[1])
    cis = np.empty(values.shape[1])
    for t in range(values.shape[1]):
        m, _, ci = ci95(values[:, t], method=ci_method)
        means[t] = m
        cis[t] = ci
    return means, cis


def value_at_step(curves: List[pd.DataFrame], column: str, step: float) -> np.ndarray:
    """Per-seed value at `step`: interpolated, or clamped to the run's first/last logged
    point if `step` falls outside its logged range (checkpoints rarely land exactly on
    eval steps -- e.g. step=0 clamps to each run's first reading).
    """
    return np.array([np.interp(step, c["step"], c[column]) for c in curves], dtype=np.float64)


def build_periodic_table(
        crax_curves: List[pd.DataFrame], omnisafe_curves: List[pd.DataFrame],
        interval: float, num_rows: int, ci_method: str,
) -> Tuple[List[List], List[List]]:
    """Checkpoint-step comparison table: (step, CRAX reward/cost, OmniSafe reward/cost).

    Returns (display_rows, numeric_rows): display_rows have "mean +/- ci" strings for
    printing/format_table; numeric_rows have separate mean/ci float columns for CSV export.
    """

    def cell(curves: List[pd.DataFrame], column: str, step: float) -> Tuple[float, float]:
        vals = value_at_step(curves, column, step)
        if len(vals) < 2:
            return float("nan"), float("nan")
        mean, _, ci = ci95(vals, method=ci_method)
        return mean, ci

    def fmt(mean: float, ci: float) -> str:
        return "N/A" if np.isnan(mean) else f"{mean:.2f} +/- {ci:.2f}"

    display_rows, numeric_rows = [], []
    for i in range(num_rows + 1):
        step = i * interval
        cr_r_mean, cr_r_ci = cell(crax_curves, "reward", step)
        cr_c_mean, cr_c_ci = cell(crax_curves, "cost", step)
        os_r_mean, os_r_ci = cell(omnisafe_curves, "reward", step)
        os_c_mean, os_c_ci = cell(omnisafe_curves, "cost", step)
        display_rows.append([
            int(step),
            fmt(cr_r_mean, cr_r_ci), fmt(cr_c_mean, cr_c_ci),
            fmt(os_r_mean, os_r_ci), fmt(os_c_mean, os_c_ci),
        ])
        numeric_rows.append([
            step,
            cr_r_mean, cr_r_ci, cr_c_mean, cr_c_ci,
            os_r_mean, os_r_ci, os_c_mean, os_c_ci,
        ])
    return display_rows, numeric_rows


def save_periodic_table(numeric_rows: List[List], out_path: Path) -> None:
    """Save the periodic comparison table as CSV and a LaTeX tabular."""
    columns = [
        "step",
        "crax_reward_mean", "crax_reward_ci95", "crax_cost_mean", "crax_cost_ci95",
        "omnisafe_reward_mean", "omnisafe_reward_ci95", "omnisafe_cost_mean", "omnisafe_cost_ci95",
    ]
    df = pd.DataFrame(numeric_rows, columns=columns)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{CRAX vs. OmniSafe/Safety-Gymnasium: PPOLag on Ant Velocity}",
        r"\label{tab:omnisafe_vs_crax_ant_velocity}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Step & CRAX Reward & CRAX Cost & OmniSafe Reward & OmniSafe Cost \\",
        r"\midrule",
    ]
    for row in numeric_rows:
        step, cr_r_m, cr_r_c, cr_c_m, cr_c_c, os_r_m, os_r_c, os_c_m, os_c_c = row

        def fmt_tex(mean: float, ci: float) -> str:
            return "N/A" if np.isnan(mean) else f"{mean:.2f} $\\pm$ {ci:.2f}"

        lines.append(
            f"{int(step):,} & {fmt_tex(cr_r_m, cr_r_c)} & {fmt_tex(cr_c_m, cr_c_c)} & "
            f"{fmt_tex(os_r_m, os_r_c)} & {fmt_tex(os_c_m, os_c_c)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = out_path.with_suffix(".tex")
    tex_path.write_text("\n".join(lines))
    print(f"Saved: {tex_path}")


def plot_comparison(
        crax_grid: np.ndarray, crax_reward: np.ndarray, crax_cost: np.ndarray,
        omnisafe_grid: np.ndarray, omnisafe_reward: np.ndarray, omnisafe_cost: np.ndarray,
        threshold: float, ci_method: str, out_path: Path,
) -> None:
    set_mpl_style()
    fig, (ax_r, ax_c) = plt.subplots(1, 2, figsize=(11, 4.2))

    colors = {"CRAX (Ours)": "#2E86AB", "OmniSafe + Safety-Gymnasium": "#E94F37"}

    for label, grid, reward, cost in (
            ("CRAX (Ours)", crax_grid, crax_reward, crax_cost),
            ("OmniSafe + Safety-Gymnasium", omnisafe_grid, omnisafe_reward, omnisafe_cost),
    ):
        if grid.size == 0:
            continue
        color = colors[label]
        r_mean, r_ci = mean_ci_band(reward, ci_method)
        ax_r.plot(grid, r_mean, color=color, linewidth=2.2, label=label)
        ax_r.fill_between(grid, r_mean - r_ci, r_mean + r_ci, color=color, alpha=0.2)

        c_mean, c_ci = mean_ci_band(cost, ci_method)
        ax_c.plot(grid, c_mean, color=color, linewidth=2.2, label=label)
        ax_c.fill_between(grid, c_mean - c_ci, c_mean + c_ci, color=color, alpha=0.2)

    ax_r.set_xlabel("Environment Steps")
    ax_r.set_ylabel("Episodic Reward")
    ax_r.set_title("Reward vs. Training Steps")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc="lower right", fontsize=10)

    ax_c.axhline(threshold, color="black", linestyle="--", alpha=0.6, label=f"Safety bound ({threshold:g})")
    ax_c.set_xlabel("Environment Steps")
    ax_c.set_ylabel("Episodic Cost")
    ax_c.set_title("Cost vs. Training Steps")
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(loc="upper right", fontsize=10)

    fig.suptitle("PPO-Lag: CRAX vs. OmniSafe/Safety-Gymnasium (Ant Velocity)")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(out_path.with_suffix(f".{suffix}"))
        print(f"Saved: {out_path.with_suffix(f'.{suffix}')}")
    plt.show()
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    crax_base = Path(args.crax_input)
    omnisafe_base = Path(args.omnisafe_input)

    crax_curves = load_crax_curves(crax_base, args.level, args.seeds)
    omnisafe_curves = load_omnisafe_curves(omnisafe_base, args.seeds)

    if not crax_curves:
        print(f"WARNING: no CRAX curves loaded from {crax_base}")
    if not omnisafe_curves:
        print(f"WARNING: no OmniSafe curves loaded from {omnisafe_base}")

    crax_grid, crax_reward, crax_cost = interpolate_to_common_grid(crax_curves, args.num_points)
    omnisafe_grid, omnisafe_reward, omnisafe_cost = interpolate_to_common_grid(omnisafe_curves, args.num_points)

    out_path = Path(args.output_fig_dir) / args.out_name
    plot_comparison(
        crax_grid, crax_reward, crax_cost,
        omnisafe_grid, omnisafe_reward, omnisafe_cost,
        args.threshold, args.ci_method, out_path,
    )

    display_rows, numeric_rows = build_periodic_table(
        crax_curves, omnisafe_curves, args.table_interval, args.table_rows, args.ci_method,
    )
    headers = ["Step", "CRAX Reward", "CRAX Cost", "OmniSafe Reward", "OmniSafe Cost"]
    print(f"\n=== CRAX vs. OmniSafe (Ant Velocity), every {args.table_interval:,.0f} steps ===")
    print(format_table(display_rows, headers))
    save_periodic_table(numeric_rows, out_path)


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CRAX vs OmniSafe/Safety-Gymnasium training-performance comparison (ant velocity).")
    p.add_argument("--crax-input", type=str, default="results/data",
                   help="Base dir of CRAX's downloaded parquet data (results/download.py --output default).")
    p.add_argument("--omnisafe-input", type=str, default="results/data/omnisafe_ant_velocity",
                   help="Dir of OmniSafe's downloaded CSVs (results/download_omnisafe_ant_velocity.py --output default).")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   help="Seeds to include in the comparison.")
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--threshold", type=float, default=25.0,
                   help="Safety cost threshold (matches --safety_bound / cost_limit).")
    p.add_argument("--num-points", type=int, default=200, help="Number of points on the shared interpolation grid.")
    p.add_argument("--ci-method", type=str, default="t", choices=["normal", "t"])
    p.add_argument("--output-fig-dir", type=str, default="figures")
    p.add_argument("--out-name", type=str, default="omnisafe_vs_crax_ant_velocity")
    p.add_argument("--table-interval", type=float, default=200_000,
                   help="Step spacing between periodic comparison-table checkpoints.")
    p.add_argument("--table-rows", type=int, default=10,
                   help="Number of checkpoint rows in the periodic comparison table.")
    return p


if __name__ == "__main__":
    main(build_args().parse_args())
