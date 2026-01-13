import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    DEFAULT_METRIC_COLS as METRIC_COLS,
    get_series,
    set_mpl_style,
    nice_grid as _nice_grid,
)

# Pretty labels
TRANSLATIONS = {
    "reward": "Reward",
    "cost": "Cost",
    "ppo": "PPO",
    "ppo_cost": "PPOCost",
    "ppo_lag": "PPOLag",
    "ppo_pid": "PPOPID",
    "ppo_saute": "PPOSaute",
    "p3o": "P3O",
    "safe_point_goal": "Safe Point Goal",
    "safe_reacher": "Safe Reacher",
    "safe_walker": "Safe Walker",
}

# Optional safety thresholds per env for cost bars (None disables)
SAFETY_THRESHOLDS: Dict[str, float] = {
    "safe_point_goal": 25.0,
    "safe_reacher": 25.0,
    "safe_walker": 25.0,
}


def load_final_values(
        base: Path,
        envs: List[str],
        levels: List[int],
        algos: List[str],
        seeds: List[int],
        metrics: List[str],
        final_mode: str = "mean_last_k",  # "last" or "mean_last_k"
        last_k: int = 10,
) -> pd.DataFrame:
    """
    Returns long DataFrame with columns:
      env, level, algo, metric, seed, value
    """
    rows = []
    for env in envs:
        for level in levels:
            for algo in algos:
                for seed in seeds:
                    fp = base / env / f"level_{level}" / algo / f"seed_{seed}.parquet"
                    if not fp.exists():
                        print(f"File not found: {fp}")
                        continue

                    df = pd.read_parquet(fp, engine="pyarrow")

                    if "_step" not in df.columns:
                        continue

                    df = df.sort_values("_step", kind="mergesort")

                    for metric in metrics:
                        series = get_series(df, algo=algo, metric=metric, metric_cols=METRIC_COLS)
                        if series is None:
                            continue
                        series = series.dropna().astype(np.float32)
                        if len(series) == 0:
                            continue

                        if final_mode == "last":
                            v = float(series.iloc[-1])
                        elif final_mode == "mean_last_k":
                            k = max(1, min(last_k, len(series)))
                            v = float(series.iloc[-k:].mean())
                        else:
                            raise ValueError(f"Unknown final_mode: {final_mode}")

                        rows.append(
                            dict(env=env, level=level, algo=algo, metric=metric, seed=seed, value=v)
                        )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate seeds -> mean + 95% CI.
    Output columns: env, level, algo, metric, mean, ci, n
    """
    if df.empty:
        return df

    g = df.groupby(["env", "level", "algo", "metric"], as_index=False)
    out = g["value"].agg(["mean", "std", "count"]).reset_index()
    out.rename(columns={"count": "n"}, inplace=True)
    out["ci"] = 1.96 * out["std"].fillna(0.0) / np.sqrt(out["n"].clip(lower=1))
    out.drop(columns=["std"], inplace=True)
    return out


def plot_final_bars(stats: pd.DataFrame, args: argparse.Namespace) -> None:
    set_mpl_style()

    envs = args.envs
    metrics = args.metrics  # expects ["reward","cost"] by default
    levels = args.levels
    algos = args.algos

    n_env = len(envs)
    nrows_env, ncols_env = _nice_grid(n_env, max_cols=args.max_cols)

    # Each env occupies len(metrics) columns (reward+cost side-by-side)
    m = len(metrics)
    total_rows = nrows_env
    total_cols = ncols_env * m

    # panel_w is now "per metric subplot"; panel_h is "per env row"
    fig_w = args.panel_w * total_cols
    fig_h = args.panel_h * total_rows
    fig, axs = plt.subplots(total_rows, total_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.14, wspace=0.35, hspace=0.55)

    legend_handles: Dict[str, plt.Line2D] = {}

    # x positions: levels as groups, algos as bars within each group
    L = len(levels)
    A = len(algos)
    x = np.arange(L)

    group_width = 0.82
    bar_w = group_width / max(A, 1)
    offsets = (np.arange(A) - (A - 1) / 2.0) * bar_w

    def get_ax(env_i: int, metric_i: int):
        """Return axis for a given env and metric index (metric side-by-side)."""
        gr = env_i // ncols_env
        gc = env_i % ncols_env
        r = gr
        c = gc * m + metric_i
        return axs[r, c]

    # plot per env
    for env_i, env in enumerate(envs):
        env_title = TRANSLATIONS.get(env, env)

        # plot each metric into its side-by-side axis
        for metric_i, metric in enumerate(metrics):
            ax = get_ax(env_i, metric_i)
            ylab = TRANSLATIONS.get(metric, metric.capitalize())

            for a_i, algo in enumerate(algos):
                ys = []
                es = []
                for level in levels:
                    sub = stats[
                        (stats["env"] == env)
                        & (stats["metric"] == metric)
                        & (stats["algo"] == algo)
                        & (stats["level"] == level)
                        ]
                    if len(sub) == 0:
                        ys.append(np.nan)
                        es.append(0.0)
                    else:
                        ys.append(float(sub["mean"].iloc[0]))
                        es.append(float(sub["ci"].iloc[0]))

                bars = ax.bar(x + offsets[a_i], ys, width=bar_w, yerr=es, capsize=2, label=algo)

                if algo not in legend_handles:
                    legend_handles[algo] = bars[0]

            ax.set_xticks(x)
            ax.set_xticklabels([str(lv) for lv in levels])
            ax.set_xlabel("Level")
            ax.set_ylabel(ylab)

            # threshold only on cost axis
            if metric == "cost" and not args.no_threshold:
                thr = SAFETY_THRESHOLDS.get(env, None)
                if thr is not None:
                    thr_line = ax.axhline(thr, linestyle="--", color="red")
                    if "Threshold" not in legend_handles:
                        legend_handles["Threshold"] = thr_line

            if args.grid:
                ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

        # Center env title above the reward+cost pair (spans m subplots)
        left_bbox = get_ax(env_i, 0).get_position()
        right_bbox = get_ax(env_i, m - 1).get_position()
        mid_x = 0.5 * (left_bbox.x0 + right_bbox.x1)
        top_y = max(left_bbox.y1, right_bbox.y1) + 0.01
        fig.text(mid_x, top_y, env_title, ha="center", va="bottom", fontsize=13)

    # hide unused env slots (all metric columns for that env cell)
    for env_i in range(n_env, nrows_env * ncols_env):
        for metric_i in range(m):
            ax = get_ax(env_i, metric_i)
            ax.axis("off")

    # global legend at bottom
    if legend_handles:
        labels = list(legend_handles.keys())
        handles = [legend_handles[k] for k in labels]
        labels = [TRANSLATIONS.get(lbl, lbl) for lbl in labels]
        fig.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=min(len(labels), 6),
            fancybox=True,
            shadow=True,
        )

    out_dir = Path(args.output_fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}_final_bars_levels_{levels[0]}_{levels[-1]}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()
    print(f"Saved figure: {out_path}")


def build_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot final results as grouped bars across levels.")
    p.add_argument("--input", type=str, default="data",
                   help="Base directory with <env>/level_<k>/<algo>/seed_*.parquet")
    p.add_argument("--envs", type=str, nargs="+", default=["safe_point_goal", "safe_reacher", "safe_walker"])
    p.add_argument("--algos", type=str, nargs="+", default=["ppo", "ppo_cost", "ppo_lag", "ppo_pid", "ppo_saute", "p3o"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--metrics", type=str, nargs="+", default=["reward", "cost"], choices=list(METRIC_COLS.keys()))

    p.add_argument("--final_mode", type=str, default="mean_last_k", choices=["last", "mean_last_k"],
                   help="How to define 'final' value per seed.")
    p.add_argument("--last_k", type=int, default=10, help="Used when final_mode=mean_last_k.")

    p.add_argument("--no_threshold", action="store_true", help="Hide safety threshold lines (cost only).")
    p.add_argument("--grid", action="store_true")

    p.add_argument("--max_cols", type=int, default=4, help="Max env columns in grid.")
    p.add_argument("--panel_w", type=float, default=3.1, help="Width per env column.")
    p.add_argument("--panel_h", type=float, default=2.3, help="Height per metric row per env row.")

    p.add_argument("--output_fig_dir", type=str, default="figures")
    p.add_argument("--out_name", type=str, default="baselines")
    return p


def main(args: argparse.Namespace) -> None:
    base = Path(__file__).parent.parent.resolve() / args.input
    df = load_final_values(
        base=base,
        envs=args.envs,
        levels=args.levels,
        algos=args.algos,
        seeds=args.seeds,
        metrics=args.metrics,
        final_mode=args.final_mode,
        last_k=args.last_k,
    )
    if df.empty:
        raise SystemExit("No data loaded. Check paths/envs/levels/algos/seeds.")
    stats = summarize(df)
    plot_final_bars(stats, args)


if __name__ == "__main__":
    args = build_args().parse_args()
    main(args)
