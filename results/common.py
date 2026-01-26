"""Common utilities for result processing scripts."""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define a consistent color palette for baselines across all plots
# Using matplotlib's tab10 colormap
_tab10_colors = cm.get_cmap("tab10").colors
BASELINES_COLORS: Dict[str, str] = {
    "ppo": _tab10_colors[0],
    "ppo_cost": _tab10_colors[1],
    "ppo_lag": _tab10_colors[2],
    "ppo_pid": _tab10_colors[3],
    "ppo_saute": _tab10_colors[4],
    "p3o": _tab10_colors[5],
    "focops": _tab10_colors[6],
}

# Default mapping matching most scripts in this repo
DEFAULT_METRIC_COLS: Dict[str, str] = {
    "reward": "episodic/sum_reward",
    "cost": "episodic/cost",
}

# Defines the primary reward metric for specific environments.
# If an environment is not listed here, it defaults to 'episodic/sum_reward'.
REWARD_METRIC_MAP = {
    'safe_walker': 'episodic/reward_forward',
    'safe_velocity': 'episodic/forward_reward',
    'safe_spider': 'episodic/reward_forward',
}
DEFAULT_REWARD_METRIC = 'episodic/sum_reward'


def get_metrics_for_env(env_name: str, metrics: list[str]) -> list[str]:
    """
    Replaces the default reward metric with the correct one for the given environment.
    """
    metrics = list(metrics)  # Make a copy
    if DEFAULT_REWARD_METRIC in metrics:
        reward_metric = REWARD_METRIC_MAP.get(env_name, DEFAULT_REWARD_METRIC)
        # Find index of default reward metric and replace it
        idx = metrics.index(DEFAULT_REWARD_METRIC)
        metrics[idx] = reward_metric
    return metrics


def set_mpl_style() -> None:
    """Apply a consistent plotting style for all result figures."""
    # Use the same seaborn style used across scripts
    plt.style.use("seaborn-v0_8-paper")
    # Improve visibility: larger fonts and thicker lines globally
    plt.rcParams.update({
        "figure.dpi": 300,
        # Base/font sizes
        "font.size": 12.5,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12.5,
        "figure.titlesize": 16,
        # Line and axes widths
        "lines.linewidth": 2.2,
        "axes.linewidth": 1.2,
        # Grid
        "grid.linewidth": 0.8,
        "grid.alpha": 0.4,
    })


def nice_grid(n: int, max_cols: int = 3) -> Tuple[int, int]:
    """Pick a visually pleasing grid (rows, cols) for n panels.

    Tries to keep rows small and aspect roughly square while respecting max_cols.
    """
    if n <= 0:
        return 1, 1
    best = None
    for cols in range(1, max_cols + 1):
        rows = int(np.ceil(n / cols))
        score = (rows, abs(rows - cols), cols)  # prioritize fewer rows, then squareness
        if best is None or score < best[0]:
            best = (score, rows, cols)
    return best[1], best[2]


def get_series(
        df: pd.DataFrame,
        algo: str,
        metric: str,
        metric_cols: Optional[Dict[str, str]] = None,
        env_name: Optional[str] = None,
) -> Optional[pd.Series]:
    """Return a numeric pandas Series for the requested metric, with corrections.

    Special handling for 'ppo_cost': during training the logged reward already has
    the cost subtracted (reward_logged = true_reward - cost). For plotting we want
    the true original reward, so we add the cost back when metric == 'reward'.

    If required columns are missing, returns None.
    """
    cols = metric_cols or DEFAULT_METRIC_COLS
    
    default_reward_col = cols.get("reward")
    cost_col = cols.get("cost")
    
    reward_col_name = default_reward_col
    if env_name:
        reward_col_name = REWARD_METRIC_MAP.get(env_name, default_reward_col)

    if metric == "reward":
        if algo == "ppo_cost" and env_name not in ["safe_point_goal", "safe_reacher", "safe_spider", "safe_block_push"]:
            # Need both reward and cost columns to reconstruct original reward
            if reward_col_name not in df.columns or cost_col not in df.columns:
                 if default_reward_col not in df.columns or cost_col not in df.columns:
                    return None
                 else:
                    reward_col_name = default_reward_col
            ser = df[reward_col_name].astype(np.float32) + df[cost_col].astype(np.float32)
            return ser
        else:
            col = reward_col_name
    elif metric == "cost":
        col = cost_col
    else:
        # Unknown metric
        col = cols.get(metric)

    if col is None or col not in df.columns:
        # Fallback to default reward column if the specific one is not found
        if metric == "reward" and default_reward_col in df.columns:
            col = default_reward_col
        else:
            return None

    return df[col].astype(np.float32)


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Smooth data with a simple moving average."""
    if window_size <= 1:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def align_and_stack(dfs: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
    """Align multiple runs by min length after sorting by _step.

    Returns (steps, values) with shape [runs, T]. If dfs is empty, returns empty arrays.
    """
    if not dfs:
        return np.array([]), np.array([[]])
    lens = [len(d) for d in dfs]
    T = int(np.min(lens))
    if T <= 0:
        return np.array([]), np.array([[]])
    trimmed = [d.sort_values("_step", kind="mergesort").iloc[:T] for d in dfs]
    steps = trimmed[0]["_step"].to_numpy(copy=True)
    vals = np.stack([d["value"].to_numpy(copy=True) for d in trimmed], axis=0)  # [R, T]
    return steps, vals
