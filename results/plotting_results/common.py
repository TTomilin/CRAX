from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Default mapping matching most scripts in this repo
DEFAULT_METRIC_COLS: Dict[str, str] = {
    "reward": "episodic/sum_reward",
    "cost": "episodic/cost",
}


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
) -> Optional[pd.Series]:
    """Return a numeric pandas Series for the requested metric, with corrections.

    Special handling for 'ppo_cost': during training the logged reward already has
    the cost subtracted (reward_logged = true_reward - cost). For plotting we want
    the true original reward, so we add the cost back when metric == 'reward'.

    If required columns are missing, returns None.
    """
    cols = metric_cols or DEFAULT_METRIC_COLS

    # Column containing the logged reward/cost as stored
    reward_col = cols.get("reward")
    cost_col = cols.get("cost")

    if metric == "reward":
        if algo == "ppo_cost":
            # Need both reward and cost columns to reconstruct original reward
            if reward_col not in df.columns or cost_col not in df.columns:
                return None
            ser = df[reward_col].astype(np.float32) + df[cost_col].astype(np.float32)
            return ser
        else:
            col = reward_col
    elif metric == "cost":
        col = cost_col
    else:
        # Unknown metric
        col = cols.get(metric)

    if col is None or col not in df.columns:
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
