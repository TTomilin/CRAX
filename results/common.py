"""Common utilities for result processing scripts."""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRANSLATIONS = {
    # Metrics
    "reward": "Reward",
    "cost": "Cost",
    # Training Modes
    "normal": "Normal",
    "curriculum": "Curriculum",
    "transfer": "Transfer",
    # Baselines
    "ppo": "PPO",
    "ppo_cost": "PPOCost",
    "ppo_lag": "PPOLag",
    "ppo_pid": "PPOPID",
    "ppo_saute": "PPOSaute",
    "p3o": "P3O",
    "focops": "FOCOPS",
    # Environments
    "safe_goal_point": "Safe Goal",
    "safe_reacher": "Safe Reacher",
    "safe_pathway_walker2d": "Safe Pathway",
    "safe_velocity_humanoid": "Safe Velocity",
    "safe_lift_spider": "Safe Spider",
    "safe_push_point": "Safe Push",
    "safe_circle_point": "Safe Circle",
    "safe_height_humanoid": "Safe Height",
}

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
    'safe_pathway_walker2d': 'episodic/reward_forward',
    'safe_velocity_humanoid': 'episodic/forward_reward',
    'safe_lift_spider': 'episodic/reward_forward',
    'safe_height_humanoid': 'episodic/forward_reward',
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
        if algo == "ppo_cost" and env_name not in ["safe_goal_point", "safe_reacher", "safe_spider", "safe_block_push"]:
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
    """Smooth data with a simple moving average that handles boundaries correctly."""
    if window_size <= 1:
        return data
    
    # Convolve data with ones to get the sum over the window
    data_sum = np.convolve(data, np.ones(window_size), 'same')
    
    # Create an array of ones with the same shape as data
    # Convolving this with ones gives the number of valid (non-padded) points in the window
    counts = np.convolve(np.ones_like(data), np.ones(window_size), 'same')
    
    # Divide sum by count to get the correct average
    return data_sum / counts


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
