"""Shared command-line arguments for the result download and plotting scripts.

The scripts under `results/download/` and `results/plotting_results/` all select
runs the same way — by environment, algorithm, seed and difficulty level — and
then differ only in what they do with them. This module defines those arguments
once as reusable parent parsers, so a script only spells out what is genuinely
its own.

A script builds its parser from one of the two factory functions and overrides
whichever defaults it needs::

    parser = cli.download_parser("Download the safety sweep.", seeds=[1, 2, 3], levels=[1])
    parser.add_argument("--bounds", type=int, nargs="+", default=[15, 25, 35])
    args = parser.parse_args()

Keyword arguments to the factories are defaults for arguments the shared
parsers already define; passing an unknown name is an error rather than a
silently ignored typo.
"""

from __future__ import annotations

import argparse
from typing import Any, Iterable, List, Optional

from results.common import DEFAULT_METRIC_COLS

# Selection defaults. Individual scripts override these where their own data
# only covers part of the benchmark.
DEFAULT_SEEDS: List[int] = list(range(1, 11))
DEFAULT_ENVS: List[str] = [
    "safe_reacher", "safe_goal_point", "safe_push_point", "safe_lift_spider",
    "safe_circle_point", "safe_height_humanoid", "safe_pathway_walker2d",
    "safe_velocity_humanoid",
]
DEFAULT_ALGOS: List[str] = [
    "ppo", "ppo_cost", "ppo_lag", "ppo_saute", "ppo_pid", "p3o", "focops",
    "sac", "sac_lag", "sac_pid",
]
# Smaller sets the plotting scripts tend to use.
DEFAULT_PLOT_ENVS: List[str] = ["safe_goal_point", "safe_reacher", "safe_push_point"]
DEFAULT_SAFE_ALGOS: List[str] = ["ppo_lag", "ppo_pid", "p3o", "focops"]
DEFAULT_LEVELS: List[int] = [1, 2, 3]

# `--metrics` means two different things: wandb metric keys when downloading,
# and the short column names of DEFAULT_METRIC_COLS when plotting.
DEFAULT_DOWNLOAD_METRICS: List[str] = ["episodic/sum_reward", "episodic/cost"]
DEFAULT_PLOT_METRICS: List[str] = ["reward", "cost"]
PLOT_METRIC_CHOICES: List[str] = list(DEFAULT_METRIC_COLS.keys())

DEFAULT_SAFETY_THRESHOLD: float = 25.0
DEFAULT_FIGURE_DIR: str = "figures"
DEFAULT_DATA_DIR: str = "data"


def apply_defaults(parser: argparse.ArgumentParser, **defaults: Any) -> argparse.ArgumentParser:
    """Overrides defaults of arguments the parser already defines.

    Raises ValueError for a name the parser does not know, so that a typo in a
    caller's override surfaces immediately instead of being ignored.
    """
    if not defaults:
        return parser
    known = {action.dest for action in parser._actions}
    unknown = sorted(set(defaults) - known)
    if unknown:
        raise ValueError(
            f"unknown argument(s) {unknown}; this parser defines {sorted(known)}"
        )
    parser.set_defaults(**defaults)
    return parser


def selection_args(*, level_arg: Optional[str] = "list",
                   omit: Iterable[str] = ()) -> argparse.ArgumentParser:
    """Run-selection arguments shared by every script.

    Args:
        level_arg: "list" for a `--levels` list, "single" for a scalar
            `--level`, or None to leave difficulty selection to the caller.
        omit: names of arguments not to define, for scripts that need their own
            version (e.g. an `--algos` restricted to a fixed set of choices) or
            that do not select on that dimension at all.
    """
    omit = set(omit)
    p = argparse.ArgumentParser(add_help=False)
    if "envs" not in omit:
        p.add_argument("--envs", type=str, nargs="+", default=DEFAULT_ENVS,
                       help="Environments to include")
    if "algos" not in omit:
        p.add_argument("--algos", type=str, nargs="+", default=DEFAULT_ALGOS,
                       help="Algorithms to include")
    if "seeds" not in omit:
        p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                       help="Seeds to include")
    if level_arg == "list":
        p.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS,
                       help="Difficulty levels to include")
    elif level_arg == "single":
        p.add_argument("--level", type=int, default=1, help="Difficulty level")
    elif level_arg is not None:
        raise ValueError(
            f"level_arg must be 'list', 'single' or None, got {level_arg!r}")
    return p


def wandb_args(*, project_default: Optional[str] = None,
               omit: Iterable[str] = ()) -> argparse.ArgumentParser:
    """WandB query arguments shared by the download scripts.

    `--project` is required unless `project_default` gives it a value.
    """
    omit = set(omit)
    p = argparse.ArgumentParser(add_help=False)
    if project_default is None:
        p.add_argument("--project", type=str, required=True,
                       help="Name of the WandB project")
    else:
        p.add_argument("--project", type=str, default=project_default,
                       help="Name of the WandB project")
    p.add_argument("--wandb_tags", type=str, nargs="+", default=[],
                   help="WandB tags to filter runs")
    if "include_runs" not in omit:
        p.add_argument("--include_runs", type=str, nargs="+", default=[],
                       help="Substrings of run names to include; runs not matching any are skipped")
    p.add_argument("--max_age_days", type=float, default=None,
                   help="Only download runs created at most this many days ago "
                        "(default: no age limit)")
    p.add_argument("--overwrite", default=False, action="store_true",
                   help="Overwrite existing files")
    return p


def figure_args(*, out_name: str = "figure",
                omit: Iterable[str] = ()) -> argparse.ArgumentParser:
    """Figure output and layout arguments shared by the plotting scripts.

    `omit` accepts individual argument names, or "layout" as a shorthand for
    every grid/panel-size argument, for scripts that only write one figure.
    """
    omit = set(omit)
    if "layout" in omit:
        omit |= {"grid", "max_cols", "panel_w", "panel_h", "smoothing_window"}
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--output_fig_dir", type=str, default=DEFAULT_FIGURE_DIR,
                   help="Directory to write figures to")
    if "out_name" not in omit:
        p.add_argument("--out_name", type=str, default=out_name,
                       help="Base name of the output figure")
    if "grid" not in omit:
        p.add_argument("--grid", action="store_true", help="Draw grid lines")
    if "max_cols" not in omit:
        p.add_argument("--max_cols", type=int, default=2, help="Max env columns in grid")
    if "panel_w" not in omit:
        p.add_argument("--panel_w", type=float, default=3.1, help="Width per subplot column")
    if "panel_h" not in omit:
        p.add_argument("--panel_h", type=float, default=2.3, help="Height per subplot row")
    if "smoothing_window" not in omit:
        p.add_argument("--smoothing_window", type=int, default=1,
                       help="Moving average window size for smoothing")
    return p


def stats_args(*, omit: Iterable[str] = ()) -> argparse.ArgumentParser:
    """Aggregation arguments shared by the scripts that report final numbers."""
    omit = set(omit)
    p = argparse.ArgumentParser(add_help=False)
    if "ci_method" not in omit:
        p.add_argument("--ci_method", type=str, default="t", choices=["normal", "t"],
                       help="Confidence interval method")
    if "last_frac" not in omit:
        p.add_argument("--last_frac", type=float, default=0.1,
                       help="Fraction of the final training steps to average over")
    if "threshold" not in omit:
        p.add_argument("--threshold", type=float, default=DEFAULT_SAFETY_THRESHOLD,
                       help="Safety cost threshold")
    return p


def download_parser(
        description: str,
        *,
        level_arg: Optional[str] = "list",
        project_default: Optional[str] = None,
        omit: Iterable[str] = (),
        **defaults: Any,
) -> argparse.ArgumentParser:
    """Builds a parser for a `results/download/` script.

    Args:
        description: parser description.
        level_arg: see `selection_args`.
        project_default: default for `--project`; None makes it required.
        omit: names of shared arguments to leave undefined, e.g. ("algos",) for
            a script that defines its own restricted `--algos`. Also accepts
            "metrics".
        **defaults: default overrides for any argument defined here.
    """
    omit = set(omit)
    parents = [
        selection_args(level_arg=level_arg, omit=omit),
        wandb_args(project_default=project_default, omit=omit),
    ]
    p = argparse.ArgumentParser(description=description, parents=parents)
    p.add_argument("--output", type=str, default=DEFAULT_DATA_DIR,
                   help="Base output directory to store the data")
    if "metrics" not in omit:
        p.add_argument("--metrics", type=str, nargs="+", default=DEFAULT_DOWNLOAD_METRICS,
                       help="WandB metrics to download")
    return apply_defaults(p, **defaults)


def plot_parser(
        description: str,
        *,
        level_arg: Optional[str] = "single",
        stats: bool = False,
        out_name: str = "figure",
        omit: Iterable[str] = (),
        **defaults: Any,
) -> argparse.ArgumentParser:
    """Builds a parser for a `results/plotting_results/` script.

    Args:
        description: parser description.
        level_arg: see `selection_args`.
        stats: whether to include `--ci_method`, `--last_frac` and `--threshold`.
        out_name: default for `--out_name`.
        omit: names of shared arguments to leave undefined, e.g. ("algos",) or
            ("metrics",), for scripts that define their own version.
        **defaults: default overrides for any argument defined here.
    """
    omit = set(omit)
    parents = [selection_args(level_arg=level_arg, omit=omit),
               figure_args(out_name=out_name, omit=omit)]
    if stats:
        parents.append(stats_args(omit=omit))
    p = argparse.ArgumentParser(description=description, parents=parents)
    if "input" not in omit:
        p.add_argument("--input", type=str, default=DEFAULT_DATA_DIR,
                       help="Base directory holding the downloaded data")
    if "metrics" not in omit:
        p.add_argument("--metrics", type=str, nargs="+", default=DEFAULT_PLOT_METRICS,
                       choices=PLOT_METRIC_CHOICES, help="Metrics to plot")
    return apply_defaults(p, **defaults)
