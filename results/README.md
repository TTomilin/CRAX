# Results

Everything needed to turn CRAX training runs into the tables and figures of the
paper: **download** logged runs from Weights & Biases, and **plot** them.

```
results/
├── common.py            # shared style, colors, name translations, sweep specs, metric helpers
├── download/            # wandb -> local parquet/csv
├── data/                # downloaded run data (see layout below)
├── plotting_results/    # parquet/csv -> figures + LaTeX tables
└── figures/             # generated figures (PDF/PNG) and tables (TeX/CSV)
```

All scripts import `results.common`, so run them **as modules from the repo root**:

```bash
python -m results.download.main_results --project <wandb_project>
python -m results.plotting_results.alg_comp --envs safe_goal_point --level 1
```

## Data layout

Downloads mirror the run configuration in the directory tree, one parquet file
per seed:

| Experiment | Path |
|------------|------|
| Main results | `data/<env>/level_<level>/<algo>/seed_<n>.parquet` |
| Hyperparameter sweeps | `data/<env>/level_<level>/<algo>/<hparam>_<value>/seed_<n>.parquet` |
| Safety bounds | `data/<env>/level_<level>/<algo>/safety_bound_<value>/seed_<n>.parquet` |
| Curriculum | `data/curriculum/<env>/<algo>/seed_<n>.parquet` |
| Transfer | `data/transfer/<env>/level_<level>/<algo>/seed_<n>.parquet` |
| OmniSafe baseline | `data/omnisafe_ant_velocity/seed_<n>.csv` |
| Throughput benchmarks | `data/performance/<run_name>/benchmark_results.csv` |

Each parquet holds the wandb history for the requested metrics, keyed by
`_step`. Default metrics are `episodic/sum_reward` and `episodic/cost`; some
environments log their reward under a different key, resolved by
`REWARD_METRIC_MAP` / `get_metrics_for_env` in `common.py` (e.g.
`safe_velocity_ant` uses `episodic/forward_reward`).

> **Note on `--output`.** Download scripts resolve `--output` relative to
> `results/download/`, while plotting scripts resolve `--input` relative to
> `results/`. To download straight into the tree the plotters read, pass
> `--output ../data`.

## Downloading (`download/`)

| Script | Downloads |
|--------|-----------|
| `main_results.py` | Baseline comparison runs for all algos, envs, and levels. Also provides the shared `build_filters` / `store_data` used by the sweep scripts. |
| `safety_param_sweep.py` | Safe-RL-method hyperparameters, one at a time per method (`SAFETY_SWEEP_SPEC`: `lagrangian_coef_rate`, `pid_kp`/`pid_ki`, `gamma_budget`, `focops_lam`, `initial_kappa`, `crpo_eta`, ...). |
| `rl_param_sweep.py` | Core PPO/MJX hyperparameters (`RL_SWEEP_SPEC`: `learning_rate`, `entropy_cost`, `discounting`, `gae_lambda`, `clipping_epsilon`). |
| `transfer_curriculum.py` | Runs tagged `TRANSFER` (unsafe PPO pre-training + safe fine-tuning) and `CURRICULUM` (levels 1→2→3, `global_step` cumulative across stages). |
| `omnisafe.py` | OmniSafe + Safety-Gymnasium baseline runs, written as CSV with `(step, reward, cost, seed)`. |

Shared flags come from `cli.download_parser` (see [Shared CLI](#shared-cli)):
`--project` (required), `--envs`, `--algos`, `--levels`, `--seeds`, `--metrics`,
`--wandb_tags`, `--include_runs`, `--output`, `--overwrite`, and `--max_age_days`
(restrict to runs created within the last N days; off by default except in the
sweep scripts, which default to 1). Only `state: finished` runs are fetched, and
existing files are skipped unless `--overwrite` is passed.

```bash
# Baselines for one env, all levels, into results/data/
python -m results.download.main_results \
  --project my-crax-project --envs safe_goal_point --levels 1 2 3 --output ../data

# Safe-RL hyperparameter sweep
python -m results.download.safety_param_sweep --project my-crax-project --output ../data
```

## Plotting (`plotting_results/`)

| Script | Produces |
|--------|----------|
| `alg_comp.py` | Reward/cost training curves per environment, mean ± CI across seeds, with the safety threshold drawn in. |
| `alg_comp_bars.py` | Final-performance bar charts (per level and aggregated) plus LaTeX result tables (`--latex` prints them; `--output_latex base.tex` writes `base_summary.tex` and `base_appendix.tex`). |
| `sample_efficiency.py` | Reward AUC and cumulative constraint violation per (env, algo, seed) — how fast a method gets good, and how much it violates on the way. |
| `seed_variance.py` | CI width vs. number of seeds: small seed set vs. large, plus the full CI-vs-#seeds trend. |
| `seed_variance_curriculum_transfer.py` | The same seed-count analysis for the curriculum and transfer regimes. |
| `curriculum_transfer.py` | Normal vs. curriculum vs. transfer: stitched training curves across stages, and final level-3 comparison bars. |
| `safety_bounds.py` | Effect of the cost budget (`--bounds`, default 15/25/35) on reward and cost. |
| `safety_param_sweep.py` | Per-(algo, hyperparameter) final reward/cost across swept values, one table and figure each. |
| `rl_param_sweep.py` | The same, for the core PPO hyperparameters. |
| `crax_vs_sg.py` | CRAX vs. OmniSafe + Safety-Gymnasium training curves on ant-velocity (PPOLag), mean ± 95% CI. |
| `throughput_comparison.py` | Throughput (SPS) and scaling efficiency vs. Safety-Gymnasium, from `data/performance/`. |

Common flags come from `cli.plot_parser` (see [Shared CLI](#shared-cli)):
`--input` (data root, default `data`), `--envs`, `--algos`, `--seeds`, `--level`,
`--metrics {reward,cost}`, `--output_fig_dir` (default `figures`, relative to the
working directory), `--out_name`, plus the layout flags `--grid`, `--max_cols`,
`--panel_w`, `--panel_h`, `--smoothing_window` and, where a script reports final
numbers, `--ci_method`, `--last_frac` and `--threshold`. Figures inherit a single
style from `common.py` (`set_mpl_style`, `BASELINES_COLORS`, `TRANSLATIONS`), so
panels stay consistent across the paper.
`crax_vs_sg.py` is the exception: its `--crax_input` / `--omnisafe_input` are
resolved relative to the working directory, hence their `results/data` defaults.

```bash
# Training curves, level 1
python -m results.plotting_results.alg_comp \
  --envs safe_goal_point safe_reacher --algos ppo_lag ppo_pid focops --level 1 \
  --output_fig_dir results/figures

# Final bars + LaTeX table for all levels
python -m results.plotting_results.alg_comp_bars \
  --levels 1 2 3 --latex --output_latex results/figures/baselines.tex  # -> *_summary.tex, *_appendix.tex

# Throughput comparison (no flags; reads data/performance/)
python -m results.plotting_results.throughput_comparison
```

`throughput_comparison.py` consumes benchmark CSVs produced by
`scripts/benchmark_crax_simulation.py` and `scripts/benchmark_safety_gymnasium.py`.

## Shared CLI

`cli.py` defines the arguments that the download and plotting scripts have in
common — the run selection (`--envs`, `--algos`, `--seeds`, `--levels`/`--level`),
the WandB query flags, the figure output and layout flags, and the aggregation
flags — as reusable parent parsers. A script builds its parser from one of two
factories and spells out only what is its own:

```python
from results import cli

def build_args() -> argparse.ArgumentParser:
    p = cli.plot_parser(
        "Plot CRAX results for different safety bounds.",
        omit=("algos",),                       # this script takes a single --algo
        out_name="safety_bounds",
        envs=["safe_goal_point", "safe_reacher"],   # override a shared default
        panel_h=1.8,
    )
    p.add_argument("--algo", type=str, default="ppo_lag")
    p.add_argument("--bounds", type=int, nargs="+", default=[15, 25, 35])
    return p
```

Keyword arguments override the default of an argument the shared parsers already
define; an unknown name raises rather than being silently ignored. `omit` drops a
shared argument for a script that needs its own version of it (`"layout"` is
shorthand for all the grid/panel-size flags). Defaults live in one place —
`cli.DEFAULT_ENVS`, `DEFAULT_ALGOS`, `DEFAULT_SEEDS`, `DEFAULT_LEVELS`,
`DEFAULT_SAFE_ALGOS`, `DEFAULT_SAFETY_THRESHOLD`.

## Metric conventions

- **Reward.** `get_series` in `common.py` reconstructs the true reward for
  `ppo_cost`, whose logged reward already has the cost subtracted.
- **Cost.** Episodic constraint violations; the safety threshold (default 25)
  is drawn as a horizontal line by the plotting scripts unless `--no_threshold`.
- **Aggregation.** Runs are aligned by `_step` and truncated to the shortest
  run (`align_and_stack`); "final" performance means the mean over the last
  `--last_k` points or `--last_frac` of training, depending on the script.
