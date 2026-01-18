# CRAX Setup Instructions

## Quick Start

### 1. Create Virtual Environment

```bash
python3.10 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 2. Install Dependencies

**Option A: Flexible versions (recommended for development)**
```bash
pip install -r requirements.txt
```

**Option B: Exact pinned versions (for reproducibility)**
```bash
pip install -r requirements-pinned.txt
```

### 3. Install Safe-Brax in Editable Mode

```bash
pip install -e .
```

### 4. Verify Installation

```bash
python -c "from brax import envs; print('✓ Brax imports successfully!')"
python -c "from brax.training.agents.ppo_lag import train; print('✓ PPO-Lagrange available!')"
```

## GPU Setup

### NVIDIA GPU with CUDA 12.x

The requirements include `jax-cuda12-plugin` which requires:
- NVIDIA GPU
- CUDA Toolkit 12.x
- cuDNN

To verify GPU is detected:
```bash
python -c "import jax; print(f'GPUs available: {jax.devices()}')"
```

### CPU-only Installation

If you don't have a GPU, remove these lines from requirements.txt:
```
jax-cuda12-plugin==0.6.0
jax-cuda12-pjrt==0.6.0
```

Then install CPU-only JAX:
```bash
pip install jax[cpu]
```

## Running Training

### Basic Usage

```bash
python train_env.py \
  --env_name safe_point_goal \
  --alg ppo_lag \
  --difficulty 1 \
  --seeds 0
```

### Using tmux for Long Training Runs

```bash
# Start training in background
tmux new -s training "python train_env.py --env_name safe_point_goal --alg ppo_lag --difficulty 1 --seeds 0"

# Detach: Ctrl+b, then d
# Reattach: tmux attach -t training
# Kill: tmux kill-session -t training
```

## Algorithm Versions

CRAX supports multiple algorithms via the --alg flag:

- PPO ("alg": "ppo") — Vanilla PPO
- PPO-Cost ("alg": "ppo_cost") — PPO with cost penalty
- PPO-Lagrange ("alg": "ppo_lag") — Lagrangian constraint handling
- PPO-PID ("alg": "ppo_pid") — PID-controlled Lagrange multiplier
- P3O ("alg": "p3o") — Penalized Proximal Policy Optimization
- FOCOPS ("alg": "focops") — First-Order Constrained Optimization in Policy Space
- PPO-Saute ("alg": "ppo_saute") — State augmentation approach for safety

The training script prints the selected algorithm on startup.

## Troubleshooting

### Mujoco Import Errors

If you see `ModuleNotFoundError: No module named 'mujoco.introspect'`:
```bash
pip install --upgrade mujoco==3.3.2 mujoco-mjx==3.3.2
```

### Version Conflicts with safety-gymnasium

The `safety-gymnasium` package requires `mujoco==2.3.3`, but Safe-Brax uses `mujoco==3.3.2`. This is fine - Safe-Brax doesn't use safety-gymnasium environments, so the warning can be ignored.

### GPU Out of Memory

Reduce the number of parallel environments:
```bash
# Example:
python train_env.py --env_name safe_point_goal --alg ppo_lag --difficulty 1 \
  --num_envs 1024 --num_eval_envs 64
```

## File Structure

```
requirements.txt         # Flexible versions (for development)
requirements-pinned.txt  # Exact versions (for reproducibility)
pyproject.toml           # Package configuration
train_env.py             # Main training script (CLI)
configs/
  training_config.py     # Shared CLI/argument definitions
  experimental_results_safe_point_goal/
    pointgoal_baselines_ppol.json  # (legacy) example config
    pointgoal_baselines_ppo.json   # (legacy) example config
scripts/
  run_from_config.py     # Legacy: run experiments from a JSON config
```

## Updating Dependencies

To update to latest compatible versions:
```bash
pip install --upgrade -r requirements.txt
pip freeze > requirements-pinned.txt  # Save new versions
```

## Weights & Biases (wandb)

Login to Weights & Biases for experiment tracking:
```bash
wandb login
```

By default, training enables wandb logging (use_wandb=True). You can disable it or set project/group/tags:
```bash
# Disable wandb
python train_env.py --env_name safe_point_goal --alg ppo_lag --use_wandb False

# Set project, group, and tags
python train_env.py --env_name safe_point_goal --alg ppo_lag \
  --wandb_project safe-brax-experimental-results \
  --wandb_group safe_point_goal \
  --wandb_tags tag1 tag2
```

