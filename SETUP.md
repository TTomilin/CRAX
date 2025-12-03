# Safe-Brax Setup Instructions

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
python -c "from brax.training.agents.ppo_lagrange_v3 import train; print('✓ PPO-Lagrange V3 available!')"
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
python train_from_config.py \
  --config configs/experimental_results_safe_point_goal/pointgoal_baselines_ppol.json \
  --seeds 0
```

### Using tmux for Long Training Runs

```bash
# Start training in background
tmux new -s training "python train_from_config.py --config CONFIG.json --seeds 0"

# Detach: Ctrl+b, then d
# Reattach: tmux attach -t training
# Kill: tmux kill-session -t training
```

## Algorithm Versions

Safe-Brax supports multiple PPO variants:

- **PPO** (`"alg": "ppo"`) - Vanilla PPO
- **PPO-Cost** (`"alg": "ppo_cost"`) - PPO with cost penalty
- **PPO-Lagrange V2** (`"alg": "ppo_lagrange_v2"`) - Up-to-date PPO-Lagrangian

The training script automatically prints which version is being used.

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
```json
{
  "num_envs": 1024,  // Try reducing from 2048
  "num_eval_envs": 64  // Try reducing from 128
}
```

## File Structure

```
requirements.txt         # Flexible versions (for development)
requirements-pinned.txt  # Exact versions (for reproducibility)
pyproject.toml          # Package configuration
train_from_config.py    # Main training script
configs/                # Training configurations
  experimental_results_safe_point_goal/
    pointgoal_baselines_ppol.json  # PPO-Lagrange config
    pointgoal_baselines_ppo.json   # Vanilla PPO config
```

## Updating Dependencies

To update to latest compatible versions:
```bash
pip install --upgrade -r requirements.txt
pip freeze > requirements-pinned.txt  # Save new versions
```

## Wandb Configuration

Login to Weights & Biases for experiment tracking:
```bash
wandb login
```

Or run without wandb:
```bash
python train_from_config.py --config CONFIG.json --seeds 0 --no-wandb
```

