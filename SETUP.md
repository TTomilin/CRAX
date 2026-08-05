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

### 3. Install CRAX in Editable Mode

```bash
pip install -e .
```

### 4. Verify Installation

```bash
python -c "from crax import envs; print('CRAX imports successfully!')"
python -c "from crax.training.agents.ppo_lag import train; print('PPO-Lagrange available!')"
python -c "from crax.training.agents.focops import train; print('FOCOPS available!')"
python -c "from crax.training.agents.p3o import train; print('P3O available!')"
```

## GPU Setup

### NVIDIA GPU with CUDA 12.x

The requirements include `jax-cuda12-plugin` which requires:
- NVIDIA GPU
- CUDA Toolkit 12.x
- cuDNN

To verify GPU is detected:
```bash
python -c "import jax; print(f'Devices: {jax.devices()}')"
```

### CPU-only Installation

If you don't have a GPU, remove these lines from requirements.txt:
```
jax-cuda12-plugin
jax-cuda12-pjrt
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

### Using a Config File

```bash
python scripts/run_from_config.py configs/pointgoal_lidar/ppol_bound_0.05.json
```

### Using tmux for Long Training Runs

```bash
# Start training in background
tmux new -s training "python train_env.py --env_name safe_point_goal --alg ppo_lag --difficulty 1 --seeds 0"

# Detach: Ctrl+b, then d
# Reattach: tmux attach -t training
# Kill: tmux kill-session -t training
```

## Available Algorithms

CRAX supports multiple constrained RL algorithms via the `--alg` flag:

| Algorithm | Flag | Description |
|-----------|------|-------------|
| **PPO** | `ppo` | Vanilla PPO (unconstrained baseline) |
| **PPO-Cost** | `ppo_cost` | PPO with cost penalty in reward |
| **PPO-Lagrange** | `ppo_lag` | PPO with Lagrangian constraint relaxation |
| **PPO-PID** | `ppo_pid` | PPO with PID-controlled Lagrange multiplier |
| **FOCOPS** | `focops` | First Order Constrained Optimization in Policy Space |
| **P3O** | `p3o` | Penalized Proximal Policy Optimization |
| **PPO-Saute** | `ppo_saute` | State Augmentation for Safe RL |

The training script prints the selected algorithm on startup.

## Available Environments

| Environment | Description |
|-------------|-------------|
| `safe_point_goal` | Point mass navigation with hazard avoidance |
| `safe_ant` | Ant locomotion with safety constraints |
| `safe_walker` | Walker2D with boundary constraints |
| `safe_reacher` | Reacher with obstacle avoidance |

Most environments support a `--difficulty` parameter (1-3) that controls hazard density.

## Troubleshooting

### Mujoco Import Errors

If you see `ModuleNotFoundError: No module named 'mujoco.introspect'`:
```bash
pip install --upgrade mujoco mujoco-mjx
```

### GPU Out of Memory

Reduce the number of parallel environments:
```bash
python train_env.py --env_name safe_point_goal --alg ppo_lag --difficulty 1 \
  --num_envs 1024 --num_eval_envs 64
```

### Rendering Issues (Headless Server)

Set the MuJoCo rendering backend:
```bash
export MUJOCO_GL=egl  # or osmesa
```

## File Structure

```
CRAX/
├── crax/
│   ├── envs/                    # Environment definitions
│   │   ├── safe_point_goal.py
│   │   ├── safe_ant.py
│   │   ├── safe_walker.py
│   │   ├── safe_reacher.py
│   │   ├── hazards.py           # Hazard generation
│   │   └── goals.py             # Goal sampling
│   └── training/
│       └── agents/
│           ├── ppo/             # Base PPO with hooks
│           ├── ppo_lag/         # PPO-Lagrange
│           ├── ppo_pid/         # PPO-PID
│           ├── focops/          # FOCOPS
│           ├── p3o/             # P3O
│           └── ppo_saute/       # Saute wrapper
├── configs/                     # Training configurations
├── scripts/                     # Utility scripts
├── train_env.py                 # Main training script (CLI)
├── requirements.txt             # Flexible versions
├── requirements-pinned.txt      # Exact versions
└── pyproject.toml               # Package configuration
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

By default, training enables wandb logging. You can disable it or configure it:
```bash
# Disable wandb
python train_env.py --env_name safe_point_goal --alg ppo_lag --use_wandb False

# Set project, group, and tags
python train_env.py --env_name safe_point_goal --alg ppo_lag \
  --wandb_project crax-experiments \
  --wandb_group safe_point_goal \
  --wandb_tags tag1 tag2
```
