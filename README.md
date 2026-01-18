# CRAX: Constrained Reinforcement Learning Accelerated with JAX

CRAX is a high-performance benchmark for **Constrained Reinforcement Learning (Safe RL)** built on top of [Brax](https://github.com/google/brax) and [MuJoCo XLA (MJX)](https://mujoco.readthedocs.io/en/stable/mjx.html). It provides GPU/TPU-accelerated environments with safety constraints and a suite of state-of-the-art safe RL algorithms.

<!-- GIF placeholders - add your environment GIFs here -->
<img src="assets/envs/safe_point_goal_level_1.png" width="200" height="200"/><img src="assets/envs/safe_point_goal_level_2.png" width="200" height="200"/><img src="assets/envs/safe_point_goal_level_3.png" width="200" height="200"/>

## Features

- **Massively parallel simulation**: Train agents across thousands of environments simultaneously on GPU/TPU
- **Configurable difficulty levels**: Environments with varying numbers and types of hazards
- **Multiple constraint types**: Cylindrical hazards, cubical hazards, boundary constraints, and more
- **State-of-the-art algorithms**: Implementations of leading constrained RL methods

## Safe RL Algorithms

CRAX includes efficient JAX implementations of:

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **PPO-Lagrange** | PPO with Lagrangian relaxation for constraints | [Ray et al., 2019](https://cdn.openai.com/safexp-short.pdf) |
| **PPO-PID** | PPO with PID-controlled Lagrange multiplier | [Stooke et al., 2020](https://arxiv.org/abs/2007.03964) |
| **FOCOPS** | First Order Constrained Optimization in Policy Space | [Zhang et al., 2020](https://arxiv.org/abs/2002.06506) |
| **P3O** | Penalized Proximal Policy Optimization | [Zhang et al., 2022](https://arxiv.org/abs/2205.11814) |
| **PPO-Saute** | State Augmentation for Safe RL | [Sootla et al., 2022](https://arxiv.org/abs/2202.06558) |

All algorithms share a common training infrastructure with hooks for custom loss functions and constraint handling, making it easy to implement new methods.

## Environments

CRAX provides safety-constrained versions of standard locomotion tasks:

- **SafePointGoal** - Point mass navigation with hazard avoidance
- **SafeAnt** - Ant locomotion with safety constraints
- **SafeWalker** - Walker2D with boundary and hazard constraints
- **SafeReacher** - Reacher with obstacle avoidance

Each environment supports:
- Configurable difficulty levels (number/density of hazards)
- Multiple hazard types (cylinders, cubes, mixed)
- Episodic cost tracking for constraint satisfaction
- Vision-based observations (optional)

## Installation

```bash
git clone https://github.com/your-repo/CRAX.git
cd CRAX
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -e .
```

For GPU support, ensure you have [CUDA and JAX with GPU support](https://github.com/google/jax#installation) installed.

## Quick Start

### Training an agent

```python
from brax import envs
from brax.training.agents.ppo_lag import train as ppo_lag_train

# Create environment
env = envs.get_environment('safe_point_goal', difficulty=1)

# Train with PPO-Lagrange
make_policy, params, metrics, _ = ppo_lag_train.train(
    environment=env,
    num_timesteps=10_000_000,
    episode_length=1000,
    num_envs=2048,
    safety_bound=25.0,  # Maximum allowed cost per episode
    lagrangian_coef_rate=0.01,
)
```

### Using the config system

```bash
python train_from_config.py --algorithm ppo_lag --env safe_point_goal --difficulty 1
```

## Project Structure

```
CRAX/
├── brax/
│   ├── envs/                    # Environment definitions
│   │   ├── safe_point_goal.py   # Point mass with hazards
│   │   ├── safe_ant.py          # Ant with constraints
│   │   ├── safe_walker.py       # Walker2D with constraints
│   │   ├── hazards.py           # Hazard generation utilities
│   │   └── goals.py             # Goal sampling utilities
│   └── training/
│       └── agents/
│           ├── ppo/             # Base PPO with extensibility hooks
│           ├── ppo_lag/         # PPO-Lagrange
│           ├── ppo_pid/         # PPO with PID controller
│           ├── focops/          # FOCOPS
│           ├── p3o/             # P3O
│           └── ppo_saute/       # Saute wrapper
├── configs/                     # Training configurations
└── scripts/                     # Utility scripts
```

## Architecture

CRAX uses a modular architecture where constrained RL algorithms are thin wrappers around a base PPO trainer:

```
ppo/train.py          # Base trainer with hooks (loss_fn, post_step_fn, init_aux_state_fn)
    │
    ├── ppo_lag/      # Lagrange multiplier update + lagrange loss
    ├── ppo_pid/      # PID controller + lagrange loss
    ├── focops/       # FOCOPS loss + nu update
    ├── p3o/          # P3O loss + kappa adaptation
    └── ppo_saute/    # Environment wrapper approach
```

This design minimizes code duplication and makes it easy to add new algorithms.

## Acknowledgements

CRAX is built on top of [Brax](https://github.com/google/brax), a differentiable physics engine by Google. We thank the Brax team for their excellent foundation.

If you use CRAX in your research, please cite:

```bibtex
@software{crax2025,
  title = {CRAX: Constrained Reinforcement Learning Accelerated with JAX},
  year = {2025},
}
```

And the original Brax paper:

```bibtex
@software{brax2021github,
  author = {C. Daniel Freeman and Erik Frey and Anton Raichuk and Sertan Girgin and Igor Mordatch and Olivier Bachem},
  title = {Brax - A Differentiable Physics Engine for Large Scale Rigid Body Simulation},
  url = {http://github.com/google/brax},
  year = {2021},
}
```
