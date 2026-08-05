# Vision-Based Training in CRAX

CRAX supports training agents from pixel observations in addition to the default state-vector observations. This document explains how to use the feature.

## Quick Start

Yes, it is as simple as adding `--vision`:

```bash
# State-based training (default, unchanged)
python train_env.py --env_name safe_goal_point --alg ppo_lag --difficulty 1

# Vision-based training — just add --vision
python train_env.py --env_name safe_goal_point --alg ppo_lag --difficulty 1 --vision
```

This works with all training scripts:

```bash
# Curriculum training with vision
python train_curriculum.py --env_name safe_goal_point --alg ppo_lag --vision

# Transfer learning with vision
python train_transfer.py --env_name safe_goal_point --vision
```

All 5 safe RL algorithms (PPO-Lag, PPO-PID, FOCOPS, P3O, PPO-Saute) and plain PPO work with vision out of the box.

## What Happens Under the Hood

When `--vision` is set, the training script:

1. Wraps the environment with `PixelObservationWrapper`, which renders pixels from MuJoCo cameras at each step
2. Switches the network architecture from MLP to CNN (Nature DQN architecture: 32-64-64 filters)
3. For safe RL algorithms, creates a CNN-based cost value network alongside the policy and value networks
4. Enables pixel augmentation (random translation) during training

The observation changes from a flat state vector to a dict:

```python
# Without --vision
obs.shape  # (62,)

# With --vision (default: pixels+state mode)
obs['pixels/vision'].shape   # (84, 84, 3)
obs['state'].shape           # (62,)
```

## Vision Options

| Flag | Default | Description |
|------|---------|-------------|
| `--vision` | `False` | Enable pixel observations |
| `--vision_cameras` | `['vision']` | Camera names from the MuJoCo XML |
| `--vision_height` | `84` | Render height in pixels |
| `--vision_width` | `84` | Render width in pixels |
| `--vision_obs_mode` | `pixels+state` | `'pixels+state'` or `'pixels'` |
| `--vision_frame_stack` | `1` | Number of frames to stack (channel-wise) |
| `--vision_grayscale` | `False` | Convert to grayscale (1 channel instead of 3) |
| `--vision_render_workers` | `4` | CPU threads for parallel rendering |

### Observation Modes

- **`pixels+state`** (default): The agent sees both rendered images AND the original state vector. The CNN processes pixels, the state is concatenated, and the MLP head produces actions. This is the recommended starting point.
- **`pixels`**: The agent sees only rendered images. Harder, but tests pure visual learning.

### Frame Stacking

Frame stacking provides temporal information (useful since a single image has no velocity info):

```bash
python train_env.py --env_name safe_goal_point --alg ppo_lag --vision --vision_frame_stack 3
```

With `frame_stack=3`, the pixel observation has shape `(84, 84, 9)` — three RGB frames concatenated along the channel dimension.

### Multiple Cameras

Agents have `vision` (front-facing) and `vision_back` cameras. Use both for a wider field of view:

```bash
python train_env.py --env_name safe_goal_point --alg ppo_lag --vision \
    --vision_cameras vision vision_back
```

This produces `obs['pixels/vision']` and `obs['pixels/vision_back']`, each processed by a separate CNN and concatenated before the MLP head.

## Performance Considerations

Vision training is significantly slower than state-based training because MuJoCo rendering runs on CPU while physics runs on GPU.

**Recommended settings for vision:**

```bash
python train_env.py --env_name safe_goal_point --alg ppo_lag --vision \
    --num_envs 128 \
    --vision_height 64 --vision_width 64 \
    --vision_render_workers 8
```

| Setting | State-based | Vision |
|---------|------------|--------|
| `num_envs` | 2048 | 64–256 |
| Resolution | N/A | 64x64 or 84x84 |
| Throughput | ~50k steps/sec | ~1k–5k steps/sec |

Tips:
- Reduce `--num_envs` (the biggest lever on rendering cost)
- Use 64x64 instead of 84x84 for faster iteration
- Increase `--vision_render_workers` to match your CPU cores
- Use `--vision_grayscale` to reduce memory by 3x
- Use `--vision_obs_mode pixels+state` (rather than `pixels`) for easier learning

## Programmatic Usage

```python
from crax import envs

# Create a vision environment
env = envs.get_environment(
    'safe_goal_point',
    level=1,
    vision=True,
    vision_kwargs=dict(
        cameras=('vision',),
        height=84,
        width=84,
        obs_mode='pixels+state',
        frame_stack=3,
    ),
)

# Use it like any other env
state = env.reset(jax.random.PRNGKey(0))
print(state.obs['pixels/vision'].shape)  # (84, 84, 9)
print(state.obs['state'].shape)          # (62,)

# Works with jit, vmap, and the full training wrapper stack
import jax
jit_step = jax.jit(env.step)
next_state = jit_step(state, jnp.zeros(env.action_size))
```

To use with a custom training loop, create the vision network factory:

```python
from run_utils import make_vision_network_factory

# For safe RL algorithms (automatically adds cost_value_network)
network_factory = make_vision_network_factory(
    'ppo_lag',
    policy_obs_key='state',
    value_obs_key='state',
)

# Pass to training
from crax.training.agents.ppo_lag import train as ppo_lag
ppo_lag.train(
    environment=env,
    network_factory=network_factory,
    augment_pixels=True,
    ...
)
```

## Available Cameras

All CRAX agents have these cameras defined in their XML:

| Camera | Description | Attached to |
|--------|-------------|-------------|
| `vision` | Front-facing egocentric view | Agent body |
| `vision_back` | Rear-facing egocentric view | Agent body |
| `track` | Third-person tracking view | Tracks agent COM |
| `fixedfar` | Fixed far-away view | World frame |

The default `vision` camera is an egocentric front-facing view with 90-degree FOV, which moves and rotates with the agent.
