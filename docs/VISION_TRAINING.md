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

All on-policy algorithms — plain PPO and the safe RL variants PPO-Lag, PPO-PID, FOCOPS, P3O, PPO-Saute, CRPO — work with vision out of the box.

The off-policy algorithms — SAC, SAC-Lag, SAC-PID — also support `--vision`, with a few differences worth knowing about: see [Off-policy: SAC / SAC-Lag / SAC-PID](#off-policy-sac--sac-lag--sac-pid) below.

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

## Off-policy: SAC / SAC-Lag / SAC-PID

SAC, SAC-Lag, and SAC-PID are off-policy: they keep a replay buffer of past transitions and train on samples from it, rather than consuming a rollout once (like PPO and its safe-RL variants do). That difference matters for vision:

- **Replay buffer memory.** Every stored transition now includes one or more `(H, W, C)` image frames. CRAX's replay buffer stores frames as `uint8` (not upcast to `float32`) so a full buffer doesn't blow up memory 4x, but `--max_replay_size` (in *transitions*, not bytes) still bounds total size directly — the state-vector default (`num_timesteps`) is far too large for images, so vision runs default to a much smaller buffer (100k transitions) unless you pass `--max_replay_size` explicitly.
- **Pixel augmentation** is applied per-sample at training time (each replay sample gets its own independent random crop-pad shift), not shared across a rollout the way PPO's is.
- **Every Q-critic gets its own CNN encoder**, independent of the policy's and of each other — 2 CNNs total for plain SAC (policy + Q), 3 for SAC-Lag/SAC-PID (policy + reward-Q + cost-Q) — more parameters/compute than the shared-encoder PPO path.

```bash
# Off-policy safe RL
python train_env.py --env_name safe_goal_point --alg sac_lag --difficulty 1 --vision \
    --max_replay_size 50000

# Off-policy, unconstrained
python train_env.py --env_name safe_goal_point --alg sac --difficulty 1 --vision \
    --max_replay_size 50000
```

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

# On-policy safe RL algorithms (automatically adds cost_value_network)
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

`make_vision_network_factory` routes to a CNN reward-Q-critic for `'sac'`, and to CNN reward-/cost-Q-critics for `'sac_lag'`/`'sac_pid'`:

```python
network_factory = make_vision_network_factory(
    'sac',
    policy_obs_key='state',
    value_obs_key='state',
)

from crax.training.agents.sac import train as sac
sac.train(
    environment=env,
    network_factory=network_factory,
    augment_pixels=True,
    max_replay_size=50_000,
    ...
)
```

```python
network_factory = make_vision_network_factory(
    'sac_lag',
    policy_obs_key='state',
    value_obs_key='state',
)

from crax.training.agents.sac_lag import train as sac_lag
sac_lag.train(
    environment=env,
    network_factory=network_factory,
    augment_pixels=True,
    max_replay_size=50_000,  # see "Off-policy: SAC-Lag / SAC-PID" above
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
