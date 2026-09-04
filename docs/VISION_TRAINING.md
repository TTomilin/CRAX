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

1. Forces `backend='mjx'` on the environment (`geom_xpos`/`cam_xpos`, which the renderer reads, are only populated by the MJX pipeline)
2. Wraps the *already-vectorized* env with `GpuPixelObservationWrapper`, which ray-traces pixels on GPU with MJWarp — no host round-trip, no CPU render pass
3. Switches the network architecture from MLP to CNN (Nature DQN architecture: 32-64-64 filters)
4. For safe RL algorithms, creates a CNN-based cost value network alongside the policy and value networks
5. Enables pixel augmentation (random translation) during training
6. Caps `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` (unless already set) so MJWarp has GPU memory to build its own render context beside JAX

Wrapper placement matters: MJWarp's render context is sized for a static
`num_envs` and cannot be `jax.vmap`'d, so the wrapper is applied *after*
`wrap_for_training` inside the trainer, not at env-construction time.

The observation changes from a flat state vector to a dict:

```python
# Without --vision
obs.shape  # (62,)

# With --vision (default obs_mode: pixels, frame_stack 3)
obs['pixels/vision'].shape   # (64, 64, 9)  uint8

# With --vision --vision_obs_mode pixels+state
obs['pixels/vision'].shape   # (64, 64, 9)
obs['state'].shape           # (62,)
```

The pixel key is always `pixels/<camera name>`.

## Vision Options

| Flag | Default | Description |
|------|---------|-------------|
| `--vision` | `False` | Enable pixel observations |
| `--vision_camera` | morphology-dependent (see below) | Single camera name from the MuJoCo XML |
| `--vision_height` | `64` | Render height in pixels |
| `--vision_width` | `64` | Render width in pixels |
| `--vision_obs_mode` | `pixels` | `'pixels'` or `'pixels+state'` |
| `--vision_frame_stack` | `3` | Number of frames to stack (channel-wise) |

Related video flags (vision runs only):

| Flag | Default | Description |
|------|---------|-------------|
| `--video_every_steps` | `25_000_000` | Log a clip to wandb roughly every N env steps, rendered from the policy's own GPU pixels. `0` disables. Cadence is bounded below by the eval interval (`--num_evals`) |
| `--periodic_video_steps` | `300` | Env steps per periodic clip |
| `--skip_video` | off | Disables periodic and end-of-run clips |

### Default Camera

If `--vision_camera` is not passed, it is picked from the environment name by
substring match (`VISION_CAMERA_OVERRIDES` in `run_utils.py`):

| Substring in `--env_name` | Camera |
|---------------------------|--------|
| `humanoid`, `ant`, `cheetah`, `walker2d`, `spider` | `track` (external chase view) |
| `reacher` | `fixedfar` |
| anything else (e.g. point agents) | `vision` (egocentric) |

### Observation Modes

- **`pixels`** (default): The agent sees only rendered images. This is the pure visual-learning setting.
- **`pixels+state`**: The agent sees both rendered images AND the original state vector. The CNN processes pixels, the state is concatenated, and the MLP head produces actions. Easier, useful as a sanity baseline.

### Frame Stacking

Frame stacking provides temporal information (a single image has no velocity info):

```bash
python train_env.py --env_name safe_goal_point --alg ppo_lag --vision --vision_frame_stack 3
```

With `frame_stack=3`, the pixel observation has shape `(64, 64, 9)` — three RGB frames concatenated along the channel dimension. Frames are stacked in `state.info['_gpu_pixel_buffer']` and shifted each step.

### One Camera Per Run

The GPU renderer renders exactly one camera per run: `--vision_camera` is a
single name, not a list. `create_render_context` is built with only that
camera active. To compare viewpoints, run separate experiments.

Agents typically define `vision` (front-facing egocentric), `vision_back`,
`track` and `fixedfar` — see [Available Cameras](#available-cameras). Cameras
listed in `--cameras` are used for *video recording*, not for observations.

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

Rendering runs on the GPU alongside physics, so vision training is far closer
to state-based throughput than a CPU-render pipeline would be — but the ray
tracer, the extra CNN encoders and the GPU memory split with MJWarp all cost.

Levers, in rough order of impact:

- Reduce `--num_envs`. The render context is allocated for exactly this many worlds, so it drives both render time and MJWarp's memory footprint.
- Reduce resolution (`--vision_height` / `--vision_width`). Cost scales with pixel count.
- Reduce `--vision_frame_stack` — it does not add render work, but it multiplies the CNN's input channels and (for off-policy algorithms) replay-buffer size.
- Set `XLA_PYTHON_CLIENT_MEM_FRACTION` yourself if the automatic `0.5` split leaves either JAX or MJWarp short.
- Set `--video_every_steps 0` (or `--skip_video`) to drop periodic clip rendering.

```bash
python train_env.py --env_name safe_goal_point --alg ppo_lag --vision \
    --num_envs 512 \
    --vision_height 64 --vision_width 64
```

If you hit an out-of-memory error at startup, it is usually MJWarp failing to
build its render context inside the memory JAX left it: lower `--num_envs`
first, then `XLA_PYTHON_CLIENT_MEM_FRACTION`.

## Programmatic Usage

`get_environment(..., vision=True)` vmaps the env internally (MJWarp needs a
static batch size), so `vision_kwargs` **must** carry `num_envs`:

```python
import jax
import jax.numpy as jnp
from crax import envs

num_envs = 8
env = envs.get_environment(
    'safe_goal_point',
    level=1,
    backend='mjx',          # required: the renderer reads mjx-only fields
    vision=True,
    vision_kwargs=dict(
        num_envs=num_envs,  # required: sizes MJWarp's render context
        camera='vision',    # one camera, not a list
        height=64,
        width=64,
        obs_mode='pixels',  # or 'pixels+state'
        frame_stack=3,
    ),
)

# The env is already batched -- reset takes a batch of keys.
state = env.reset(jax.random.split(jax.random.PRNGKey(0), num_envs))
print(state.obs['pixels/vision'].shape)  # (8, 64, 64, 9) uint8

jit_step = jax.jit(env.step)
next_state = jit_step(state, jnp.zeros((num_envs, env.action_size)))
```

`envs.create(..., vision=True)` does the same after applying the
Episode/Vmap/AutoReset training wrappers, taking `num_envs` from `batch_size`
if `vision_kwargs['num_envs']` is absent.

Do **not** wrap a single unbatched env and `jax.vmap` the result — the FFI
call shape-checks before vmap's batching rule applies. See the module
docstring in `crax/envs/wrappers/pixel_observation_gpu.py`.

To use with a custom training loop, create the vision network factory. Note
that the trainers apply `GpuPixelObservationWrapper` themselves (after
`wrap_for_training`), so pass an **unwrapped** env plus `vision_kwargs` —
don't hand them the `vision=True` env built above:

```python
from crax import envs
from run_utils import make_vision_network_factory

env = envs.get_environment('safe_goal_point', level=1, backend='mjx')

# obs_mode 'pixels'       -> obs_key '' (pixels only, no state branch)
# obs_mode 'pixels+state' -> obs_key 'state'
obs_key = ''

# On-policy safe RL algorithms (automatically adds cost_value_network)
network_factory = make_vision_network_factory(
    'ppo_lag',
    policy_obs_key=obs_key,
    value_obs_key=obs_key,
)

from crax.training.agents.ppo_lag import train as ppo_lag
ppo_lag.train(
    environment=env,
    network_factory=network_factory,
    augment_pixels=True,
    vision_kwargs=dict(
        camera='vision', height=64, width=64,
        obs_mode='pixels', frame_stack=3,
    ),  # num_envs is filled in by the trainer
    ...
)
```

`make_vision_network_factory` routes to a CNN reward-Q-critic for `'sac'`, and to CNN reward-/cost-Q-critics for `'sac_lag'`/`'sac_pid'`:

```python
network_factory = make_vision_network_factory(
    'sac',
    policy_obs_key=obs_key,
    value_obs_key=obs_key,
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
    policy_obs_key=obs_key,
    value_obs_key=obs_key,
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

`vision` is an egocentric front-facing view with 90-degree FOV that moves and
rotates with the agent. It is the default for point agents; the locomotion
morphologies default to `track` instead (see [Default Camera](#default-camera)).

Any of these can be passed to `--vision_camera` (one per run) or listed in
`--cameras` for video recording. `scripts/check_vision_cameras.py` prints the
cameras actually present in a given environment's XML.
