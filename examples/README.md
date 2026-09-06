# CRAX Examples

Small, self-contained scripts showing how to use CRAX. Run them from the repo
root with the package installed (`pip install -e .`).

| Example                                           | Description |
|---------------------------------------------------|-------------|
| [`01_run_episode.py`](01_run_episode.py) | Create an environment, run one episode with random actions, and save an MP4 with a reward/cost overlay. |

## 01 — Run Episode

```bash
python examples/01_run_episode.py
```

Writes `videos/safe_goal_point_level1_random.mp4` and prints the episode return
and cumulative safety cost. Each frame is annotated with the running return and
cost; pass `--no_overlay` to get clean frames.

Useful flags:

```bash
python examples/01_run_episode.py \
  --env_name safe_goal_point \     # any name from the CRAX registry
  --level 2 \                      # difficulty 1, 2 or 3
  --steps 500 \
  --camera fixedfar \                 # camera to render from
  --width 640 --height 480 \
  --no_overlay \               # drop the reward/cost text
  --out videos/goal_point.mp4
```

Rendering is offscreen through MuJoCo, so an OpenGL backend is needed. The
script defaults to `MUJOCO_GL=egl`, which works headless; override the env var
(e.g. `MUJOCO_GL=osmesa`) if EGL is unavailable on your machine. See
[SETUP.md](../SETUP.md) for headless rendering details.
