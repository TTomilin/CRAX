import os
import time

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from brax.envs.safe_ant import SafeAnt


def periodic_actions(t, action_size: int, scale: float = 0.5, omega: float = 0.2):
    """
    Generates a deterministic periodic action vector of length `action_size`.
    Uses sinusoids with phase offsets to excite multiple joints.
    """
    # Create evenly spaced phase offsets for each actuator
    phases = jnp.linspace(0.0, 2 * jnp.pi, num=action_size, endpoint=False)
    base = omega * t + phases
    return scale * jnp.sin(base).astype(jnp.float32)


def run_episodes(env, jit_step, jit_reset, steps_per_ep: int = 150, num_episodes: int = 3, seed: int = 0):
    print(f"Starting {num_episodes} episodes")

    all_states = []
    all_rewards = []  # list of (r_t, r_cum)
    all_costs = []    # list of (c_t, c_cum)
    all_feet = []     # list of feet_on_ground metric per step (float)
    all_xvel = []     # forward velocity per step (float)

    key = jax.random.PRNGKey(seed)

    t0 = time.perf_counter()

    for ep in range(num_episodes):
        key, ep_key = jax.random.split(key)
        state = jit_reset(ep_key)

        # include reset state only once to avoid long pauses at the beginning
        if ep == 0:
            all_states.append(state)

        cum_r = 0.0
        cum_c = 0.0

        for t in range(steps_per_ep):
            # deterministic periodic actions with slight deterministic per-ep scale
            scale = 0.5 + 0.1 * ((ep % 3) - 1)
            action = periodic_actions(t, env.action_size, scale=scale)

            state = jit_step(state, action)
            all_states.append(state)

            # Scalars to host
            r = float(np.asarray(jax.device_get(state.reward)).reshape(()))
            c = state.info.get("cost", state.metrics.get("cost", jnp.array(0.0)))
            c = float(np.asarray(jax.device_get(c)).reshape(()))

            feet = state.metrics.get("feet_on_ground", jnp.array(np.nan, dtype=jnp.float32))
            feet = float(np.asarray(jax.device_get(feet)).reshape(()))

            xv = state.metrics.get("x_velocity", jnp.array(np.nan, dtype=jnp.float32))
            xv = float(np.asarray(jax.device_get(xv)).reshape(()))

            cum_r += r
            cum_c += c
            all_rewards.append((r, cum_r))
            all_costs.append((c, cum_c))
            all_feet.append(feet)
            all_xvel.append(xv)

        print(f"Finished episode {ep + 1}/{num_episodes}")

    rollout_s = time.perf_counter() - t0
    print(f"[timing] rollout: {rollout_s:.3f}s  ({num_episodes*steps_per_ep} steps)")
    print("Finished all episodes")

    return all_states, all_rewards, all_costs, all_feet, all_xvel


def render_states(env, states, rewards, costs, feet_on_ground, xvel,
                  base_agent_name="safe_ant", camera="fixedfar", width=320, height=240, fps=50,
                  show_metrics=True, font="DejaVuSans-Bold"):
    t0 = time.perf_counter()
    pipeline_states = [s.pipeline_state for s in states]

    t_render0 = time.perf_counter()
    frames = env.render(
        pipeline_states,
        width=width,
        height=height,
        camera=camera,
    )
    render_s = time.perf_counter() - t_render0

    if show_metrics:
        try:
            font_obj = ImageFont.truetype(f"{font}.ttf", 14)
        except (OSError, IOError):
            font_obj = ImageFont.load_default()

        out = []
        for i, frame in enumerate(frames):
            if i == 0:
                r_cum = 0.0
                c_cum = 0.0
                feet = feet_on_ground[0] if len(feet_on_ground) > 0 else 0.0
                xv = xvel[0] if len(xvel) > 0 else 0.0
            else:
                _, r_cum = rewards[i - 1]
                _, c_cum = costs[i - 1]
                feet = feet_on_ground[i - 1] if i - 1 < len(feet_on_ground) else 0.0
                xv = xvel[i - 1] if i - 1 < len(xvel) else 0.0

            img = Image.fromarray(frame.astype(np.uint8))
            draw = ImageDraw.Draw(img)

            texts = [
                (f"Reward: {r_cum:.2f}", (50, 220, 50)),
                (f"Cost: {c_cum:.2f}", (230, 60, 60)),
                (f"Feet on ground (restricted): {feet:.0f}", (220, 220, 80)),
                (f"x vel: {xv:.3f}", (200, 200, 255)),
            ]
            for idx, (txt, color) in enumerate(texts):
                draw.text((10, 10 + 30 * idx), txt, font=font_obj, fill=color,
                          stroke_width=2, stroke_fill=(0, 0, 0))

            out.append(np.array(img))
        frames = out

    os.makedirs("videos", exist_ok=True)
    path = f"videos/{base_agent_name}.mp4"
    t_write0 = time.perf_counter()
    iio.imwrite(path, np.stack(frames), fps=fps)
    write_s = time.perf_counter() - t_write0
    print("Saved video", path)
    total_s = time.perf_counter() - t0
    print(f"[timing] render: {render_s:.3f}s | write: {write_s:.3f}s | total video: {total_s:.3f}s")


def _init_safe_ant(difficulty: int = 1):
    env = SafeAnt(difficulty=difficulty)
    step_jit = jax.jit(env.step)
    reset_jit = jax.jit(env.reset)
    return env, step_jit, reset_jit


if __name__ == '__main__':
    # Basic manual run to visualize the environment
    env, step, reset = _init_safe_ant(difficulty=1)
    states, rewards, costs, feet, xvel = run_episodes(env, step, reset, steps_per_ep=300, num_episodes=2)
    render_states(env, states, rewards, costs, feet, xvel, base_agent_name="safe_ant")
