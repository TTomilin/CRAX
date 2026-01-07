import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
from brax import base
from PIL import Image, ImageDraw, ImageFont
import os


from brax.envs.safe_reacher import SafeReacher


def deterministic_actions(t, scale=0.5, omega=0.2):
    """
    Returns a simple, deterministic 2D action that traces a circle over time.
    Suitable for the 2-DoF SafeReacher actuators.
    """
    return jnp.array([
        scale * jnp.cos(omega * t),
        scale * jnp.sin(omega * t),
    ], dtype=jnp.float32)


def run_episodes(env, jit_step, jit_reset, steps_per_ep: int = 100, num_episodes: int = 5, seed: int = 42):
    print(f"Starting {num_episodes} episodes")

    all_states = []
    all_rewards = []
    all_costs = []
    all_dists = []
    all_tip_xy = []     # list of (x, y)
    all_target_xy = []  # list of (x, y)

    key = jax.random.PRNGKey(seed)

    for ep in range(num_episodes):
        key, ep_key = jax.random.split(key)
        state = jit_reset(ep_key)

        # Add reset frame only for the very first episode (avoid duplicate "pause" frames)
        if ep == 0:
            all_states.append(state)

        cum_r = 0.0
        cum_c = 0.0

        for t in range(steps_per_ep):
            # deterministic action + a bit of deterministic noise per-episode
            noise_key = jax.random.fold_in(ep_key, t)
            action = deterministic_actions(t)
            step_action = action * jax.random.uniform(noise_key)

            state = jit_step(state, step_action)
            all_states.append(state)

            r = float(np.asarray(jax.device_get(state.reward)).reshape(()))
            c = state.info.get("cost", state.metrics.get("cost", jnp.array(0.0)))
            c = float(np.asarray(jax.device_get(c)).reshape(()))

            # distance tip <-> goal from metrics
            d = state.metrics.get("dist", jnp.array(np.nan, dtype=jnp.float32))
            d = float(np.asarray(jax.device_get(d)).reshape(()))

            # extract tip and target positions (XY)
            ps = state.pipeline_state
            tip_pos = (
                ps.x.take(env._tip_body_id)
                .do(base.Transform.create(pos=env._tip_offset))
                .pos
            )
            target_pos = ps.x.pos[env._target_body_id]

            tip_xy = np.asarray(jax.device_get(tip_pos[:2]))
            target_xy = np.asarray(jax.device_get(target_pos[:2]))

            cum_r += r
            cum_c += c
            all_rewards.append((r, cum_r))
            all_costs.append((c, cum_c))
            all_dists.append(d)
            all_tip_xy.append(tuple(tip_xy))
            all_target_xy.append(tuple(target_xy))

        print(f"Finished episode {ep + 1}/{num_episodes}")

    print("Finished all episodes")
    return all_states, all_rewards, all_costs, all_dists, all_tip_xy, all_target_xy


def render_states(env, states, rewards, costs, dists, tip_xy, target_xy, base_agent_name="safe_reacher",
                  camera="fixedfar", width=320, height=240, fps=10,
                  show_metrics=True, font="DejaVuSans-Bold"):
    pipeline_states = [s.pipeline_state for s in states]

    frames = env.render(
        pipeline_states,
        width=width,
        height=height,
        camera=camera
    )

    if show_metrics:
        try:
            font_obj = ImageFont.truetype(f"{font}.ttf", 20)
        except (OSError, IOError):
            font_obj = ImageFont.load_default()

        out = []
        for i, frame in enumerate(frames):
            if i == 0:
                r_cum = 0.0
                c_cum = 0.0
                d = float(np.asarray(states[0].metrics.get("dist", jnp.array(np.nan))).reshape(()))
            else:
                _, r_cum = rewards[i - 1]
                _, c_cum = costs[i - 1]
                d = dists[i - 1]

            img = Image.fromarray(frame.astype(np.uint8))
            draw = ImageDraw.Draw(img)

            reward_text = f"Reward: {r_cum:.2f}"
            cost_text = f"Cost: {c_cum:.2f}"
            tip = tip_xy[i - 1] if i > 0 else tip_xy[0]
            tgt = target_xy[i - 1] if i > 0 else target_xy[0]

            dist_text = f"Dist: {d:.3f}"
            tip_text = f"Tip: ({tip[0]:.3f}, {tip[1]:.3f})"
            tgt_text = f"Goal: ({tgt[0]:.3f}, {tgt[1]:.3f})"

            draw.text((10, 10), tip_text, font=font_obj, fill=(180, 220, 255),
                      stroke_width=2, stroke_fill=(0, 0, 0))
            draw.text((10, 40), tgt_text, font=font_obj, fill=(255, 200, 120),
                      stroke_width=2, stroke_fill=(0, 0, 0))
            draw.text((10, 70), reward_text, font=font_obj, fill=(50, 220, 50),
                      stroke_width=2, stroke_fill=(0, 0, 0))
            draw.text((10, 100), cost_text, font=font_obj, fill=(230, 60, 60),
                      stroke_width=2, stroke_fill=(0, 0, 0))
            draw.text((10, 130), dist_text, font=font_obj, fill=(230, 230, 230),
                      stroke_width=2, stroke_fill=(0, 0, 0))

            out.append(np.array(img))
        frames = out

    os.makedirs("videos", exist_ok=True)
    path = f"videos/{base_agent_name}.mp4"
    iio.imwrite(path, np.stack(frames), fps=fps)
    print("Saved video", path)


def _init_safe_reacher():
    env = SafeReacher()
    step_jit = jax.jit(env.step)
    reset_jit = jax.jit(env.reset)
    return env, step_jit, reset_jit


if __name__ == '__main__':
    env, step, reset = _init_safe_reacher()
    states, rewards, costs, dists, tip_xy, target_xy = run_episodes(env, step, reset)
    render_states(env, states, rewards, costs, dists, tip_xy, target_xy, "safe_reacher")
