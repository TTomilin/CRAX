"""
Render egocentric and allocentric agent views:
python scripts/render_agent_views.py --level 2 --output paper/figures/vision_views.pdf
"""

import argparse
import os
import sys

import jax
import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crax import envs
from results.common import TRANSLATIONS, allocentric_camera


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=str, nargs="+", default=["safe_goal_point", "safe_push_point", "safe_circle_point"])
    p.add_argument("--level", type=int, default=2)
    p.add_argument("--size", type=int, default=480, help="Render resolution (training uses 64)")
    p.add_argument("--backend", type=str, default="mjx")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str, default="paper/figures/vision_views.pdf")
    return p.parse_args()


def render(mj_model, renderer, pipeline_state, camera):
    d = mujoco.MjData(mj_model)
    d.qpos[:] = np.asarray(pipeline_state.qpos)
    d.qvel[:] = np.asarray(pipeline_state.qvel)
    if mj_model.nmocap > 0:
        d.mocap_pos[:] = np.asarray(pipeline_state.mocap_pos)
        d.mocap_quat[:] = np.asarray(pipeline_state.mocap_quat)
    mujoco.mj_forward(mj_model, d)
    renderer.update_scene(d, camera=camera)
    return renderer.render().copy()


def main():
    args = parse_args()
    fig, axs = plt.subplots(1, 2 * len(args.envs), figsize=(2.0 * 2 * len(args.envs), 2.4))
    for i, env_name in enumerate(args.envs):
        env = envs.get_environment(env_name, level=args.level, backend=args.backend)
        state = jax.jit(env.reset)(jax.random.PRNGKey(args.seed))
        mj_model = env.sys.mj_model
        renderer = mujoco.Renderer(mj_model, height=args.size, width=args.size)
        views = [("Egocentric", "vision"), ("Allocentric", allocentric_camera(env_name))]
        for j, (label, camera) in enumerate(views):
            ax = axs[2 * i + j]
            ax.imshow(render(mj_model, renderer, state.pipeline_state, camera),
                      interpolation="nearest")
            ax.set_title(label, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
        renderer.close()
        left, right = axs[2 * i].get_position(), axs[2 * i + 1].get_position()
        fig.text(0.5 * (left.x0 + right.x1), 0.97, TRANSLATIONS.get(env_name, env_name),
                 ha="center", va="top", fontsize=13)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.02, wspace=0.08)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
