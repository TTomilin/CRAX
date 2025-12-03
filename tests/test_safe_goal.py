import jax
import numpy as np
from brax.envs.SafePointGoal import SafePointGoal, default_config
import imageio.v3 as iio


def run_random_episode(env, jit_step, jit_reset, steps: int = 100):
    seed = jax.random.PRNGKey(4242)
    action = jax.numpy.ones(env.action_size)
    state = jit_reset(seed)

    print(f"Starting episode")

    states = [state]
    for i in range(steps):
        key = jax.random.PRNGKey(i)
        step_action = action*jax.random.uniform(key)
        state = jit_step(state, step_action)
        states.append(state)
        print("Step", i, "Actions", step_action)

    print(f"Finished episode")
    return states

def render_states(env, states, base_agent_name):
    # 2. Render the states
    pipeline_states = [s.pipeline_state for s in states]

    frames = env.render(
        pipeline_states,
        width=320,
        height=240,
        camera="fixedfar"  # or camera ID
    )

    # 3. Save as video
    path = "videos/" + base_agent_name + ".mp4"
    iio.imwrite(path, np.stack(frames), fps=100)
    print("Saved video ", path)

def _init_safe_goal(base_agent_name="ant.xml"):
    cfg = default_config()
    cfg.base_agent_file_name = base_agent_name
    env = SafePointGoal(cfg)
    step_jit = jax.jit(env.step)
    reset_jit = jax.jit(env.reset)
    return env, step_jit, reset_jit


if __name__ == '__main__':
    names = ["ant.xml", "half_cheetah.xml", "hopper.xml",
             "humanoid.xml", "humanoidstandup.xml", "point.xml",
             "swimmer.xml", "walker2d.xml"]
    # names = ["ant.xml"]

    for name in names:
        env, step, reset = _init_safe_goal(base_agent_name=name)
        states = run_random_episode(env, step, reset, steps=300)
        render_states(env, states, name.split(".")[0])
