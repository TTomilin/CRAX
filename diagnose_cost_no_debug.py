#!/usr/bin/env python3
"""Check that HumanoidHop cost updates correctly with debug disabled.

This script instantiates the environment with contact metrics disabled
(which also turns off the debug flag) and steps through a short rollout
while printing the cost stored in ``state.info['cost']``.  It also prints
the vertical positions of both shins so you can correlate contact events
with cost spikes when running locally.

Usage (locally, not via sbatch):

    python diagnose_cost_no_debug.py --steps 40 --seed 0
"""

from __future__ import annotations

import argparse
from typing import Iterable

import jax
import jax.numpy as jnp
from jax import device_get
import numpy as np

from brax import envs


def _to_float(value: jax.Array) -> float:
    """Convert a JAX scalar array to a Python float."""

    return float(device_get(value))


def run_rollout(steps: int, seed: int) -> Iterable[float]:
    """Run a short rollout and yield the cost at each step."""

    env_kwargs = {
        "hopping_leg": "left",
        "contact_threshold": 0.0,  # make it easy to trigger violations
        "cost_weight": 1.0,
        "enable_contact_metrics": False,  # disables debug mode
    }
    env = envs.get_environment("humanoid_hop", **env_kwargs)

    print("HumanoidHop diagnostic (debug disabled)")
    print(f"  debug flag: {getattr(env, '_debug', 'unknown')} (expected False)")
    print(f"  hopping_leg: {env._hopping_leg}")
    print(f"  contact_threshold: {env._contact_threshold}")
    print()

    key = jax.random.PRNGKey(seed)
    state = env.reset(key)

    zero_action = jnp.zeros(env.action_size)
    costs: list[float] = []

    header = "step | cost     | left_z   | right_z  | ||action|| | done"
    print(header)
    print("-" * len(header))

    for step in range(steps):
        if step < 10:
            action = zero_action
        else:
            key, subkey = jax.random.split(key)
            action = jax.random.uniform(
                subkey, shape=(env.action_size,), minval=-1.0, maxval=1.0
            )

        state = env.step(state, action)

        cost_value = state.info.get("cost")
        if cost_value is None:
            raise RuntimeError("'cost' missing from state.info; measurement failed")

        cost_scalar = _to_float(cost_value)
        costs.append(cost_scalar)

        left_z = _to_float(state.pipeline_state.x.pos[env._left_foot_body_idx, 2])
        right_z = _to_float(state.pipeline_state.x.pos[env._right_foot_body_idx, 2])
        action_norm = _to_float(jnp.linalg.norm(action))
        done_flag = int(round(_to_float(state.done)))

        print(
            f"{step:4d} | {cost_scalar:8.4f} | {left_z:8.4f} | {right_z:8.4f} |"
            f" {action_norm:9.4f} | {done_flag}"
        )

        if done_flag:
            break

    return costs


def summarize_costs(costs: Iterable[float]) -> None:
    """Print summary statistics for the collected costs."""

    cost_array = np.array(list(costs), dtype=np.float32)
    if cost_array.size == 0:
        print("No costs recorded; the rollout terminated immediately.")
        return

    unique_count = len({f"{c:.6f}" for c in cost_array})

    print()
    print("Cost summary (Python-side, debug disabled):")
    print(f"  mean: {cost_array.mean():.6f}")
    print(f"  std:  {cost_array.std():.6f}")
    print(f"  min:  {cost_array.min():.6f}")
    print(f"  max:  {cost_array.max():.6f}")
    print(f"  distinct samples (rounded to 1e-6): {unique_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that the HumanoidHop contact cost is populated without"
            " enabling debug metrics."
        )
    )
    parser.add_argument("--steps", type=int, default=40, help="number of rollout steps")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    args = parser.parse_args()

    costs = run_rollout(steps=args.steps, seed=args.seed)
    summarize_costs(costs)


if __name__ == "__main__":
    main()


