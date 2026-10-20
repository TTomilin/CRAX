"""Counterfactual hazard relabelling for on-policy safe RL.

Pathway hazards are non-collidable, so the walker's dynamics do not depend on
where the hazards are. A recorded segment can therefore be replayed exactly
under a different hazard layout: actions, rewards, terminations and joint
states stay the same, and only the lidar part of the observation and the hazard
cost change. Both are recomputed from the kinematics the env stores in
`info['cf_kin']` / `info['cf_feet']`.

The relabelled segments are off-policy w.r.t. the observations they now carry,
so the PPO trainer re-anchors their behaviour log-probs at the current policy
and corrects with truncated importance weights (see `ppo/train.py`).

Modes:
  - 'random': one fresh layout from the task distribution per segment
    (unbiased augmentation of the hazard context).
  - 'optimistic': the lowest-cost of `num_candidates` fresh layouts per
    segment, i.e. hindsight placements where the realised gait was safe. This
    is what shows a conservative policy that long strides can pay off.
"""

import math
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from training import types

MODES = ('none', 'random', 'optimistic')
CF_EXTRA_FIELDS = ('cf_kin', 'cf_feet', 'hazard_cost')

# Distance (m) the walker may cover within one unroll segment
_MAX_SEGMENT_TRAVEL = 4.0


def _sample_layout_x(env, key: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Vectorised version of `env.sample_hazard_positions` returning (x, y)."""
    n_h = env._num_hazards
    k0, kg, ky = jax.random.split(key, 3)
    start = jax.random.uniform(k0, (), minval=1.0, maxval=2.0)
    gaps = jax.random.uniform(kg, (n_h,), minval=env._min_gap, maxval=env._max_gap)
    xs = start + jnp.cumsum(gaps)
    ys = jax.random.uniform(ky, (n_h,), minval=-env._lateral_jitter, maxval=env._lateral_jitter)
    return xs, ys


def make_hazard_relabel_fn(
        environment: Any,
        mode: str = 'optimistic',
        num_candidates: int = 4,
) -> Callable[[types.Transition, jax.Array], Tuple[types.Transition, Dict[str, jax.Array]]]:
    """Builds `relabel_fn(data, key) -> (data, metrics)` for [B, T] pathway transitions."""
    env = environment.unwrapped
    for attr in ('hazard_lidar', 'hazard_cost_from_feet', '_lidar_bins'):
        if not hasattr(env, attr):
            raise ValueError(
                f"Counterfactual relabelling needs a SafePathway env, got {type(env).__name__}")
    if mode not in MODES[1:]:
        raise ValueError(f"Unknown counterfactual mode '{mode}', expected one of {MODES[1:]}")
    n_h = env._num_hazards
    # Hazards kept around the segment start: every hazard within lidar range
    # behind it, and within lidar range of anywhere the walker can reach ahead
    behind = math.ceil(env._lidar_max_dist / env._min_gap) + 1
    ahead = math.ceil((env._lidar_max_dist + _MAX_SEGMENT_TRAVEL) / env._min_gap) + 1
    window = min(max(behind + ahead, env._lidar_max_hazards), n_h)
    bins = env._lidar_bins
    num_candidates = num_candidates if mode == 'optimistic' else 1
    height = jnp.float32(env._hazard_height)

    def candidate(kin0_x, key):
        xs, ys = _sample_layout_x(env, key)
        # xs is sorted, so the hazards nearest the segment form a contiguous window
        first = jnp.clip(jnp.searchsorted(xs, kin0_x) - behind, 0, n_h - window)
        ids = first + jnp.arange(window)
        layout = jnp.stack([xs[ids], ys[ids], jnp.full((window,), height)], axis=-1)
        return layout, ids

    def relabel_segment(obs, next_obs, cost, hazard_cost, kin, feet, key):
        # obs/next_obs [T, D], cost/hazard_cost [T], kin [T, 2, 3], feet [T, F, 3]
        keys = jax.random.split(key, num_candidates)
        layouts, ids = jax.vmap(candidate, in_axes=(None, 0))(kin[0, 0, 0], keys)
        step_cost = jax.vmap(
            lambda lay, hid: jax.vmap(env.hazard_cost_from_feet, in_axes=(0, None, None))(feet, lay, hid)
        )(layouts, ids)  # [K, T]
        best = jnp.argmin(step_cost.sum(axis=-1)) if num_candidates > 1 else 0
        layout, new_hazard_cost = layouts[best], step_cost[best]

        lidar = jax.vmap(env.hazard_lidar, in_axes=(0, None))(kin[:, 0], layout)
        next_lidar = jax.vmap(env.hazard_lidar, in_axes=(0, None))(kin[:, 1], layout)
        obs = obs.at[:, -bins:].set(lidar)
        next_obs = next_obs.at[:, -bins:].set(next_lidar)
        # Terminal (unhealthy) cost is hazard-independent and carried over
        cost = cost - hazard_cost + new_hazard_cost
        return obs, next_obs, cost, new_hazard_cost

    def relabel_fn(data: types.Transition, key: jax.Array):
        if not isinstance(data.observation, jax.Array):
            raise ValueError("Counterfactual relabelling supports flat state observations only")
        se = data.extras['state_extras']
        keys = jax.random.split(key, data.observation.shape[0])
        obs, next_obs, cost, hazard_cost = jax.vmap(relabel_segment)(
            data.observation, data.next_observation, se['cost'], se['hazard_cost'],
            se['cf_kin'], se['cf_feet'], keys)
        new_se = {**se, 'cost': cost, 'hazard_cost': hazard_cost}
        cf = data._replace(
            observation=obs, next_observation=next_obs,
            extras={**data.extras, 'state_extras': new_se})
        metrics = {
            'cf/real_hazard_cost': jnp.mean(se['hazard_cost']),
            'cf/hazard_cost': jnp.mean(hazard_cost),
        }
        return cf, metrics

    return relabel_fn


def counterfactual_train_kwargs(
        environment: Any,
        extra_fields: Tuple[str, ...],
        action_repeat: int,
        counterfactual_mode: str,
        counterfactual_ratio: float,
        counterfactual_candidates: int,
        counterfactual_rho_clip: float,
) -> Dict[str, Any]:
    """Kwargs for `ppo_train.train` that enable counterfactual relabelling (empty if disabled)."""
    if counterfactual_mode in (None, 'none') or counterfactual_ratio <= 0:
        return {'extra_fields': extra_fields}
    if action_repeat != 1:
        raise ValueError("Counterfactual relabelling requires action_repeat=1: "
                         "info['cf_kin'] only holds the last physics sub-step")
    return {
        'extra_fields': tuple(extra_fields) + CF_EXTRA_FIELDS,
        'counterfactual_fn': make_hazard_relabel_fn(
            environment, mode=counterfactual_mode, num_candidates=counterfactual_candidates),
        'counterfactual_ratio': counterfactual_ratio,
        'counterfactual_rho_clip': counterfactual_rho_clip,
    }
