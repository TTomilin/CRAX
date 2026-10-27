"""Counterfactual hazard relabelling for on-policy safe RL.

Pathway, Reacher and Button floor hazards and SafeGrid lava are non-collidable, so the agent's dynamics do not
depend on where the hazards are. A recorded segment can therefore be replayed
exactly under a different hazard layout: actions, rewards, terminations and
joint states stay the same, and only the lidar part of the observation and the
hazard cost change. Both are recomputed from kinematics the env stores in its
info: `cf_kin` / `cf_feet` (Pathway), `cf_theta` (Reacher) or `cf_view` (Button);
SafeGrid observations encode the agent and goal cells themselves.

The relabelled segments are off-policy w.r.t. the observations they now carry,
so the PPO trainer re-anchors their behaviour log-probs at the current policy
and corrects with truncated importance weights (see `ppo/train.py`).

Modes:
  - 'random': one fresh layout from the task distribution per segment
    (unbiased augmentation of the hazard context).
  - 'optimistic': the lowest-cost of `num_candidates` fresh layouts per
    segment, i.e. hindsight placements where the realised gait was safe. This
    is what shows a conservative policy that long strides can pay off.
  - 'adaptive': 'optimistic' for a share `p` of the segments and 'random' for
    the rest, with `p` set by the constraint slack of the real batch (see
    `make_slack_gate`). Hindsight layouts bias the cost critic low, which helps
    a conservative policy but fights the multiplier once the cost exceeds the
    bound.
  - 'gated': like 'adaptive', but the non-optimistic segments keep their real
    layout instead of a random one, so a violating policy trains on real data
    only. On Reacher a random layout is pessimistic (the policy dodges its own
    layout, not a fresh one), and feeding it to a violating PPO-Lag policy
    drives the multiplier up without bound.
"""

import math
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from training import types

MODES = ('none', 'random', 'optimistic', 'adaptive', 'gated')
# Modes whose optimistic share comes from the slack gate
GATED_MODES = ('adaptive', 'gated')
CF_EXTRA_FIELDS = ('cf_kin', 'cf_feet', 'hazard_cost')
REACHER_CF_EXTRA_FIELDS = ('cf_theta',)
BUTTON_CF_EXTRA_FIELDS = ('cf_view', 'cf_first_view', 'cf_anchors', 'hazard_cost')
GRID_CF_EXTRA_FIELDS = ('cf_pos',)

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


def _optimistic_flags(mode: str, key: jax.Array, p_optimistic: jax.Array, num_segments: int) -> jax.Array:
    """Per-segment flags for keeping the lowest-cost layout."""
    if mode in GATED_MODES:
        return jax.random.bernoulli(key, p_optimistic, (num_segments,))
    return jnp.full((num_segments,), mode == 'optimistic')


def _keep_real(keep: jax.Array, relabelled: Any, real: Any) -> Any:
    """Per segment, the `real` arrays where `keep` and the `relabelled` ones elsewhere."""
    def select(x, r):
        return jnp.where(keep.reshape(keep.shape + (1,) * (x.ndim - 1)), r, x)

    return jax.tree_util.tree_map(select, relabelled, real)


def make_hazard_relabel_fn(
        environment: Any,
        mode: str = 'optimistic',
        num_candidates: int = 4,
) -> Callable[..., Tuple[types.Transition, Dict[str, jax.Array]]]:
    """Builds `relabel_fn(data, key, p_optimistic=1.0) -> (data, metrics)` for [B, T] pathway transitions.

    `p_optimistic` is the share of segments that keep the lowest-cost layout
    in 'adaptive' / 'gated' mode; the rest keep the first candidate, a random
    layout ('adaptive'), or their real layout ('gated').
    """
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
    num_candidates = 1 if mode == 'random' else num_candidates
    height = jnp.float32(env._hazard_height)

    def candidate(kin0_x, key):
        xs, ys = _sample_layout_x(env, key)
        # xs is sorted, so the hazards nearest the segment form a contiguous window
        first = jnp.clip(jnp.searchsorted(xs, kin0_x) - behind, 0, n_h - window)
        ids = first + jnp.arange(window)
        layout = jnp.stack([xs[ids], ys[ids], jnp.full((window,), height)], axis=-1)
        return layout, ids

    def relabel_segment(obs, next_obs, cost, hazard_cost, kin, feet, optimistic, key):
        # obs/next_obs [T, D], cost/hazard_cost [T], kin [T, 2, 3], feet [T, F, 3]
        keys = jax.random.split(key, num_candidates)
        layouts, ids = jax.vmap(candidate, in_axes=(None, 0))(kin[0, 0, 0], keys)
        step_cost = jax.vmap(
            lambda lay, hid: jax.vmap(env.hazard_cost_from_feet, in_axes=(0, None, None))(feet, lay, hid)
        )(layouts, ids)  # [K, T]
        # Candidates are i.i.d., so the first one is a random layout
        best = jnp.where(optimistic, jnp.argmin(step_cost.sum(axis=-1)), 0) if num_candidates > 1 else 0
        layout, new_hazard_cost = layouts[best], step_cost[best]

        lidar = jax.vmap(env.hazard_lidar, in_axes=(0, None))(kin[:, 0], layout)
        next_lidar = jax.vmap(env.hazard_lidar, in_axes=(0, None))(kin[:, 1], layout)
        obs = obs.at[:, -bins:].set(lidar)
        next_obs = next_obs.at[:, -bins:].set(next_lidar)
        # Terminal (unhealthy) cost is hazard-independent and carried over
        cost = cost - hazard_cost + new_hazard_cost
        return obs, next_obs, cost, new_hazard_cost

    def relabel_fn(data: types.Transition, key: jax.Array, p_optimistic: jax.Array = 1.0):
        if not isinstance(data.observation, jax.Array):
            raise ValueError("Counterfactual relabelling supports flat state observations only")
        se = data.extras['state_extras']
        num_segments = data.observation.shape[0]
        key_gate, key_seg = jax.random.split(key)
        optimistic = _optimistic_flags(mode, key_gate, p_optimistic, num_segments)
        keys = jax.random.split(key_seg, num_segments)
        obs, next_obs, cost, hazard_cost = jax.vmap(relabel_segment)(
            data.observation, data.next_observation, se['cost'], se['hazard_cost'],
            se['cf_kin'], se['cf_feet'], optimistic, keys)
        if mode == 'gated':
            obs, next_obs, cost, hazard_cost = _keep_real(
                ~optimistic, (obs, next_obs, cost, hazard_cost),
                (data.observation, data.next_observation, se['cost'], se['hazard_cost']))
        new_se = {**se, 'cost': cost, 'hazard_cost': hazard_cost}
        cf = data._replace(
            observation=obs, next_observation=next_obs,
            extras={**data.extras, 'state_extras': new_se})
        metrics = {
            'cf/real_hazard_cost': jnp.mean(se['hazard_cost']),
            'cf/hazard_cost': jnp.mean(hazard_cost),
            'cf/optimistic_share': jnp.mean(optimistic),
        }
        return cf, metrics

    return relabel_fn


def make_reacher_relabel_fn(
        environment: Any,
        mode: str = 'optimistic',
        num_candidates: int = 4,
) -> Callable[..., Tuple[types.Transition, Dict[str, jax.Array]]]:
    """`make_hazard_relabel_fn` for SafeReacher.

    Reacher layouts are placed around the goal, so a segment keeps its goal and
    gets a fresh layout around it (`sample_hazard_positions_batched`). A segment
    that crosses an episode boundary gets a second layout for the goal it ends
    with. The joint angles in the observation give the lidar heading, and
    `info['cf_theta']` (the post-step angles, which survive auto-resets) gives
    the cost; the reacher cost is entirely hazard cost.
    """
    env = environment.unwrapped
    for attr in ('hazard_lidar', 'hazard_cost_from_angles', 'sample_hazard_positions_batched'):
        if not hasattr(env, attr):
            raise ValueError(f"Reacher relabelling needs a SafeReacher env, got {type(env).__name__}")
    if mode not in MODES[1:]:
        raise ValueError(f"Unknown counterfactual mode '{mode}', expected one of {MODES[1:]}")
    bins = env._lidar_bins
    num_candidates = 1 if mode == 'random' else num_candidates
    # Observation: [cos(theta) (2), sin(theta) (2), target xy (2), ..., lidar (bins)]
    goal_slice = slice(4, 6)

    def heading(obs):
        return jnp.arctan2(obs[..., 2], obs[..., 0])

    def relabel_segment(obs, next_obs, theta, optimistic, key):
        # obs/next_obs [T, D], theta [T, 2]
        goal_first, goal_last = obs[0, goal_slice], next_obs[-1, goal_slice]

        def candidate(k):
            k_first, k_last = jax.random.split(k)
            return (env.sample_hazard_positions_batched(k_first, goal_first),
                    env.sample_hazard_positions_batched(k_last, goal_last))

        def layout_for(o, first, last):
            # Rows of the episode the segment starts in keep its goal
            same = jnp.all(jnp.abs(o[goal_slice] - goal_first) < 1e-6)
            return jnp.where(same, first, last)

        def segment_cost(first, last):
            layouts = jax.vmap(layout_for, in_axes=(0, None, None))(obs, first, last)
            return jax.vmap(env.hazard_cost_from_angles)(theta, layouts)  # [T]

        firsts, lasts = jax.vmap(candidate)(jax.random.split(key, num_candidates))
        step_cost = jax.vmap(segment_cost)(firsts, lasts)  # [K, T]
        # Candidates are i.i.d., so the first one is a random layout
        best = jnp.where(optimistic, jnp.argmin(step_cost.sum(axis=-1)), 0) if num_candidates > 1 else 0
        first, last = firsts[best], lasts[best]

        def lidar(o):
            return env.hazard_lidar(heading(o), layout_for(o, first, last))

        obs = obs.at[:, -bins:].set(jax.vmap(lidar)(obs))
        next_obs = next_obs.at[:, -bins:].set(jax.vmap(lidar)(next_obs))
        return obs, next_obs, step_cost[best]

    def relabel_fn(data: types.Transition, key: jax.Array, p_optimistic: jax.Array = 1.0):
        if not isinstance(data.observation, jax.Array):
            raise ValueError("Counterfactual relabelling supports flat state observations only")
        se = data.extras['state_extras']
        num_segments = data.observation.shape[0]
        key_gate, key_seg = jax.random.split(key)
        optimistic = _optimistic_flags(mode, key_gate, p_optimistic, num_segments)
        obs, next_obs, cost = jax.vmap(relabel_segment)(
            data.observation, data.next_observation, se['cf_theta'], optimistic,
            jax.random.split(key_seg, num_segments))
        if mode == 'gated':
            obs, next_obs, cost = _keep_real(
                ~optimistic, (obs, next_obs, cost), (data.observation, data.next_observation, se['cost']))
        cf = data._replace(
            observation=obs, next_observation=next_obs,
            extras={**data.extras, 'state_extras': {**se, 'cost': cost}})
        metrics = {
            'cf/real_hazard_cost': jnp.mean(se['cost']),
            'cf/hazard_cost': jnp.mean(cost),
            'cf/optimistic_share': jnp.mean(optimistic),
        }
        return cf, metrics

    return relabel_fn


def make_button_relabel_fn(
        environment: Any,
        mode: str = 'optimistic',
        num_candidates: int = 4,
) -> Callable[..., Tuple[types.Transition, Dict[str, jax.Array]]]:
    """`make_hazard_relabel_fn` for SafeButton.

    Only the floor hazards move: gremlins (collidable and moving), buttons and
    the wrong-button cost stay real, so the relabelled cost is the real cost
    with the floor-hazard part swapped. `info['cf_view']` holds the agent pose
    and kept-hazard positions behind the observation and the next observation,
    read from the pipeline state. An auto-reset restores the first state, so
    the next observation of a done step is rebuilt from `cf_first_view`.
    Layouts are fixed per env, so one layout serves a whole segment.
    """
    env = environment.unwrapped
    for attr in ('cf_hazard_obs', 'cf_hazard_cost', 'sample_cf_hazard_positions'):
        if not hasattr(env, attr):
            raise ValueError(f"Button relabelling needs a SafeButton env, got {type(env).__name__}")
    if mode not in MODES[1:]:
        raise ValueError(f"Unknown counterfactual mode '{mode}', expected one of {MODES[1:]}")
    tail = env.hazard_obs_size
    num_candidates = 1 if mode == 'random' else num_candidates
    hazard_obs = jax.vmap(env.cf_hazard_obs, in_axes=(0, None))
    hazard_cost = jax.vmap(env.cf_hazard_cost, in_axes=(0, None))

    def relabel_segment(obs, next_obs, done, view, first_view, anchors, optimistic, key):
        # obs/next_obs [T, D], done [T], view [T, 2, V], first_view [T, V], anchors [T, A, 2]
        layouts = jax.vmap(env.sample_cf_hazard_positions, in_axes=(0, None))(
            jax.random.split(key, num_candidates), anchors[0])  # [K, n_cf, 2]
        step_cost = jax.vmap(hazard_cost, in_axes=(None, 0))(view[:, 1, :2], layouts)  # [K, T]
        # Candidates are i.i.d., so the first one is a random layout
        best = jnp.where(optimistic, jnp.argmin(step_cost.sum(axis=-1)), 0) if num_candidates > 1 else 0
        layout = layouts[best]
        next_view = jnp.where(done[:, None], first_view, view[:, 1])
        obs = obs.at[:, -tail:].set(hazard_obs(view[:, 0], layout))
        next_obs = next_obs.at[:, -tail:].set(hazard_obs(next_view, layout))
        return obs, next_obs, step_cost[best]

    def relabel_fn(data: types.Transition, key: jax.Array, p_optimistic: jax.Array = 1.0):
        if not isinstance(data.observation, jax.Array):
            raise ValueError("Counterfactual relabelling supports flat state observations only")
        se = data.extras['state_extras']
        num_segments = data.observation.shape[0]
        key_gate, key_seg = jax.random.split(key)
        optimistic = _optimistic_flags(mode, key_gate, p_optimistic, num_segments)
        obs, next_obs, floor_cost = jax.vmap(relabel_segment)(
            data.observation, data.next_observation, data.discount == 0, se['cf_view'], se['cf_first_view'],
            se['cf_anchors'], optimistic, jax.random.split(key_seg, num_segments))
        if mode == 'gated':
            obs, next_obs, floor_cost = _keep_real(
                ~optimistic, (obs, next_obs, floor_cost), (data.observation, data.next_observation, se['hazard_cost']))
        cost = se['cost'] - se['hazard_cost'] + floor_cost
        cf = data._replace(
            observation=obs, next_observation=next_obs,
            extras={**data.extras, 'state_extras': {**se, 'cost': cost, 'hazard_cost': floor_cost}})
        metrics = {
            'cf/real_hazard_cost': jnp.mean(se['hazard_cost']),
            'cf/hazard_cost': jnp.mean(floor_cost),
            'cf/optimistic_share': jnp.mean(optimistic),
        }
        return cf, metrics

    return relabel_fn


def make_grid_relabel_fn(
        environment: Any,
        mode: str = 'optimistic',
        num_candidates: int = 4,
) -> Callable[..., Tuple[types.Transition, Dict[str, jax.Array]]]:
    """`make_hazard_relabel_fn` for SafeGrid.

    The observation holds the agent and goal cells, so every row (auto-reset
    ones included) is rebuilt from its own cells under a new lava map; the
    cost uses the post-step cell `info['cf_pos']`, and lava is the only cost.
    New lava stays off the segment's goal cells, as the env keeps it off goals.
    Layouts are fixed per env, so one lava map serves a whole segment.
    """
    env = environment.unwrapped
    for attr in ('sample_lava', 'obs_from', 'decode_cells'):
        if not hasattr(env, attr):
            raise ValueError(f"Grid relabelling needs a SafeGrid env, got {type(env).__name__}")
    if mode not in MODES[1:]:
        raise ValueError(f"Unknown counterfactual mode '{mode}', expected one of {MODES[1:]}")
    num_candidates = 1 if mode == 'random' else num_candidates
    size = env._size
    obs_from = jax.vmap(env.obs_from, in_axes=(0, 0, None))

    def relabel_segment(obs, next_obs, cf_pos, optimistic, key):
        # obs/next_obs [T, D], cf_pos [T, 2]
        pos, goal = env.decode_cells(obs)
        next_pos, next_goal = env.decode_cells(next_obs)
        goals = jnp.concatenate([goal, next_goal])
        keep_clear = jnp.zeros((size, size), dtype=bool).at[goals[:, 0], goals[:, 1]].set(True)
        lavas = jax.vmap(env.sample_lava, in_axes=(0, None))(
            jax.random.split(key, num_candidates), keep_clear)  # [K, size, size]
        step_cost = env._lava_cost * lavas[:, cf_pos[:, 0], cf_pos[:, 1]].astype(jnp.float32)  # [K, T]
        # Candidates are i.i.d., so the first one is a random layout
        best = jnp.where(optimistic, jnp.argmin(step_cost.sum(axis=-1)), 0) if num_candidates > 1 else 0
        lava = lavas[best]
        return obs_from(pos, goal, lava), obs_from(next_pos, next_goal, lava), step_cost[best]

    def relabel_fn(data: types.Transition, key: jax.Array, p_optimistic: jax.Array = 1.0):
        if not isinstance(data.observation, jax.Array):
            raise ValueError("Counterfactual relabelling supports flat state observations only")
        se = data.extras['state_extras']
        num_segments = data.observation.shape[0]
        key_gate, key_seg = jax.random.split(key)
        optimistic = _optimistic_flags(mode, key_gate, p_optimistic, num_segments)
        obs, next_obs, cost = jax.vmap(relabel_segment)(
            data.observation, data.next_observation, se['cf_pos'], optimistic,
            jax.random.split(key_seg, num_segments))
        if mode == 'gated':
            obs, next_obs, cost = _keep_real(
                ~optimistic, (obs, next_obs, cost), (data.observation, data.next_observation, se['cost']))
        cf = data._replace(
            observation=obs, next_observation=next_obs,
            extras={**data.extras, 'state_extras': {**se, 'cost': cost}})
        metrics = {
            'cf/real_hazard_cost': jnp.mean(se['cost']),
            'cf/hazard_cost': jnp.mean(cost),
            'cf/optimistic_share': jnp.mean(optimistic),
        }
        return cf, metrics

    return relabel_fn


def make_relabel_fn(environment: Any, mode: str, num_candidates: int) -> Tuple[Callable[..., Any], Tuple[str, ...]]:
    """The relabelling function for `environment` and the extra rollout fields it reads."""
    if hasattr(environment.unwrapped, 'hazard_cost_from_angles'):
        return make_reacher_relabel_fn(environment, mode, num_candidates), REACHER_CF_EXTRA_FIELDS
    if hasattr(environment.unwrapped, 'sample_lava'):
        return make_grid_relabel_fn(environment, mode, num_candidates), GRID_CF_EXTRA_FIELDS
    if hasattr(environment.unwrapped, 'sample_cf_hazard_positions'):
        return make_button_relabel_fn(environment, mode, num_candidates), BUTTON_CF_EXTRA_FIELDS
    return make_hazard_relabel_fn(environment, mode, num_candidates), CF_EXTRA_FIELDS


def make_slack_gate(
        step_bound_fn: Callable[[Any], jax.Array],
        sharpness: float,
) -> Callable[[Any, jax.Array], jax.Array]:
    """Share of optimistic segments from the constraint slack of the real batch.

    `p = sigmoid(-sharpness * (cost - bound) / bound)` with the per-step cost
    and bound: mostly optimistic while the policy is under the bound (the
    multiplier is falling, or clamped at 0), mostly random ('adaptive') or real
    ('gated') once it violates.
    This is the sign of the Lagrangian update, without its dead zone at 0.
    """
    def gate_fn(aux_state, mean_cost):
        step_bound = step_bound_fn(aux_state)
        return jax.nn.sigmoid(-sharpness * (mean_cost - step_bound) / step_bound)

    return gate_fn


def counterfactual_train_kwargs(
        environment: Any,
        extra_fields: Tuple[str, ...],
        action_repeat: int,
        counterfactual_mode: str,
        counterfactual_ratio: float,
        counterfactual_candidates: int,
        counterfactual_rho_clip: float,
        counterfactual_gate_sharpness: float = 10.0,
        step_bound_fn: Optional[Callable[[Any], jax.Array]] = None,
) -> Dict[str, Any]:
    """Kwargs for `ppo_train.train` that enable counterfactual relabelling (empty if disabled).

    `step_bound_fn(aux_state)` returns the per-step cost bound; 'adaptive' and 'gated' modes need it.
    """
    if counterfactual_mode in (None, 'none') or counterfactual_ratio <= 0:
        return {'extra_fields': extra_fields}
    if action_repeat != 1:
        raise ValueError("Counterfactual relabelling requires action_repeat=1: "
                         "the stored kinematics only hold the last physics sub-step")
    relabel_fn, cf_fields = make_relabel_fn(environment, counterfactual_mode, counterfactual_candidates)
    kwargs = {
        'extra_fields': tuple(extra_fields) + cf_fields,
        'counterfactual_fn': relabel_fn,
        'counterfactual_ratio': counterfactual_ratio,
        'counterfactual_rho_clip': counterfactual_rho_clip,
    }
    if counterfactual_mode in GATED_MODES:
        if step_bound_fn is None:
            raise ValueError(f"'{counterfactual_mode}' counterfactual relabelling needs the per-step cost bound")
        kwargs['counterfactual_gate_fn'] = make_slack_gate(step_bound_fn, counterfactual_gate_sharpness)
    return kwargs
