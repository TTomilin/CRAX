"""SafeGridMomentum: a SafeGrid where repeated moves build up speed.

Moving in the same direction as the previous step moves one cell further than
that step did, up to `max_speed` cells; changing direction drops back to one
cell and staying put stops. Cells passed over are skipped: lava costs only on
the cell the agent lands on, and a goal is collected only by landing on it.
A fast agent therefore clears thin lava strips at no cost if it times its
landing, while a slow one has to walk around them. Speed trades control for
progress: at speed the agent cannot move a single cell without breaking its
run, so a careless sprint lands in lava.

The grid is an open room with lava in thin straight strips or, with
`num_rivers`, in vertical rivers whose gaps alternate between the top and the
bottom, so walking across means a long detour. A small `view_radius` hides
what lies beyond the first lava cell, so a leap is taken without knowing
where it lands. With `safe_goals`
(default) the agent and goals spawn in the largest lava-free region, so every
goal can also be reached by walking without cost.

The observation extends SafeGrid's with the current heading (one-hot) and
speed. Neither depends on lava, so `relabel_obs` keeps them and rebuilds only
the lava window under a counterfactual layout.
"""

from typing import Optional

import jax
from flax import struct
from jax import numpy as jp
import numpy as np

from crax.envs.base import State
from crax.envs.safe_grid import SafeGrid, _MOVES


@struct.dataclass
class MomentumState:
  pos: jax.Array      # (2,) int32 agent cell (x, y)
  goal: jax.Array     # (2,) int32 goal cell
  lava: jax.Array     # (size, size) bool, indexed [x, y]
  spawn: jax.Array    # (size, size) bool, cells the agent and goals may spawn on
  rng: jax.Array      # goal respawns
  heading: jax.Array  # () int32 index into _MOVES of the last move, 0 when standing
  speed: jax.Array    # () int32 cells covered by the last move


class SafeGridMomentum(SafeGrid):
  """Goal collection with momentum: repeated moves go further and skip over lava."""

  def __init__(
      self,
      size: int = 25,
      max_speed: int = 2,
      num_strips: int = 14,
      strip_length: tuple = (3, 9),
      strip_width: int = 1,
      num_rivers: int = 0,
      river_width: int = 2,
      river_gap: int = 2,
      river_shift: int = 1,
      episode_length: int = 300,
      **kwargs,
  ):
    super().__init__(layout='open', size=size, episode_length=episode_length, **kwargs)
    self._max_speed = max_speed
    self._num_strips = num_strips
    self._strip_length = strip_length
    self._strip_width = strip_width
    self._num_rivers = num_rivers
    self._river_width = river_width
    self._river_gap = river_gap
    self._river_shift = river_shift
    strip = (size - 2 - num_rivers * river_width) / (num_rivers + 1)
    if num_rivers and strip < 2 * river_shift + 1:
      raise ValueError(f"{num_rivers} rivers of width {river_width} shifted by up to {river_shift} "
                       f"do not fit a grid of size {size}")
    self._river_x = np.array([1 + round((b + 1) * strip) + b * river_width for b in range(num_rivers)])

  # ------------------------------------------------------------------
  # Layout
  # ------------------------------------------------------------------

  def sample_lava(self, key: jax.Array, keep_clear: Optional[jax.Array] = None) -> jax.Array:
    """Lava map [size, size]: rivers if `num_rivers` is set, else straight strips.

    `keep_clear` is an optional [size, size] mask of cells that stay lava-free.
    """
    lava = self._rivers(key) if self._num_rivers else self._strips(key)
    if keep_clear is not None:
      lava &= ~keep_clear
    return lava

  def _rivers(self, key: jax.Array) -> jax.Array:
    """Vertical rivers, shifted by up to `river_shift` cells, each with a gap at its top or bottom end.

    The gaps alternate between ends, so walking from one side of the grid to the other snakes
    through every gap, while a running jump clears a river anywhere.
    """
    k_parity, k_shift = jax.random.split(key)
    parity = jax.random.randint(k_parity, (), 0, 2)
    x0 = jp.asarray(self._river_x) + jax.random.randint(
        k_shift, (self._num_rivers,), -self._river_shift, self._river_shift + 1)
    gap_top = (jp.arange(self._num_rivers) + parity) % 2 == 0
    x, y = self._cells[..., 0, None], self._cells[..., 1, None]  # [size, size, 1]
    in_river = (x >= x0) & (x < x0 + self._river_width)
    in_gap = jp.where(gap_top, y > self._size - 2 - self._river_gap, y < 1 + self._river_gap)
    return jp.any(in_river & ~in_gap, axis=-1) & self._free

  def _strips(self, key: jax.Array) -> jax.Array:
    """Straight strips of random length and orientation."""
    k_center, k_vertical, k_length = jax.random.split(key, 3)
    idx = jax.random.randint(k_center, (self._num_strips,), 0, self._interior.shape[0])
    centers = self._interior[idx]  # [S, 2]
    vertical = jax.random.bernoulli(k_vertical, 0.5, (self._num_strips,))
    half = jax.random.randint(k_length, (self._num_strips,), self._strip_length[0],
                              self._strip_length[1] + 1) / 2.0
    d = self._cells[:, :, None, :] - centers[None, None]  # [size, size, S, 2]
    along = jp.where(vertical, d[..., 1], d[..., 0])
    across = jp.where(vertical, d[..., 0], d[..., 1])
    strips = (jp.abs(along) < half) & (across >= 0) & (across < self._strip_width)
    return jp.any(strips, axis=-1) & self._free

  # ------------------------------------------------------------------
  # Env API
  # ------------------------------------------------------------------

  def reset(self, rng: jax.Array) -> State:
    state = super().reset(rng)
    grid = state.pipeline_state
    zero = jp.zeros((), dtype=jp.int32)
    grid = MomentumState(pos=grid.pos, goal=grid.goal, lava=grid.lava, spawn=grid.spawn, rng=grid.rng,
                         heading=zero, speed=zero)
    return state.replace(pipeline_state=grid, obs=self._obs(grid))

  def step(self, state: State, action: jax.Array) -> State:
    grid = state.pipeline_state
    idx = self.action_to_move(action)
    run = (idx == grid.heading) & (idx != 0)
    speed = jp.where(idx == 0, 0, jp.where(run, jp.minimum(grid.speed + 1, self._max_speed), 1))
    # The room is open, so only the border can cut a move short
    pos = jp.clip(grid.pos + jp.asarray(_MOVES)[idx] * speed, 1, self._size - 2)
    speed = jp.sum(jp.abs(pos - grid.pos)).astype(jp.int32)

    reached = jp.all(pos == grid.goal)
    rng, k_goal = jax.random.split(grid.rng)
    new_goal = self._sample_free_cell(k_goal, ~grid.spawn | self._one_hot(pos))
    goal = jp.where(reached, new_goal, grid.goal)

    dist_before = jp.linalg.norm((grid.goal - grid.pos).astype(jp.float32))
    dist_after = jp.linalg.norm((grid.goal - pos).astype(jp.float32))
    reward = self._goal_reward * reached + self._distance_reward * (dist_before - dist_after)
    on_lava = self._at(grid.lava, pos).astype(jp.float32)
    cost = self._lava_cost * on_lava

    grid = grid.replace(pos=pos, goal=goal, rng=rng, heading=jp.where(speed > 0, idx, 0), speed=speed)
    metrics = {**state.metrics, 'reward': reward, 'cost': cost, 'goals': reached.astype(jp.float32),
               'on_lava': on_lava}
    info = {**state.info, 'cost': cost, 'cf_pos': pos}
    return state.replace(pipeline_state=grid, obs=self._obs(grid), reward=reward, done=jp.zeros(()),
                         metrics=metrics, info=info)

  # ------------------------------------------------------------------
  # Observation
  # ------------------------------------------------------------------

  def _obs(self, grid: MomentumState) -> jax.Array:
    heading = jax.nn.one_hot(grid.heading - 1, 4)  # all zero when standing
    motion = jp.concatenate([heading, (grid.speed / self._max_speed)[None]])
    return jp.concatenate([self.obs_from(grid.pos, grid.goal, grid.lava), motion])

  def relabel_obs(self, obs: jax.Array, lava: jax.Array) -> jax.Array:
    return jp.concatenate([super().relabel_obs(obs, lava), obs[..., -5:]])

  @property
  def observation_size(self) -> int:
    return super().observation_size + 5
