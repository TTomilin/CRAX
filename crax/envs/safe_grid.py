"""SafeGrid: a JAX gridworld with lava for constrained RL.

The agent walks a walled grid and collects goals: each goal reached is
rewarded and respawns elsewhere, with a dense shaping reward for getting
closer. Lava cells cost 1 per step spent on them but do not block, kill or
otherwise change the dynamics, so a shortcut through lava trades cost for
reward. Walls are fixed per level (an open room or four rooms joined by
doorways); lava comes in random blobs, which can block doorways and force the
agent to choose between crossing lava and staying in its room.

Observation (all in [0, 1]):
  - agent cell and goal cell, normalised by `size - 1` (2 + 2)
  - a (2r+1)x(2r+1) window around the agent with wall, lava and goal channels

Everything in the observation is a function of the agent cell, the goal cell
and the lava map, which is what counterfactual relabelling relies on
(`training/counterfactual.py`): a recorded trajectory can be replayed exactly
under another lava map.

Actions are continuous (2,) in [-1, 1]: the larger component picks the axis
and its sign the direction; if both are below `dead_zone` the agent stays.
"""

from typing import List, Optional, Sequence, Tuple

import jax
from flax import struct
from jax import numpy as jp
import numpy as np

from crax.envs.base import Env, State

# Four rooms joined by one doorway per wall; '#' wall, '.' floor
_FOUR_ROOMS_13 = """
#############
#.....#.....#
#.....#.....#
#...........#
#.....#.....#
#.....#.....#
###.#####.###
#.....#.....#
#.....#.....#
#...........#
#.....#.....#
#.....#.....#
#############
"""

_FOUR_ROOMS_15 = """
###############
#......#......#
#......#......#
#......#......#
#.............#
#......#......#
#......#......#
####.#####.####
#......#......#
#......#......#
#.............#
#......#......#
#......#......#
#......#......#
###############
"""

LAYOUTS = {
    'four_rooms_13': _FOUR_ROOMS_13,
    'four_rooms_15': _FOUR_ROOMS_15,
}

# Move per action index: stay, +x, -x, +y, -y
_MOVES = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int32)


@struct.dataclass
class GridState:
  pos: jax.Array    # (2,) int32 agent cell (x, y)
  goal: jax.Array   # (2,) int32 goal cell
  lava: jax.Array   # (size, size) bool, indexed [x, y]
  rng: jax.Array    # goal respawns


def parse_layout(layout: str) -> np.ndarray:
  """Wall map [x, y] from rows of '#'/'.' (first row is the top, y = size - 1)."""
  rows = [r for r in layout.strip().splitlines()]
  if any(len(r) != len(rows) for r in rows):
    raise ValueError("Layouts must be square")
  return np.array([[c == '#' for c in r] for r in rows[::-1]], dtype=bool).T


def open_room(size: int) -> np.ndarray:
  walls = np.zeros((size, size), dtype=bool)
  walls[[0, -1], :] = True
  walls[:, [0, -1]] = True
  return walls


class SafeGrid(Env):
  """Goal collection in a walled gridworld with cost-only lava."""

  def __init__(
      self,
      layout: str = 'open',
      size: int = 11,
      num_lava_blobs: int = 4,
      lava_radius: float = 1.0,
      view_radius: int = 3,
      episode_length: int = 200,
      goal_reward: float = 1.0,
      distance_reward: float = 0.1,
      lava_cost: float = 1.0,
      dead_zone: float = 0.25,
      **kwargs,
  ):
    walls = open_room(size) if layout == 'open' else parse_layout(LAYOUTS.get(layout, layout))
    self._walls_np = walls
    self._size = walls.shape[0]
    self._walls = jp.asarray(walls)
    self._free = ~self._walls
    self._num_lava_blobs = num_lava_blobs
    self._lava_radius = lava_radius
    self._view_radius = view_radius
    self.episode_length = episode_length
    self._goal_reward = goal_reward
    self._distance_reward = distance_reward
    self._lava_cost = lava_cost
    self._dead_zone = dead_zone
    xs, ys = np.meshgrid(np.arange(self._size), np.arange(self._size), indexing='ij')
    self._cells = jp.asarray(np.stack([xs, ys], axis=-1), dtype=jp.float32)  # [size, size, 2]
    self._interior = jp.asarray(np.argwhere(self._free_np()), dtype=jp.float32)

  def _free_np(self) -> np.ndarray:
    return ~self._walls_np

  # ------------------------------------------------------------------
  # Layout
  # ------------------------------------------------------------------

  def sample_lava(self, key: jax.Array, keep_clear: Optional[jax.Array] = None) -> jax.Array:
    """Lava map [size, size]: discs of `lava_radius` around random floor cells.

    `keep_clear` is an optional [size, size] mask of cells that stay lava-free.
    """
    idx = jax.random.randint(key, (self._num_lava_blobs,), 0, self._interior.shape[0])
    centers = self._interior[idx]  # [B, 2]
    d = jp.linalg.norm(self._cells[:, :, None, :] - centers[None, None], axis=-1)
    lava = jp.any(d <= self._lava_radius, axis=-1) & self._free
    if keep_clear is not None:
      lava &= ~keep_clear
    return lava

  def _sample_free_cell(self, key: jax.Array, blocked: jax.Array) -> jax.Array:
    """A uniformly random cell that is neither wall nor `blocked`."""
    ok = (self._free & ~blocked).reshape(-1)
    flat = jax.random.choice(key, ok.shape[0], p=ok / jp.sum(ok))
    return jp.array([flat // self._size, flat % self._size], dtype=jp.int32)

  def _at(self, grid: jax.Array, cell: jax.Array) -> jax.Array:
    return grid[cell[0], cell[1]]

  # ------------------------------------------------------------------
  # Env API
  # ------------------------------------------------------------------

  def reset(self, rng: jax.Array) -> State:
    rng, k_lava, k_pos, k_goal = jax.random.split(rng, 4)
    lava = self.sample_lava(k_lava)
    pos = self._sample_free_cell(k_pos, lava)
    goal = self._sample_free_cell(k_goal, lava | self._one_hot(pos))
    grid = GridState(pos=pos, goal=goal, lava=lava, rng=rng)
    obs = self.obs_from(pos, goal, lava)
    zero = jp.zeros(())
    metrics = {'reward': zero, 'cost': zero, 'goals': zero, 'on_lava': zero}
    info = {'cost': zero, 'cf_pos': pos}
    return State(grid, obs, zero, zero, metrics, info)

  def step(self, state: State, action: jax.Array) -> State:
    grid = state.pipeline_state
    move = jp.asarray(_MOVES)[self.action_to_move(action)]
    target = grid.pos + move
    pos = jp.where(self._at(self._walls, target), grid.pos, target)

    reached = jp.all(pos == grid.goal)
    rng, k_goal = jax.random.split(grid.rng)
    new_goal = self._sample_free_cell(k_goal, grid.lava | self._one_hot(pos))
    goal = jp.where(reached, new_goal, grid.goal)

    dist_before = jp.linalg.norm((grid.goal - grid.pos).astype(jp.float32))
    dist_after = jp.linalg.norm((grid.goal - pos).astype(jp.float32))
    reward = self._goal_reward * reached + self._distance_reward * (dist_before - dist_after)
    on_lava = self._at(grid.lava, pos).astype(jp.float32)
    cost = self._lava_cost * on_lava

    grid = grid.replace(pos=pos, goal=goal, rng=rng)
    obs = self.obs_from(pos, goal, grid.lava)
    metrics = {**state.metrics, 'reward': reward, 'cost': cost, 'goals': reached.astype(jp.float32),
               'on_lava': on_lava}
    info = {**state.info, 'cost': cost, 'cf_pos': pos}
    return state.replace(pipeline_state=grid, obs=obs, reward=reward, done=jp.zeros(()), metrics=metrics,
                         info=info)

  def action_to_move(self, action: jax.Array) -> jax.Array:
    """Index into _MOVES: dominant axis and its sign, or stay inside the dead zone."""
    axis = jp.argmax(jp.abs(action[:2]))
    value = action[axis]
    idx = 1 + 2 * axis + (value < 0)
    return jp.where(jp.max(jp.abs(action[:2])) < self._dead_zone, 0, idx)

  def _one_hot(self, cell: jax.Array) -> jax.Array:
    return jp.zeros((self._size, self._size), dtype=bool).at[cell[0], cell[1]].set(True)

  # ------------------------------------------------------------------
  # Observation
  # ------------------------------------------------------------------

  def obs_from(self, pos: jax.Array, goal: jax.Array, lava: jax.Array) -> jax.Array:
    """Observation for an agent at `pos` with `goal` and the lava map `lava`."""
    r = self._view_radius
    scale = self._size - 1
    walls = jp.pad(self._walls, r, constant_values=True)
    lava = jp.pad(lava, r, constant_values=False)
    goal_map = jp.pad(self._one_hot(goal), r, constant_values=False)
    w = 2 * r + 1

    def window(grid):
      return jax.lax.dynamic_slice(grid, (pos[0], pos[1]), (w, w)).astype(jp.float32).reshape(-1)

    return jp.concatenate([
        pos.astype(jp.float32) / scale,
        goal.astype(jp.float32) / scale,
        window(walls), window(lava), window(goal_map),
    ])

  def decode_cells(self, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """(agent cell, goal cell) from an observation."""
    cells = jp.round(obs[..., :4] * (self._size - 1)).astype(jp.int32)
    return cells[..., :2], cells[..., 2:4]

  @property
  def observation_size(self) -> int:
    return 4 + 3 * (2 * self._view_radius + 1) ** 2

  @property
  def action_size(self) -> int:
    return 2

  @property
  def backend(self) -> str:
    return 'grid'

  # ------------------------------------------------------------------
  # Rendering
  # ------------------------------------------------------------------

  def render(
      self,
      trajectory: List[GridState],
      height: int = 480,
      width: int = 480,
      camera: Optional[str] = None,
  ) -> Sequence[np.ndarray]:
    from crax.envs.grid_render import render_grid_frames
    return render_grid_frames(self._walls_np, trajectory, height=height, width=width)
