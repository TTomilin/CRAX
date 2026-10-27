"""SafeGridDetour: a SafeGrid where the cost-free route is a long detour.

Vertical bands split a large grid into strips, and the agent and goals spawn
only in the strips. Each band can be crossed in three ways:
  - a safe crossing at one end of the band, free of lava. The safe ends
    alternate between bands (top, bottom, top, ...), so the cost-free route
    between distant strips snakes through the whole grid;
  - a shortcut in the middle of the band, `shallow_depth` lava cells deep;
  - anywhere else, through the full band width of lava.
With `doors` set, the bands are walls with that many fixed doorways and
crossings only happen at doorways; without it every row of a band is a lava
river that can be crossed anywhere. Lava only costs, so a policy that never
touches it is always feasible but slow, while a few cheap steps at the right
shortcut cut the route several-fold.

The observation extends SafeGrid's with the lava depth of every band at every
crossing row, so it stays a function of the agent cell, the goal cell and the
lava map and counterfactual relabelling replays it exactly.
"""

from typing import Optional

import jax
from jax import numpy as jp
import numpy as np

from crax.envs.safe_grid import SafeGrid


class SafeGridDetour(SafeGrid):
  """Goal collection where avoiding all lava means long detours."""

  def __init__(
      self,
      size: int = 31,
      num_bands: int = 3,
      band_width: int = 3,
      doors: Optional[int] = 5,
      num_shortcuts: int = 1,
      shallow_depth: int = 1,
      shortcut_margin: float = 0.25,
      episode_length: int = 500,
      **kwargs,
  ):
    super().__init__(layout='open', size=size, episode_length=episode_length, **kwargs)
    interior = size - 2
    strip = (interior - num_bands * band_width) / (num_bands + 1)
    if strip < 1:
      raise ValueError(f"{num_bands} bands of width {band_width} do not fit a grid of size {size}")
    self._band_x = np.array([1 + int(round((b + 1) * strip)) + b * band_width for b in range(num_bands)])
    rows = np.arange(1, size - 1) if doors is None else np.round(np.linspace(1, size - 2, doors)).astype(int)
    if len(rows) < 3:
      raise ValueError("Bands need at least three crossing rows: two ends and a shortcut")
    self._rows = rows
    self._band_width = band_width
    self._num_shortcuts = num_shortcuts
    self._shallow_depth = shallow_depth
    # Shortcuts sit strictly between the ends, at least `shortcut_margin` of the band away from both
    last = len(rows) - 1
    self._shortcut_lo = max(1, int(np.ceil(shortcut_margin * last)))
    self._shortcut_hi = min(last - 1, int(np.floor((1 - shortcut_margin) * last)))
    if self._shortcut_hi - self._shortcut_lo + 1 < num_shortcuts:
      raise ValueError(f"Too few rows for {num_shortcuts} shortcuts with margin {shortcut_margin}")

    band_cols = np.zeros((size, size), dtype=bool)
    for x0 in self._band_x:
      band_cols[x0:x0 + band_width, 1:-1] = True
    walls = self._walls_np.copy()
    if doors is not None:
      walls |= band_cols
      walls[:, rows] &= ~band_cols[:, rows]
    self._walls_np = walls
    self._walls = jp.asarray(walls)
    self._free = ~self._walls
    self._interior = jp.asarray(np.argwhere(~walls), dtype=jp.float32)
    self._strips = jp.asarray(~walls & ~band_cols)
    # Lava fills a crossing from the band's centre column outwards
    offset = np.abs(np.arange(band_width) - (band_width - 1) / 2)
    self._fill_rank = jp.asarray(np.argsort(np.argsort(offset, kind='stable')))

  # ------------------------------------------------------------------
  # Layout
  # ------------------------------------------------------------------

  def crossing_depths(self, key: jax.Array) -> jax.Array:
    """Lava depth [num_bands, num_rows] of every crossing for a new layout."""
    num_bands, num_rows = len(self._band_x), len(self._rows)
    k_parity, k_short = jax.random.split(key)
    parity = jax.random.randint(k_parity, (), 0, 2)
    safe_top = (jp.arange(num_bands) + parity) % 2 == 0
    safe_row = jp.where(safe_top, num_rows - 1, 0)
    num_candidates = self._shortcut_hi - self._shortcut_lo + 1
    shortcut = jax.vmap(
        lambda k: self._shortcut_lo + jax.random.permutation(k, num_candidates)[:self._num_shortcuts])(
        jax.random.split(k_short, num_bands))  # [num_bands, num_shortcuts]
    row = jp.arange(num_rows)
    is_short = jp.any(row[None, :, None] == shortcut[:, None, :], axis=-1)
    depth = jp.where(is_short, self._shallow_depth, self._band_width)
    return jp.where(row[None] == safe_row[:, None], 0, depth)

  def lava_from_depths(self, depths: jax.Array) -> jax.Array:
    """Lava map [size, size] for crossing depths [num_bands, num_rows]."""
    lava = jp.zeros((self._size, self._size), dtype=bool)
    cols = jp.arange(self._band_width)
    for b, x0 in enumerate(self._band_x):
      block = self._fill_rank[:, None] < depths[b][None, :]  # [band_width, num_rows]
      lava = lava.at[x0 + cols[:, None], self._rows[None, :]].set(block)
    return lava

  def sample_lava(self, key: jax.Array, keep_clear: Optional[jax.Array] = None) -> jax.Array:
    lava = self.lava_from_depths(self.crossing_depths(key))
    if keep_clear is not None:
      lava &= ~keep_clear
    return lava

  def spawn_mask(self, lava: jax.Array) -> jax.Array:
    return self._strips & ~lava

  # ------------------------------------------------------------------
  # Observation
  # ------------------------------------------------------------------

  def obs_from(self, pos: jax.Array, goal: jax.Array, lava: jax.Array) -> jax.Array:
    cols = self._band_x[:, None] + np.arange(self._band_width)[None]  # [num_bands, band_width]
    depth = lava[cols][:, :, self._rows].sum(axis=1) / self._band_width  # [num_bands, num_rows]
    return jp.concatenate([super().obs_from(pos, goal, lava), depth.reshape(-1).astype(jp.float32)])

  @property
  def observation_size(self) -> int:
    return super().observation_size + len(self._band_x) * len(self._rows)
