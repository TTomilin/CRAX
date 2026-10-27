"""Frames for SafeGrid trajectories, drawn with PIL.

Warm floor tiles, bevelled slate walls, glowing lava with drifting bubbles, a
star-marked goal, and the agent with a fading trail. A bar on top counts steps,
goals and accumulated lava cost. Drawn at 2x and downsampled for smooth edges.
"""

import math
from typing import List, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_SUPERSAMPLE = 2
_TRAIL = 12

_BG = (31, 36, 48)
_FLOOR = ((243, 237, 224), (236, 228, 211))
_GRID_LINE = (217, 207, 186)
_WALL = (59, 66, 82)
_WALL_LIGHT = (76, 86, 106)
_WALL_DARK = (46, 52, 64)
_LAVA = ((201, 42, 42), (232, 89, 12), (247, 103, 7), (255, 169, 77))
_BUBBLE = (255, 224, 102)
_GOAL_RING = (47, 158, 68)
_GOAL_FILL = (81, 207, 102)
_AGENT = (28, 126, 214)
_AGENT_HOT = (250, 82, 82)
_TRAIL_RGB = (77, 171, 247)
_TEXT = (233, 236, 239)
_TEXT_COST = (255, 146, 43)


def _font(size: int):
  try:
    return ImageFont.load_default(size=size)
  except TypeError:  # Pillow < 10.1
    return ImageFont.load_default()


def _star(cx: float, cy: float, r: float) -> List[float]:
  pts = []
  for i in range(10):
    a = -math.pi / 2 + i * math.pi / 5
    rr = r if i % 2 == 0 else r * 0.45
    pts += [cx + rr * math.cos(a), cy + rr * math.sin(a)]
  return pts


def _static_layer(walls: np.ndarray, lava: np.ndarray, cell: int, frame: int) -> Image.Image:
  size = walls.shape[0]
  img = Image.new('RGB', (size * cell, size * cell), _BG)
  d = ImageDraw.Draw(img)

  def box(x, y, inset=0.0):
    # Grid x to the right, y up
    x0 = x * cell + inset * cell
    y0 = (size - 1 - y) * cell + inset * cell
    return [x0, y0, x0 + cell * (1 - 2 * inset), y0 + cell * (1 - 2 * inset)]

  for x in range(size):
    for y in range(size):
      if walls[x, y]:
        continue
      d.rectangle(box(x, y), fill=_FLOOR[(x + y) % 2], outline=_GRID_LINE, width=max(1, cell // 40))

  # Lava: a solid base so neighbouring cells merge into pools, then hot cores and bubbles
  for x in range(size):
    for y in range(size):
      if not lava[x, y]:
        continue
      d.rectangle(box(x, y), fill=_LAVA[0])
  for x in range(size):
    for y in range(size):
      if not lava[x, y]:
        continue
      for level, inset in enumerate((0.06, 0.18, 0.32)):
        d.rounded_rectangle(box(x, y, inset), radius=cell * (0.3 - inset * 0.5), fill=_LAVA[level + 1])
      for b in range(2):
        seed = (x * 73856093) ^ (y * 19349663) ^ (b * 83492791)
        phase = (seed % 997) / 997 * 2 * math.pi + frame * 0.35
        bx = box(x, y)[0] + cell * (0.25 + 0.5 * ((seed >> 3) % 101) / 100)
        by = box(x, y)[1] + cell * (0.5 + 0.28 * math.sin(phase))
        br = cell * (0.05 + 0.03 * (1 + math.sin(phase * 1.3)))
        d.ellipse([bx - br, by - br, bx + br, by + br], fill=_BUBBLE)

  bevel = max(2, cell // 10)
  for x in range(size):
    for y in range(size):
      if not walls[x, y]:
        continue
      x0, y0, x1, y1 = box(x, y)
      d.rectangle([x0, y0, x1, y1], fill=_WALL_DARK)
      d.rectangle([x0, y0, x1 - bevel, y1 - bevel], fill=_WALL_LIGHT)
      d.rectangle([x0 + bevel, y0 + bevel, x1 - bevel, y1 - bevel], fill=_WALL)
  return img


def render_grid_frames(walls: np.ndarray, trajectory: Sequence, height: int = 480,
                       width: int = 480) -> List[np.ndarray]:
  """One RGB frame [height, width, 3] per GridState in `trajectory` (unbatched)."""
  size = walls.shape[0]
  s = _SUPERSAMPLE
  hud = max(28, height // 14)
  cell = max(4, min(width, height - hud) // size) * s
  board = size * cell
  canvas_w, canvas_h = width * s, height * s
  font = _font(int(hud * s * 0.5))

  positions = [np.asarray(state.pos) for state in trajectory]
  goals = [np.asarray(state.goal) for state in trajectory]
  frames = []
  cost = 0.0
  reached = 0
  for t, state in enumerate(trajectory):
    lava = np.asarray(state.lava)
    pos, goal = positions[t], goals[t]
    if t > 0:
      cost += float(lava[pos[0], pos[1]])
      reached += int(np.any(goals[t] != goals[t - 1]))

    img = Image.new('RGB', (canvas_w, canvas_h), _BG)
    ox = (canvas_w - board) // 2
    oy = hud * s + (canvas_h - hud * s - board) // 2
    img.paste(_static_layer(walls, lava, cell, t), (ox, oy))

    def center(c):
      return ox + (c[0] + 0.5) * cell, oy + (size - 1 - c[1] + 0.5) * cell

    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    # Goal: soft halo, ring and star
    gx, gy = center(goal)
    for k, alpha in ((0.62, 50), (0.5, 90)):
      od.ellipse([gx - k * cell, gy - k * cell, gx + k * cell, gy + k * cell], fill=_GOAL_FILL + (alpha,))
    # Trail of recent cells, fading with age
    trail = positions[max(0, t - _TRAIL):t]
    for age, c in enumerate(reversed(trail), start=1):
      if np.all(c == pos):
        continue
      cx, cy = center(c)
      r = cell * (0.22 - 0.012 * age)
      od.ellipse([cx - r, cy - r, cx + r, cy + r], fill=_TRAIL_RGB + (int(170 * (1 - age / (_TRAIL + 1))),))
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')

    d = ImageDraw.Draw(img)
    r = cell * 0.36
    d.ellipse([gx - r, gy - r, gx + r, gy + r], fill=_GOAL_RING)
    r = cell * 0.29
    d.ellipse([gx - r, gy - r, gx + r, gy + r], fill=_GOAL_FILL)
    d.polygon(_star(gx, gy, cell * 0.2), fill=(255, 255, 255))

    # Agent, outlined red while standing in lava, with a chevron towards its last move
    ax, ay = center(pos)
    hot = bool(lava[pos[0], pos[1]])
    r = cell * 0.34
    d.ellipse([ax - r - cell * 0.06, ay - r - cell * 0.06, ax + r + cell * 0.06, ay + r + cell * 0.06],
              fill=_AGENT_HOT if hot else (255, 255, 255))
    d.ellipse([ax - r, ay - r, ax + r, ay + r], fill=_AGENT)
    if t > 0 and np.any(pos != positions[t - 1]):
      dx, dy = (pos - positions[t - 1]).astype(float)
      ux, uy = dx, -dy  # image y points down
      px, py = -uy, ux
      tip = (ax + ux * r * 0.75, ay + uy * r * 0.75)
      base = (ax - ux * r * 0.2, ay - uy * r * 0.2)
      d.polygon([tip, (base[0] + px * r * 0.5, base[1] + py * r * 0.5),
                 (base[0] - px * r * 0.5, base[1] - py * r * 0.5)], fill=(255, 255, 255))
    else:
      rr = r * 0.25
      d.ellipse([ax - rr, ay - rr, ax + rr, ay + rr], fill=(255, 255, 255))

    # HUD
    pad = hud * s * 0.25
    d.text((pad, pad), f"step {t}", font=font, fill=_TEXT)
    d.text((canvas_w * 0.38, pad), f"goals {reached}", font=font, fill=_TEXT)
    d.text((canvas_w * 0.68, pad), f"cost {cost:.0f}", font=font, fill=_TEXT_COST if cost else _TEXT)

    frames.append(np.asarray(img.resize((width, height), Image.LANCZOS)))
  return frames
