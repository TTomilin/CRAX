"""
Centralized difficulty mapping for safety environments.

This module defines a small, extensible system to translate a difficulty
level (1, 2, 3) into environment-specific parameter overrides.

It is intentionally lightweight and modular: add new env handlers or tweak
mappings in a single place without touching training code or env classes.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


_DIFFICULTY_OVERRIDES: dict[str, dict[int, dict[str, Any]]] = {
    "safe_reacher": {
        1: {"num_hazards": 3},
        2: {"num_hazards": 6},
        3: {"num_hazards": 9},
    },
    "safe_walker": {
        1: {"max_gap": 6.0},
        2: {"max_gap": 4.0},
        3: {"max_gap": 2.0},
    },
    "humanoid_height_constrained": {
        1: {"max_height": 1.60},
        2: {"max_height": 1.45},
        3: {"max_height": 1.30},
    },
    "ant_velocity_constrained": {
        1: {"max_velocity": 1.0},
        2: {"max_velocity": 0.5},
        3: {"max_velocity": 0.3},
    },
    "safe_point_goal": {
        1: {
            "goal_type": "cylinder",
            "goal_count": 2,
            "goal_size": 0.2,
            "goal_height": 0.2,
            "hazard_specs": [
                {"type": "cylinder", "count": 12, "size": 0.3, "height": 0.01, "collidable": False},
                {"type": "outer_wall", "offset": 0.5, "thickness": 0.06, "height": 0.1, "collidable": True,
                 "fixed": True},
            ],
        },
        2: {
            "goal_type": "cylinder",
            "goal_count": 2,
            "goal_size": 0.2,
            "goal_height": 0.2,
            "hazard_specs": [
                {"type": "cube", "count": 3, "size": 0.3, "height": 0.01, "collidable": False},
                {"type": "cube", "count": 2, "size": 0.2, "height": 0.5, "collidable": True},
                {"type": "cylinder", "count": 4, "size": 0.4, "height": 0.01, "collidable": False},
                {"type": "cylinder", "count": 3, "size": 0.2, "height": 0.4, "collidable": True},
                {"type": "outer_wall", "offset": 0.5, "thickness": 0.06, "height": 0.1, "collidable": True,
                 "fixed": True},
            ],
        },
        3: {
            "goal_type": "cylinder",
            "goal_count": 2,
            "goal_size": 0.2,
            "goal_height": 0.2,
            "hazard_specs": [
                {"type": "cube", "count": 6, "size": 0.25, "height": 0.01, "collidable": False},
                {"type": "cube", "count": 4, "size": 0.2, "height": 0.5, "collidable": True},
                {"type": "cylinder", "count": 6, "size": 0.35, "height": 0.01, "collidable": False},
                {"type": "cylinder", "count": 4, "size": 0.2, "height": 0.4, "collidable": True},
                {"type": "outer_wall", "offset": 0.5, "thickness": 0.06, "height": 0.1, "collidable": True,
                 "fixed": True},
            ],
        },
    },
}


def _merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges src into dst and returns dst."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge_dict(dst[k], v)
        else:
            dst[k] = v
    return dst


def apply_difficulty(env_name: str, env_kwargs: dict[str, Any] | None, level: int) -> dict[str, Any]:
    """Apply difficulty-level overrides to environment kwargs.

    Args:
        env_name: Name of the environment
        env_kwargs: User-provided environment kwargs (can be None)
        level: Difficulty level (1, 2, or 3)

    Returns:
        Merged kwargs dict with difficulty overrides applied first, then env_kwargs
    """
    if env_name not in _DIFFICULTY_OVERRIDES:
        print(f"Warning: Environment '{env_name}' does not support difficulty levels.")
        return env_kwargs or {}

    env_kwargs = deepcopy(env_kwargs or {})
    overrides = deepcopy(_DIFFICULTY_OVERRIDES[env_name][level])

    # All envs use flat kwargs: merge overrides then env_kwargs (env_kwargs wins)
    out = _merge_dict(deepcopy(overrides), deepcopy(env_kwargs))
    return out
