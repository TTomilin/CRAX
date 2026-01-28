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
        1: {"num_hazards": 4},
        2: {"num_hazards": 7},
        3: {"num_hazards": 10},
    },
    "safe_walker": {
        1: {"max_gap": 6.0},
        2: {"max_gap": 4.0},
        3: {"max_gap": 2.0},
    },
    "safe_height": {
        1: {"max_height": 1.0},  # Easiest - can stand upright
        2: {"max_height": 0.8},  # Medium - must crouch slightly
        3: {"max_height": 0.6},  # Hardest - must crouch significantly
    },
    # Unified safe_velocity environment - level determines threshold multiplier:
    # Level 1: 1.0x baseline (easiest), Level 2: 0.75x, Level 3: 0.5x (hardest)
    # The actual threshold is computed in safe_velocity.py based on (agent, level)
    "safe_velocity": {
        1: {"level": 1},
        2: {"level": 2},
        3: {"level": 3},
    },
    # Safe spider: 6-legged robot that must keep certain legs off ground
    # Level 1: 2 legs up (diagonal), Level 2: 3 legs up (tripod), Level 3: 4 legs up
    "safe_spider": {
        1: {"restricted_feet": ["front_left", "back_right"]},  # Diagonal opposite
        2: {"restricted_feet": ["front_left", "mid_right", "back_left"]},  # Alternating tripod
        3: {"restricted_feet": ["front_left", "front_right", "back_left", "back_right"]},  # Only mid legs touch
    },
    "safe_point_goal": {
        1: {
            "goal_type": "cylinder",
            "goal_count": 2,
            "goal_size": 0.2,
            "goal_height": 0.2,
            "hazard_specs": [
                {"type": "cylinder", "count": 12, "size": 0.4, "height": 0.01, "collidable": False},
                {"type": "outer_wall", "offset": 0.5, "thickness": 0.06, "height": 0.1, "collidable": True,
                 "fixed": True},
            ],
        },
        2: {
            "goal_type": "cylinder",
            "goal_count": 2,
            "goal_size": 0.18,
            "goal_height": 0.2,
            "hazard_specs": [
                {"type": "cylinder", "count": 8, "size": 0.4, "height": 0.01, "collidable": False},
                {"type": "cylinder", "count": 8, "size": 0.3, "height": 0.4, "collidable": True},
                {"type": "outer_wall", "offset": 0.5, "thickness": 0.06, "height": 0.1, "collidable": True,
                 "fixed": True},
            ],
        },
        3: {
            "goal_type": "cylinder",
            "goal_count": 2,
            "goal_size": 0.16,
            "goal_height": 0.2,
            "hazard_specs": [
                {"type": "cube", "count": 6, "size": 0.3, "height": 0.01, "collidable": False},
                {"type": "cube", "count": 4, "size": 0.25, "height": 0.5, "collidable": True},
                {"type": "cylinder", "count": 6, "size": 0.35, "height": 0.01, "collidable": False},
                {"type": "cylinder", "count": 4, "size": 0.25, "height": 0.4, "collidable": True},
                {"type": "outer_wall", "offset": 0.5, "thickness": 0.06, "height": 0.1, "collidable": True,
                 "fixed": True},
            ],
        },
    },
    "safe_point_circle": {
        # Level 1 (vertical walls)
        1: {
            "boundary_x": 1.125,
            "boundary_y": None,
        },
        # Level 2 (square boundary, 1 randomly placed hazard)
        2: {
            "boundary_x": 1.05,
            "boundary_y": 1.05,
            "hazard_specs": [
                {"type": "cylinder", "count": 1, "size": 0.075, "height": 0.075, "alpha_transparent": 1.0,
                 "collidable": False, "fixed": False},
            ],
        },
        # Level 3 (smaller boundary, 2 randomly placed hazards)
        3: {
            "boundary_x": 0.975,
            "boundary_y": 0.975,
            "hazard_specs": [
                {"type": "cylinder", "count": 2, "size": 0.1, "height": 0.1, "alpha_transparent": 1.0,
                 "collidable": False, "fixed": False},
            ],
        },
    },
    "block_push_goal": {
        # Level 1: Stationary goal
        1: {
            "goal_velocity": 0.0,
        },
        # Level 2: Slow moving goal
        2: {
            "goal_velocity": 0.3,
        },
        # Level 3: Fast moving goal
        3: {
            "goal_velocity": 0.6,
        },
    },
    "safe_point_button": {
        # Level 1: Hazards and gremlins, constrained buttons
        1: {
            "placement_extents": (-1.5, -1.5, 1.5, 1.5),
            "buttons_constrained": True,
            "hazard_specs": [
                {"type": "cylinder", "count": 4, "size": 0.2, "height": 0.2, "collidable": True, "fixed": False},
                {"type": "gremlin", "count": 4, "size": 0.1, "height": 0.1, "travel": 0.35, "collidable": True,
                 "fixed": False},
            ],
        },
        # Level 2: More hazards and gremlins
        2: {
            "placement_extents": (-1.8, -1.8, 1.8, 1.8),
            "buttons_constrained": True,
            "hazard_specs": [
                {"type": "cylinder", "count": 8, "size": 0.2, "height": 0.2, "collidable": True, "fixed": False},
                {"type": "gremlin", "count": 6, "size": 0.1, "height": 0.1, "travel": 0.35, "collidable": True,
                 "fixed": False},
            ],
        },
        # Level 3: Even more hazards and gremlins in a smaller space
        3: {
            "placement_extents": (-1.2, -1.2, 1.2, 1.2),
            "buttons_constrained": True,
            "hazard_specs": [
                {"type": "cylinder", "count": 12, "size": 0.2, "height": 0.2, "collidable": True, "fixed": False},
                {"type": "gremlin", "count": 8, "size": 0.1, "height": 0.1, "travel": 0.45, "collidable": True,
                 "fixed": False},
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


def get_supported_levels(env_name: str) -> list[int]:
    """Returns the list of supported difficulty levels for an environment.

    Args:
        env_name: Name of the environment

    Returns:
        List of supported levels (e.g., [1, 2, 3]) or empty list if not supported
    """
    if env_name not in _DIFFICULTY_OVERRIDES:
        return []
    return sorted(_DIFFICULTY_OVERRIDES[env_name].keys())


def supports_difficulty(env_name: str) -> bool:
    """Check if an environment supports difficulty levels."""
    return env_name in _DIFFICULTY_OVERRIDES


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
