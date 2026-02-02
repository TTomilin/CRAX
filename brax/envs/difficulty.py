"""
Centralized difficulty mapping for safety environments.

This module defines a small, extensible system to translate a difficulty
level (1, 2, 3) into environment-specific parameter overrides.

It is intentionally lightweight and modular: add new env handlers or tweak
mappings in a single place without touching training code or env classes.

Naming Convention Support:
- New style: safe_[task]_[agent] (e.g., safe_goal_point, safe_circle_point)
- Old style: safe_[agent]_[task] or safe_[task] (e.g., safe_point_goal, safe_walker)

Both naming conventions are supported for backward compatibility.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

# ============================================================================
# Task-based difficulty configurations
# These define difficulty levels for each task type, independent of agent
# ============================================================================

_TASK_DIFFICULTY_CONFIGS: dict[str, dict[int, dict[str, Any]]] = {
    # Goal navigation task - navigate to goal while avoiding hazards
    "goal": {
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

    # Circle task - navigate in circles while staying within boundaries
    "circle": {
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
                {"type": "cylinder", "count": 1, "size": 0.15, "height": 0.15, "alpha_transparent": 1.0,
                 "collidable": False, "fixed": False},
            ],
        },
        # Level 3 (smaller boundary, 2 randomly placed hazards)
        3: {
            "boundary_x": 0.975,
            "boundary_y": 0.975,
            "hazard_specs": [
                {"type": "cylinder", "count": 2, "size": 0.2, "height": 0.2, "alpha_transparent": 1.0,
                 "collidable": False, "fixed": False},
            ],
        },
    },

    # Button task - press the correct button among multiple buttons
    "button": {
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

    # Push task - push a block to a goal
    "push": {
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

    # Pathway task - traverse hazard corridor (formerly "run")
    "pathway": {
        1: {"max_gap": 6.0},
        2: {"max_gap": 4.0},
        3: {"max_gap": 2.0},
    },

    # Height task - maintain head below height threshold
    "height": {
        1: {"max_height": 1.20},
        2: {"max_height": 1.10},
        3: {"max_height": 1.00},
    },

    # Lift task for Ant - keep certain feet off the ground
    "lift_ant": {
        # Level 1: Front-left leg must stay off ground
        1: {"restricted_feet": ["front_left"]},
        # Level 2: Diagonal legs (front-left and back-right) must stay off
        2: {"restricted_feet": ["front_left", "back_right"]},
        # Level 3: Only back-right can touch (all others restricted)
        3: {"restricted_feet": ["front_left", "front_right", "back_left"]},
    },

    # Lift task for Spider - keep certain feet off the ground
    "lift_spider": {
        # Level 1: Keep 2 legs up (front-left + back-right diagonal)
        1: {"restricted_feet": ["front_left", "back_right"]},
        # Level 2: Keep 3 legs up (alternating tripod)
        2: {"restricted_feet": ["front_left", "mid_right", "back_left"]},
        # Level 3: Keep 4 legs up (only center legs may touch)
        3: {"restricted_feet": ["front_left", "front_right", "back_left", "back_right"]},
    },

    # Reach task - reach target while avoiding hazards
    "reach": {
        1: {"num_hazards": 4},
        2: {"num_hazards": 7},
        3: {"num_hazards": 10},
    },

    # Velocity task - maintain velocity below threshold
    "velocity": {
        # The actual threshold is computed in safe_velocity.py based on (agent, level)
        1: {"level": 1},
        2: {"level": 2},
        3: {"level": 3},
    },
}

# ============================================================================
# Environment name to task mapping
# Maps both old and new naming conventions to task configurations
# ============================================================================

_ENV_TO_TASK: dict[str, str] = {
    # New naming convention: safe_[task]_[agent]
    "safe_goal_point": "goal",
    "safe_circle_point": "circle",
    "safe_button_point": "button",
    "safe_push_point": "push",
    "safe_pathway_walker2d": "pathway",
    "safe_height_humanoid": "height",
    "safe_lift_ant": "lift_ant",
    "safe_lift_spider": "lift_spider",
    "safe_reach_reacher": "reach",

    # Old naming convention (backward compatibility)
    "safe_point_goal": "goal",
    "safe_point_circle": "circle",
    "safe_point_button": "button",
    "block_push_goal": "push",
    "safe_walker": "pathway",
    "safe_height": "height",
    "safe_ant": "lift_ant",
    "safe_spider": "lift_spider",
    "safe_reacher": "reach",
    "safe_velocity": "velocity",
}


def _merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges src into dst and returns dst."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge_dict(dst[k], v)
        else:
            dst[k] = v
    return dst


def get_task_for_env(env_name: str) -> str | None:
    """Returns the task type for an environment name.

    Args:
        env_name: Name of the environment (new or old naming convention)

    Returns:
        Task type string (e.g., "goal", "circle") or None if not found
    """
    return _ENV_TO_TASK.get(env_name)


def get_supported_levels(env_name: str) -> list[int]:
    """Returns the list of supported difficulty levels for an environment.

    Args:
        env_name: Name of the environment

    Returns:
        List of supported levels (e.g., [1, 2, 3]) or empty list if not supported
    """
    task = _ENV_TO_TASK.get(env_name)
    if task is None or task not in _TASK_DIFFICULTY_CONFIGS:
        return []
    return sorted(_TASK_DIFFICULTY_CONFIGS[task].keys())


def supports_difficulty(env_name: str) -> bool:
    """Check if an environment supports difficulty levels."""
    task = _ENV_TO_TASK.get(env_name)
    return task is not None and task in _TASK_DIFFICULTY_CONFIGS


def apply_difficulty(env_name: str, env_kwargs: dict[str, Any] | None, level: int) -> dict[str, Any]:
    """Apply difficulty-level overrides to environment kwargs.

    Args:
        env_name: Name of the environment
        env_kwargs: User-provided environment kwargs (can be None)
        level: Difficulty level (1, 2, or 3)

    Returns:
        Merged kwargs dict with difficulty overrides applied first, then env_kwargs
    """
    task = _ENV_TO_TASK.get(env_name)

    if task is None:
        print(f"Warning: Environment '{env_name}' does not have a task mapping for difficulty levels.")
        return env_kwargs or {}

    if task not in _TASK_DIFFICULTY_CONFIGS:
        print(f"Warning: Task '{task}' does not have difficulty configurations defined.")
        return env_kwargs or {}

    if level not in _TASK_DIFFICULTY_CONFIGS[task]:
        print(f"Warning: Level {level} not defined for task '{task}'. Available levels: {list(_TASK_DIFFICULTY_CONFIGS[task].keys())}")
        return env_kwargs or {}

    env_kwargs = deepcopy(env_kwargs or {})
    overrides = deepcopy(_TASK_DIFFICULTY_CONFIGS[task][level])

    # All envs use flat kwargs: merge overrides then env_kwargs (env_kwargs wins)
    out = _merge_dict(deepcopy(overrides), deepcopy(env_kwargs))
    return out


def register_task_difficulty(task_name: str, level: int, config: dict[str, Any]) -> None:
    """Register or update a difficulty configuration for a task.

    Args:
        task_name: Name of the task (e.g., "goal", "circle")
        level: Difficulty level (typically 1, 2, or 3)
        config: Configuration dict to apply at this level
    """
    if task_name not in _TASK_DIFFICULTY_CONFIGS:
        _TASK_DIFFICULTY_CONFIGS[task_name] = {}
    _TASK_DIFFICULTY_CONFIGS[task_name][level] = config


def register_env_task_mapping(env_name: str, task_name: str) -> None:
    """Register a mapping from environment name to task type.

    Args:
        env_name: Name of the environment
        task_name: Name of the task this environment uses
    """
    _ENV_TO_TASK[env_name] = task_name
