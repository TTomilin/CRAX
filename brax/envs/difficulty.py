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

from ml_collections.config_dict import ConfigDict 

from brax.envs.SafePointGoal import default_config as spg_default_config

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
            "goals": {"type": "cylinder", "count": 2, "size": 0.2, "height": 0.2},
            "hazards": {
                "specs": [
                    {"type": "cylinder", "count": 12, "size": 0.3, "height": 0.01, "collidable": False},
                    {"type": "outer_wall", "offset": 0.5, "thickness": 0.06, "height": 0.1, "collidable": True,
                     "fixed": True},
                ]
            },
        },
        2: {
            "goals": {"type": "cylinder", "count": 2, "size": 0.2, "height": 0.2},
            "hazards": {
                "specs": [
                    {"type": "cube", "count": 3, "size": 0.3, "height": 0.01, "collidable": False},
                    {"type": "cube", "count": 2, "size": 0.2, "height": 0.5, "collidable": True},
                    {"type": "cylinder", "count": 4, "size": 0.4, "height": 0.01, "collidable": False},
                    {"type": "cylinder", "count": 3, "size": 0.2, "height": 0.4, "collidable": True},
                    {"type": "outer_wall", "offset": 0.5, "thickness": 0.06, "height": 0.1, "collidable": True,
                     "fixed": True},
                ]
            },
        },
        3: {
            "goals": {"type": "cylinder", "count": 2, "size": 0.2, "height": 0.2},
            "hazards": {
                "specs": [
                    {"type": "cube", "count": 6, "size": 0.25, "height": 0.01, "collidable": False},
                    {"type": "cube", "count": 4, "size": 0.2, "height": 0.5, "collidable": True},
                    {"type": "cylinder", "count": 6, "size": 0.35, "height": 0.01, "collidable": False},
                    {"type": "cylinder", "count": 4, "size": 0.2, "height": 0.4, "collidable": True},
                    {"type": "outer_wall", "offset": 0.5, "thickness": 0.06, "height": 0.1, "collidable": True,
                     "fixed": True},
                ]
            },
        },
    }
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
    if env_name not in _DIFFICULTY_OVERRIDES:
        print(f"Warning: Environment '{env_name}' does not support difficulty levels.")
        return env_kwargs or {}

    env_kwargs = deepcopy(env_kwargs or {})
    overrides = deepcopy(_DIFFICULTY_OVERRIDES[env_name][level])

    # SafePointGoal: build/merge config and return {"config": ...}
    if env_name == "safe_point_goal":

        def _merge_into_cfg(dst: ConfigDict, src: Any) -> ConfigDict:
            """
            Recursively merge src into dst.
            - src can be dict or ConfigDict
            - nested dict/ConfigDict become nested ConfigDicts
            """
            for k, v in src.items():
                if isinstance(v, (ConfigDict, dict)):
                    cur = dst.get(k, None)
                    if not isinstance(cur, ConfigDict):
                        cur = ConfigDict()
                    dst[k] = _merge_into_cfg(cur, v)
                else:
                    dst[k] = deepcopy(v)
            return dst

        # start from default config
        base_cfg = spg_default_config() if spg_default_config is not None else None
        cfg = (deepcopy(base_cfg))

        # merge difficulty overrides (dict) into cfg
        _merge_into_cfg(cfg, deepcopy(overrides))

        # merge in any top-level kwargs into cfg
        top = {k: v for k, v in env_kwargs.items() if k != "config"}
        _merge_into_cfg(cfg, deepcopy(top))

        return {"config": cfg}

    # Other envs: merge overrides then env_kwargs (env_kwargs wins)
    out = _merge_dict(deepcopy(overrides), deepcopy(env_kwargs))
    return out
