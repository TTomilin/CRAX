"""Shared utilities for velocity-constrained environments."""

from __future__ import annotations

import jax
from jax import numpy as jp


def compute_velocity(pos_before: jax.Array, pos_after: jax.Array, dt: float) -> jax.Array:
    """Returns velocity from two position samples."""
    return (pos_after - pos_before) / dt


def planar_speed(velocity_xy: jax.Array) -> jax.Array:
    """Returns planar speed from xy velocity."""
    return jp.linalg.norm(velocity_xy, axis=-1)


def binary_velocity_cost(
        velocity_value: jax.Array,
        threshold: float,
        cost_weight: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Binary cost: 1 if velocity exceeds threshold, else 0."""
    threshold_value = jp.asarray(threshold)
    violation = (velocity_value > threshold_value).astype(jp.float32)
    return violation * cost_weight, violation


def hinge_velocity_cost(
        velocity_value: jax.Array,
        max_velocity: float,
        cost_weight: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Soft-hinge cost for velocity exceeding a maximum."""
    violation = jp.maximum(0.0, velocity_value - max_velocity)
    return violation * cost_weight, violation


def broadcast_to_reward(value: jax.Array, reward: jax.Array) -> jax.Array:
    """Broadcasts a value to match the reward shape."""
    return jp.broadcast_to(value, jp.shape(reward))


def add_velocity_cost_metrics(
        metrics: dict,
        reward: jax.Array,
        *,
        velocity_value: jax.Array,
        threshold: float,
        violation: jax.Array,
        cost: jax.Array,
) -> dict:
    """Adds standardized velocity cost metrics to the metrics dict."""
    threshold_value = jp.asarray(threshold)
    metrics.update(
        velocity_value=broadcast_to_reward(velocity_value, reward),
        velocity_magnitude=broadcast_to_reward(velocity_value, reward),
        velocity_threshold=broadcast_to_reward(threshold_value, reward),
        velocity_violation=broadcast_to_reward(violation, reward),
        velocity_cost=broadcast_to_reward(cost, reward),
        cost=broadcast_to_reward(cost, reward),
        reward_unscaled=broadcast_to_reward(reward, reward),
    )
    return metrics


def build_velocity_info(
        info: dict | None,
        *,
        cost: jax.Array,
        velocity_value: jax.Array,
        threshold: float,
        violation: jax.Array,
        step_count: int | None = None,
) -> dict:
    """Builds a new info dict with velocity constraint details."""
    base = dict(info or {})
    threshold_value = jp.asarray(threshold)
    base.update({
        "cost": cost,
        "velocity_value": velocity_value,
        "velocity_magnitude": velocity_value,
        "velocity_threshold": threshold_value,
        "velocity_violation": violation,
    })
    if step_count is not None:
        base["step_count"] = step_count
    return base
