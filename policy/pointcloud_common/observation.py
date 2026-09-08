"""Shared offline/live RGB-D projection, sampling, and robot action layout."""

from __future__ import annotations

import numpy as np


def action_layout(info: dict, action_type: str):
    if action_type not in ("joint", "ee"):
        raise ValueError("action_type must be joint or ee")
    count = len(info["arm_dim"])
    if count not in (1, 2) or len(info["ee_dim"]) != count:
        raise ValueError("Unsupported robot layout")
    result = []
    for i in range(count):
        prefix = "" if count == 1 else ("left_" if i == 0 else "right_")
        result.extend(
            [
                (
                    prefix + ("arm_joint_state" if action_type == "joint" else "ee_pose"),
                    info["arm_dim"][i] if action_type == "joint" else 7,
                ),
                (prefix + "ee_joint_state", info["ee_dim"][i]),
            ]
        )
    return result


def vector(state: dict, layout):
    values = []
    for key, dim in layout:
        value = np.asarray(state[key], dtype=np.float32).reshape(-1)
        if value.shape != (dim,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key}")
        values.append(value)
    return np.concatenate(values)


def points(camera: dict, *, count: int, channels: int, depth_scale: float):
    depth = np.asarray(camera["depth"], dtype=np.float32).squeeze() * depth_scale
    intr = np.asarray(camera["intrinsic_matrix"], dtype=np.float32)
    if depth.ndim != 2 or intr.shape != (3, 3) or intr[0, 0] <= 0 or intr[1, 1] <= 0:
        raise ValueError("Expected depth [H,W] and valid camera intrinsics")
    y, x = np.nonzero(np.isfinite(depth) & (depth > 0))
    if not len(x):
        raise ValueError("No valid depth pixels")
    # Deterministic raster-uniform sampling. Persisted as an adapter setting;
    # no hidden FPS or frame-to-frame RNG differences between offline and live.
    indices = np.linspace(0, len(x) - 1, count).astype(np.int64)
    x, y = x[indices], y[indices]
    z = depth[y, x]
    xyz = np.stack(
        ((x - intr[0, 2]) * z / intr[0, 0], (y - intr[1, 2]) * z / intr[1, 1], z), axis=-1
    )
    if channels == 6:
        color = np.asarray(camera["color"])
        if color.shape != (*depth.shape, 3) or color.dtype != np.uint8:
            raise ValueError("XYZRGB requires registered uint8 RGB at depth resolution")
        xyz = np.concatenate((xyz, color[y, x].astype(np.float32) / 255.0), axis=-1)
    if channels not in (3, 6):
        raise ValueError("Point channels must be 3 or 6")
    return xyz.astype(np.float32)
