"""Small quaternion helpers. MuJoCo uses wxyz order."""

from __future__ import annotations

import numpy as np

# 90 degrees about +Y: body +Z (capsule axis) lands on world +X.
LAY_FLAT_X = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product. Applies ``b`` first, then ``a``."""
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by unit quaternion ``q`` (wxyz)."""
    w = float(q[0])
    qvec = np.asarray(q[1:], dtype=np.float64)
    vec = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(qvec, vec)
    return vec + w * t + np.cross(qvec, t)


def yaw_quat(yaw: float) -> np.ndarray:
    """Rotation about world +Z by ``yaw`` radians."""
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def axis_from_quat(q: np.ndarray, local_axis: np.ndarray | None = None) -> np.ndarray:
    if local_axis is None:
        local_axis = np.array([0.0, 0.0, 1.0])
    axis = quat_rotate(q, local_axis)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0])
    return axis / norm
