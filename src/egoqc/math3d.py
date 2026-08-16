from __future__ import annotations

import numpy as np


def euler_xyz_to_matrix(euler: np.ndarray) -> np.ndarray:
    """Match scipy Rotation.from_euler("xyz"): extrinsic x, y, z rotations."""
    euler = np.asarray(euler, dtype=np.float64)
    x, y, z = np.moveaxis(euler, -1, 0)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    out = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    out[..., 0, 0] = cz * cy
    out[..., 0, 1] = cz * sy * sx - sz * cx
    out[..., 0, 2] = cz * sy * cx + sz * sx
    out[..., 1, 0] = sz * cy
    out[..., 1, 1] = sz * sy * sx + cz * cx
    out[..., 1, 2] = sz * sy * cx - cz * sx
    out[..., 2, 0] = -sy
    out[..., 2, 1] = cy * sx
    out[..., 2, 2] = cy * cx
    return out


def matrix_to_euler_xyz(matrix: np.ndarray) -> np.ndarray:
    """Inverse of :func:`euler_xyz_to_matrix` with a stable gimbal-lock branch."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix shape must end in (3, 3), got {matrix.shape}")
    sy = np.clip(-matrix[..., 2, 0], -1.0, 1.0)
    y = np.arcsin(sy)
    regular = np.abs(np.cos(y)) > 1e-7
    x_regular = np.arctan2(matrix[..., 2, 1], matrix[..., 2, 2])
    z_regular = np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0])
    # At gimbal lock x and z are not independently identifiable. Set z=0 and
    # choose x so that converting back still represents the same rotation.
    x_locked = np.arctan2(-matrix[..., 1, 2], matrix[..., 1, 1])
    x = np.where(regular, x_regular, x_locked)
    z = np.where(regular, z_regular, 0.0)
    return np.stack([x, y, z], axis=-1)


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues conversion, including a stable zero-angle limit."""

    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"axis-angle shape must end in 3, got {axis_angle.shape}")
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    axis = axis_angle / np.maximum(angle, 1e-12)
    x, y, z = np.moveaxis(axis, -1, 0)
    zero = np.zeros_like(x)
    skew = np.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], axis=-1
    ).reshape(axis_angle.shape[:-1] + (3, 3))
    eye = np.broadcast_to(np.eye(3), skew.shape)
    sine = np.sin(angle)[..., None]
    cosine = np.cos(angle)[..., None]
    rotation = eye + sine * skew + (1.0 - cosine) * (skew @ skew)
    return np.where((angle < 1e-10)[..., None], eye, rotation)


def so3_errors(rotations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotations = np.asarray(rotations, dtype=np.float64)
    eye = np.eye(3)
    gram = np.swapaxes(rotations, -1, -2) @ rotations
    orth = np.linalg.norm(gram - eye, axis=(-2, -1))
    det = np.abs(np.linalg.det(rotations) - 1.0)
    return orth, det


def geodesic_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    rel = np.swapaxes(a, -1, -2) @ b
    cosine = (np.trace(rel, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def transform_points(extrinsics_w2c: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    r = extrinsics_w2c[..., :3, :3]
    t = extrinsics_w2c[..., :3, 3]
    return np.einsum("...ij,...j->...i", r, points_world) + t


def longest_false_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return int(longest)
