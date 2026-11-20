# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Blender (meters, Z-up, right-handed) to Unreal (centimeters, Z-up, left-handed).

Blender world axes are +X right, +Y forward, +Z up. Unreal world axes are +X forward,
+Y right, +Z up. The change of basis is a reflection, which is what turns the
right-handed coordinates into left-handed ones:

    unreal_x = blender_y * 100
    unreal_y = blender_x * 100
    unreal_z = blender_z * 100

Mesh vertices are converted with the same mapping before export, and instance
transforms are converted to match. Unreal must import those meshes with scene-axis
and scene-unit conversion disabled, or it will convert them a second time.
"""

from __future__ import annotations

import math

import numpy as np

CM_PER_METER = 100.0

# Columns are the Unreal images of the Blender basis vectors. det(M) == -1.
BLENDER_TO_UNREAL = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def blender_point_to_unreal(point_m: np.ndarray) -> np.ndarray:
    """Convert a Blender world point in meters to Unreal centimeters."""
    point = np.asarray(point_m, dtype=np.float64)
    return CM_PER_METER * (BLENDER_TO_UNREAL @ point)


def convert_points_to_unreal(points_m: np.ndarray) -> np.ndarray:
    """Convert an (N, 3) array of Blender local or world points to Unreal centimeters."""
    points = np.asarray(points_m, dtype=np.float64)
    return CM_PER_METER * (BLENDER_TO_UNREAL @ points.T).T


def blender_matrix_to_unreal(matrix: np.ndarray) -> dict:
    """Convert a Blender 4x4 world matrix into an Unreal location, rotation and scale.

    The linear part is conjugated by the axis-change matrix so it applies to vertices
    that were already converted into Unreal centimeters. Scale is expressed in the
    rotated Unreal axes. ``shear`` is the largest leftover off-diagonal after that
    split; Unreal instances cannot represent shear.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    location = blender_point_to_unreal(translation)
    unreal_linear = BLENDER_TO_UNREAL @ linear @ BLENDER_TO_UNREAL
    rotation, scale, shear = decompose_linear(unreal_linear)
    return {
        "location": [float(value) for value in location],
        "rotation_quat": matrix_to_quat(rotation),
        "rotation_euler_deg": matrix_to_rotator(rotation),
        "scale": [float(value) for value in scale],
        "shear": float(shear),
    }


def blender_camera_to_unreal(
    matrix: np.ndarray,
) -> tuple[list[float], list[float], dict]:
    """Convert a Blender camera world matrix to an Unreal cine-camera transform.

    A Blender camera looks down local -Z, with local +X right and local +Y up.
    An Unreal camera looks down local +X, with local +Y right and local +Z up.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    rotation = matrix[:3, :3]
    forward = BLENDER_TO_UNREAL @ (-rotation[:, 2])
    up = BLENDER_TO_UNREAL @ rotation[:, 1]
    forward = _normalize(forward)
    up = up - forward * float(np.dot(up, forward))
    up = _normalize(up)
    right = _normalize(np.cross(up, forward))
    unreal_rotation = np.column_stack([forward, right, up])
    location = [float(value) for value in blender_point_to_unreal(matrix[:3, 3])]
    return location, matrix_to_quat(unreal_rotation), matrix_to_rotator(unreal_rotation)


def decompose_linear(linear: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Split a 3x3 matrix into a proper rotation, a scale vector and a shear amount."""
    matrix = np.asarray(linear, dtype=np.float64)
    left, _singular, right = np.linalg.svd(matrix)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left = left.copy()
        left[:, -1] *= -1.0
        rotation = left @ right
    local = rotation.T @ matrix
    scale = np.diag(local).astype(np.float64).copy()
    shear = float(np.max(np.abs(local - np.diag(scale))))
    return rotation, scale, shear


def matrix_to_quat(rotation: np.ndarray) -> list[float]:
    """Return an ``(x, y, z, w)`` quaternion, matching ``unreal.Quat``."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    quat = np.array([x, y, z, w], dtype=np.float64)
    if quat[3] < 0.0:
        quat *= -1.0
    quat /= np.linalg.norm(quat)
    return [float(value) for value in quat]


def rotator_to_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Column-vector matrix for an Unreal rotator.

    Pitch rotates around Y, yaw around Z and roll around X. The columns are the
    rotated X, Y and Z axes, matching ``FRotationMatrix`` with row vectors transposed.
    """
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    roll_rad = math.radians(roll)
    sp, cp = math.sin(pitch_rad), math.cos(pitch_rad)
    sy, cy = math.sin(yaw_rad), math.cos(yaw_rad)
    sr, cr = math.sin(roll_rad), math.cos(roll_rad)
    x_axis = np.array([cp * cy, cp * sy, sp])
    y_axis = np.array(
        [
            sr * sp * cy - cr * sy,
            sr * sp * sy + cr * cy,
            -sr * cp,
        ]
    )
    z_axis = np.array(
        [
            -(cr * sp * cy + sr * sy),
            cy * sr - cr * sp * sy,
            cr * cp,
        ]
    )
    return np.column_stack([x_axis, y_axis, z_axis])


def matrix_to_rotator(rotation: np.ndarray) -> dict:
    """Extract Unreal pitch, yaw and roll in degrees from a column-vector matrix."""
    matrix = np.asarray(rotation, dtype=np.float64)
    x_axis = matrix[:, 0]
    y_axis = matrix[:, 1]
    z_axis = matrix[:, 2]
    pitch = math.degrees(math.atan2(x_axis[2], math.hypot(x_axis[0], x_axis[1])))
    yaw = math.degrees(math.atan2(x_axis[1], x_axis[0]))
    partial = rotator_to_matrix(pitch, yaw, 0.0)
    sy_axis = partial[:, 1]
    roll = math.degrees(
        math.atan2(float(np.dot(z_axis, sy_axis)), float(np.dot(y_axis, sy_axis)))
    )
    return {
        "pitch": _clean_angle(pitch),
        "yaw": _clean_angle(yaw),
        "roll": _clean_angle(roll),
    }


def basis_from_x_axis(direction: np.ndarray) -> np.ndarray:
    """Build a rotation whose +X axis is ``direction`` and whose determinant is +1."""
    x_axis = _normalize(np.asarray(direction, dtype=np.float64))
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(x_axis, reference))) > 0.99:
        reference = np.array([0.0, 1.0, 0.0])
    z_axis = reference - x_axis * float(np.dot(reference, x_axis))
    z_axis = _normalize(z_axis)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / length


def _clean_angle(angle: float) -> float:
    if abs(angle) < 1e-8:
        return 0.0
    return float(angle)
