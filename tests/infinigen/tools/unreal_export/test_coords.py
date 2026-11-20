# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import numpy as np

from infinigen.tools.unreal_export.coords import (
    blender_camera_to_unreal,
    blender_matrix_to_unreal,
    blender_point_to_unreal,
    matrix_to_quat,
    matrix_to_rotator,
    rotator_to_matrix,
)


def test_point_maps_blender_axes_onto_unreal_axes():
    # Blender +X is right, which is Unreal +Y. 1 m -> 100 cm.
    right = blender_point_to_unreal(np.array([1.0, 0.0, 0.0]))
    forward = blender_point_to_unreal(np.array([0.0, 1.0, 0.0]))
    up = blender_point_to_unreal(np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(right, [0.0, 100.0, 0.0])
    np.testing.assert_allclose(forward, [100.0, 0.0, 0.0])
    np.testing.assert_allclose(up, [0.0, 0.0, 100.0])


def test_translation_rotation_and_scale_preserve_a_local_point():
    # 90 degree right-handed yaw in Blender, non-uniform scale, then a translation.
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    scale = np.diag([2.0, 3.0, 4.0])
    linear = rotation @ scale
    translation = np.array([1.0, 2.0, 3.0])
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = translation

    local_blender = np.array([1.0, 0.0, 0.0])
    world_blender = linear @ local_blender + translation
    expected = blender_point_to_unreal(world_blender)

    converted = blender_matrix_to_unreal(matrix)
    quat = converted["rotation_quat"]
    rot = _quat_to_matrix(quat)
    scale_u = np.array(converted["scale"])
    local_unreal = blender_point_to_unreal(
        local_blender
    )  # meters->cm and axis change, origin-free
    # blender_point_to_unreal includes the centimeter scale, which is the mesh-vertex conversion.
    got = rot @ (scale_u * local_unreal) + np.array(converted["location"])
    np.testing.assert_allclose(got, expected, atol=1e-6)
    np.testing.assert_allclose(converted["location"], [200.0, 100.0, 300.0])
    assert converted["shear"] < 1e-8


def test_identity_transform_is_zero_rotation_unit_scale():
    converted = blender_matrix_to_unreal(np.eye(4))
    np.testing.assert_allclose(converted["location"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        converted["rotation_quat"], [0.0, 0.0, 0.0, 1.0], atol=1e-8
    )
    np.testing.assert_allclose(converted["scale"], [1.0, 1.0, 1.0])
    euler = converted["rotation_euler_deg"]
    assert euler["pitch"] == 0.0
    assert euler["yaw"] == 0.0
    assert euler["roll"] == 0.0


def test_rotator_roundtrip_matches_unreal_axis_order():
    # Yaw 90 degrees turns Unreal forward (+X) onto +Y.
    matrix = rotator_to_matrix(pitch=0.0, yaw=90.0, roll=0.0)
    np.testing.assert_allclose(matrix[:, 0], [0.0, 1.0, 0.0], atol=1e-8)
    euler = matrix_to_rotator(matrix)
    assert abs(euler["pitch"]) < 1e-6
    assert abs(euler["yaw"] - 90.0) < 1e-6
    assert abs(euler["roll"]) < 1e-6

    for pitch, yaw, roll in (
        (20.0, -35.0, 10.0),
        (-80.0, 170.0, 45.0),
        (0.0, 0.0, 90.0),
    ):
        recovered = matrix_to_rotator(rotator_to_matrix(pitch, yaw, roll))
        rebuilt = rotator_to_matrix(**recovered)
        original = rotator_to_matrix(pitch, yaw, roll)
        np.testing.assert_allclose(rebuilt, original, atol=1e-6)


def test_camera_looking_down_keeps_view_direction():
    # Identity Blender camera looks down local -Z, which is world -Z.
    location, quat, euler = blender_camera_to_unreal(np.eye(4))
    np.testing.assert_allclose(location, [0.0, 0.0, 0.0])
    forward = _quat_to_matrix(quat)[:, 0]
    np.testing.assert_allclose(forward, [0.0, 0.0, -1.0], atol=1e-8)
    assert abs(euler["pitch"] + 90.0) < 1e-4


def test_matrix_to_quat_is_unit_xyzw():
    quat = matrix_to_quat(np.eye(3))
    assert len(quat) == 4
    assert abs(np.linalg.norm(quat) - 1.0) < 1e-8
    assert quat[3] > 0.0


def _quat_to_matrix(quat):
    x, y, z, w = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
