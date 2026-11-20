# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import math

import numpy as np

from infinigen.tools.unreal_export.lights import (
    WATT_PER_SQ_M_TO_LUX,
    directional_light_from_direction,
    nishita_sun_direction_blender,
    sun_lamp_travel_direction_blender,
)


def test_nishita_sun_is_overhead_at_zenith_and_on_plus_y_at_the_horizon():
    zenith = nishita_sun_direction_blender(math.pi / 2, 0.3)
    np.testing.assert_allclose(zenith, [0.0, 0.0, 1.0], atol=1e-8)
    horizon = nishita_sun_direction_blender(0.0, 0.0)
    np.testing.assert_allclose(horizon, [0.0, 1.0, 0.0], atol=1e-8)
    assert abs(np.linalg.norm(horizon) - 1.0) < 1e-8


def test_sun_lamp_rays_follow_local_negative_z():
    # Light object rotated so local -Z points along Blender +X.
    rotation = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    direction = sun_lamp_travel_direction_blender(matrix)
    np.testing.assert_allclose(direction, [1.0, 0.0, 0.0], atol=1e-8)


def test_directional_light_x_axis_is_the_travel_direction():
    travel = np.array([0.0, 0.0, -1.0])
    light = directional_light_from_direction(
        travel_direction_blender=travel,
        intensity_lux=5.0 * WATT_PER_SQ_M_TO_LUX,
        color=(1.0, 0.9, 0.8),
        name="Sun",
        source="sun_lamp",
    )
    from infinigen.tools.unreal_export.coords import rotator_to_matrix

    euler = light["rotation_euler_deg"]
    forward = rotator_to_matrix(euler["pitch"], euler["yaw"], euler["roll"])[:, 0]
    # Blender -Z travel becomes Unreal -Z as well.
    np.testing.assert_allclose(light["direction"], [0.0, 0.0, -1.0], atol=1e-8)
    np.testing.assert_allclose(forward, light["direction"], atol=1e-6)
    assert light["intensity_lux"] == 500.0
    assert light["color"] == [1.0, 0.9, 0.8]
