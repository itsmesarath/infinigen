# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Sun direction and intensity shared by the Blender exporter and the unit tests.

Nishita sun direction matches Cycles ``spherical_to_direction(elevation - pi/2,
rotation - pi/2)``. Blender sun lamps shine along local -Z. The manifest stores
the direction light travels, in Unreal space, which is the +X axis of the
directional light.
"""

from __future__ import annotations

import math

import numpy as np

from infinigen.tools.unreal_export.coords import (
    BLENDER_TO_UNREAL,
    basis_from_x_axis,
    matrix_to_quat,
    matrix_to_rotator,
)

# Blender sun energy is watts per square meter. A clear-day 1000 W/m^2 is about
# 100000 lux, which is the figure Unreal uses for direct sunlight.
WATT_PER_SQ_M_TO_LUX = 100.0
NISHITA_LUX_PER_INTENSITY = 100_000.0


def nishita_sun_direction_blender(elevation: float, rotation: float) -> np.ndarray:
    """Unit vector pointing from the scene toward the Nishita sun, in Blender space."""
    theta = elevation - math.pi / 2.0
    phi = rotation - math.pi / 2.0
    sin_theta = math.sin(theta)
    return np.array(
        [
            sin_theta * math.cos(phi),
            sin_theta * math.sin(phi),
            math.cos(theta),
        ],
        dtype=np.float64,
    )


def sun_lamp_travel_direction_blender(matrix: np.ndarray) -> np.ndarray:
    """Direction a Blender sun lamp pushes light, taken from its world matrix."""
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    direction = -rotation[:, 2]
    return direction / np.linalg.norm(direction)


def sun_color_from_elevation(elevation_rad: float) -> list[float]:
    """Warm the sun toward the horizon. This is a stand-in for the Nishita disk."""
    blend = float(np.clip(elevation_rad / (math.pi / 2.0), 0.0, 1.0))
    horizon = np.array([1.0, 0.55, 0.25])
    noon = np.array([1.0, 0.98, 0.95])
    color = (1.0 - blend) * horizon + blend * noon
    return [float(channel) for channel in color]


def directional_light_from_direction(
    travel_direction_blender: np.ndarray,
    intensity_lux: float,
    color: tuple[float, float, float] | list[float],
    name: str,
    source: str,
) -> dict:
    travel = np.asarray(travel_direction_blender, dtype=np.float64)
    travel = travel / np.linalg.norm(travel)
    unreal_direction = BLENDER_TO_UNREAL @ travel
    unreal_direction = unreal_direction / np.linalg.norm(unreal_direction)
    rotation = basis_from_x_axis(unreal_direction)
    return {
        "name": name,
        "source": source,
        "kind": "directional",
        "direction": [float(value) for value in unreal_direction],
        "rotation_quat": matrix_to_quat(rotation),
        "rotation_euler_deg": matrix_to_rotator(rotation),
        "intensity_lux": float(intensity_lux),
        "color": [float(channel) for channel in color],
    }


def nishita_light(
    elevation: float,
    rotation: float,
    sun_intensity: float,
    strength: float,
    name: str = "NishitaSun",
) -> dict:
    toward_sun = nishita_sun_direction_blender(elevation, rotation)
    light = directional_light_from_direction(
        travel_direction_blender=-toward_sun,
        intensity_lux=float(sun_intensity) * NISHITA_LUX_PER_INTENSITY,
        color=sun_color_from_elevation(elevation),
        name=name,
        source="nishita",
    )
    light["raw"] = {
        "sun_elevation_rad": float(elevation),
        "sun_rotation_rad": float(rotation),
        "sun_intensity": float(sun_intensity),
        "background_strength": float(strength),
    }
    return light
