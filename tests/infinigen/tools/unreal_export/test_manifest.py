# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import json

import pytest

from infinigen.tools.unreal_export.manifest import (
    ManifestError,
    build_manifest,
    load_manifest,
    validate_manifest,
)


def _sample(**overrides):
    manifest = build_manifest(
        assets=[
            {
                "id": "asset_0001_Rock",
                "name": "Rock",
                "file": "assets/asset_0001_Rock.fbx",
                "textures": {"base_color": "textures/asset_0001_Rock_DIFFUSE.png"},
                "baked": False,
            }
        ],
        instances=[
            {
                "asset_id": "asset_0001_Rock",
                "name": "Rock",
                "location": [0.0, 0.0, 0.0],
                "rotation_quat": [0.0, 0.0, 0.0, 1.0],
                "rotation_euler_deg": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                "scale": [1.0, 1.0, 1.0],
            },
            {
                "asset_id": "asset_0001_Rock",
                "name": "Rock.001",
                "location": [0.0, 300.0, 0.0],
                "rotation_quat": [0.0, 0.0, 0.0, 1.0],
                "rotation_euler_deg": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                "scale": [1.0, 1.0, 1.0],
            },
        ],
        terrain={
            "name": "OpaqueTerrain",
            "heightmap": "terrain/heightmap.png",
            "heightmap_raw": "terrain/heightmap.r16",
            "resolution": [505, 505],
            "scale": {"x": 100.0, "y": 80.0, "z": 100.0},
            "location": [-1000.0, -1000.0, 0.0],
        },
        cameras=[
            {
                "name": "Camera",
                "sensor_width_mm": 36.0,
                "sensor_height_mm": 24.0,
                "frames": [
                    {
                        "frame": 1,
                        "location": [-500.0, 0.0, 200.0],
                        "rotation_quat": [0.0, 0.0, 0.0, 1.0],
                        "rotation_euler_deg": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                        "focal_length_mm": 35.0,
                    }
                ],
            }
        ],
        lights=[
            {
                "name": "Sun",
                "source": "sun_lamp",
                "kind": "directional",
                "direction": [0.0, 0.0, -1.0],
                "rotation_euler_deg": {"pitch": -90.0, "yaw": 0.0, "roll": 0.0},
                "intensity_lux": 500.0,
                "color": [1.0, 1.0, 1.0],
            }
        ],
        fps=24.0,
        frame_start=1,
        frame_end=1,
        mesh_format="fbx",
    )
    manifest.update(overrides)
    return manifest


def test_manifest_roundtrip_uses_relative_paths_and_unreal_units(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = _sample()
    path.write_text(json.dumps(manifest))
    loaded = load_manifest(path)
    assert loaded["version"] == 1
    assert loaded["units"]["distance"] == "centimeters"
    assert loaded["units"]["up_axis"] == "Z"
    assert loaded["units"]["handedness"] == "left"
    assert loaded["assets"][0]["file"] == "assets/asset_0001_Rock.fbx"
    assert len(loaded["instances"]) == 2
    assert loaded["instances"][0]["asset_id"] == loaded["instances"][1]["asset_id"]
    assert loaded["fps"] == 24.0


def test_validate_rejects_an_instance_whose_asset_is_missing():
    manifest = _sample()
    manifest["instances"][0]["asset_id"] = "missing"
    with pytest.raises(ManifestError, match="missing"):
        validate_manifest(manifest)


def test_validate_rejects_a_non_square_or_illegal_heightmap():
    manifest = _sample()
    manifest["terrain"]["resolution"] = [500, 500]
    with pytest.raises(ManifestError, match="Landscape"):
        validate_manifest(manifest)


def test_validate_rejects_absolute_asset_paths():
    manifest = _sample()
    manifest["assets"][0]["file"] = "/tmp/rock.fbx"
    with pytest.raises(ManifestError, match="relative"):
        validate_manifest(manifest)


def test_missing_camera_frame_fields_are_reported():
    manifest = _sample()
    del manifest["cameras"][0]["frames"][0]["focal_length_mm"]
    with pytest.raises(ManifestError, match="focal_length_mm"):
        validate_manifest(manifest)
