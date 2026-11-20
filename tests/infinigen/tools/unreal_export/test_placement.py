# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import pytest

from infinigen.tools.unreal_export.manifest import build_manifest
from infinigen.tools.unreal_export.placement import (
    UnrealImportError,
    fbx_import_options,
    format_landscape_instructions,
    plan_import,
    transform_channel_values,
)


def _manifest():
    return build_manifest(
        assets=[
            {
                "id": "asset_0001_Rock",
                "name": "Rock",
                "file": "assets/asset_0001_Rock.fbx",
                "textures": {"base_color": "textures/rock.png"},
                "baked": True,
            },
            {
                "id": "asset_0002_Tree",
                "name": "Tree",
                "file": "assets/asset_0002_Tree.fbx",
                "textures": {},
                "baked": False,
            },
        ],
        instances=[
            {
                "asset_id": "asset_0001_Rock",
                "name": f"Rock.{index}",
                "location": [float(index), 0.0, 0.0],
                "rotation_quat": [0.0, 0.0, 0.0, 1.0],
                "rotation_euler_deg": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                "scale": [1.0, 1.0, 1.0],
            }
            for index in range(5)
        ]
        + [
            {
                "asset_id": "asset_0002_Tree",
                "name": "Tree",
                "location": [10.0, 20.0, 30.0],
                "rotation_quat": [0.0, 0.0, 0.0, 1.0],
                "rotation_euler_deg": {"pitch": 1.0, "yaw": 2.0, "roll": 3.0},
                "scale": [2.0, 2.0, 2.0],
            }
        ],
        terrain={
            "name": "OpaqueTerrain",
            "heightmap": "terrain/heightmap.png",
            "heightmap_raw": "terrain/heightmap.r16",
            "preview_mesh": "terrain/terrain_preview.fbx",
            "resolution": [505, 505],
            "components": {"x": 4, "y": 4},
            "sections_per_component": 2,
            "quads_per_section": 63,
            "scale": {"x": 200.0, "y": 200.0, "z": 50.0},
            "location": [1.0, 2.0, 3.0],
            "rotation_euler_deg": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        },
        cameras=[
            {
                "name": "Camera",
                "sensor_width_mm": 36.0,
                "sensor_height_mm": 20.25,
                "frames": [
                    {
                        "frame": 1,
                        "location": [0.0, 0.0, 100.0],
                        "rotation_quat": [0.0, 0.0, 0.0, 1.0],
                        "rotation_euler_deg": {
                            "pitch": -10.0,
                            "yaw": 20.0,
                            "roll": 5.0,
                        },
                        "focal_length_mm": 35.0,
                    },
                    {
                        "frame": 2,
                        "location": [100.0, 0.0, 100.0],
                        "rotation_quat": [0.0, 0.0, 0.0, 1.0],
                        "rotation_euler_deg": {
                            "pitch": -12.0,
                            "yaw": 25.0,
                            "roll": 5.0,
                        },
                        "focal_length_mm": 40.0,
                    },
                ],
            }
        ],
        lights=[],
        fps=24.0,
        frame_start=1,
        frame_end=2,
        mesh_format="fbx",
    )


def test_plan_places_one_instancer_per_asset_not_per_copy():
    plan = plan_import(_manifest(), nanite=True, generate_lods=False, placement="hism")
    assert plan["placement"] == "hism"
    assert plan["nanite"] is True
    assert plan["generate_lods"] is False
    counts = {group["asset_id"]: group["count"] for group in plan["instance_groups"]}
    assert counts == {"asset_0001_Rock": 5, "asset_0002_Tree": 1}
    assert len(plan["instance_groups"]) == 2
    assert plan["fbx_options"]["convert_scene"] is False
    assert plan["fbx_options"]["convert_scene_unit"] is False
    assert plan["fbx_options"]["import_uniform_scale"] == 1.0


def test_transform_track_channel_order_is_location_then_roll_pitch_yaw():
    frame = _manifest()["cameras"][0]["frames"][0]
    channels = transform_channel_values(frame)
    assert channels == [0.0, 0.0, 100.0, 5.0, -10.0, 20.0, 1.0, 1.0, 1.0]


def test_landscape_instructions_include_the_numbers_unreal_asks_for(tmp_path):
    terrain = _manifest()["terrain"]
    text = format_landscape_instructions(
        terrain, heightmap_path=tmp_path / "heightmap.png"
    )
    assert "505 x 505" in text
    assert "Section Size: 63 x 63" in text
    assert "Sections Per Component: 2 x 2" in text
    assert "Number of Components: 4 x 4" in text
    assert "Scale: X=200" in text
    assert "Y=200" in text
    assert "Z=50" in text
    assert "InfinigenLandscape" in text
    assert str(tmp_path / "heightmap.png") in text


def test_foliage_placement_is_explicit_and_unknown_modes_fail():
    plan = plan_import(
        _manifest(), nanite=False, generate_lods=True, placement="foliage"
    )
    assert plan["placement"] == "foliage"
    assert plan["generate_lods"] is True
    assert plan["nanite"] is False
    with pytest.raises(UnrealImportError, match="placement"):
        plan_import(
            _manifest(), nanite=True, generate_lods=False, placement="actor-per-copy"
        )


def test_fbx_options_stay_stable_for_the_preconverted_meshes():
    options = fbx_import_options()
    assert options["convert_scene"] is False
    assert options["convert_scene_unit"] is False
