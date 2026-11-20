# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Pure planning for the Unreal importer.

The Editor script in ``unreal/import_infinigen_scene.py`` executes this plan.
These functions do not import ``unreal`` or ``bpy``, so they can be tested here.
"""

from __future__ import annotations

from pathlib import Path

from infinigen.tools.unreal_export.manifest import validate_manifest

_PLACEMENTS = ("hism", "ism", "foliage")


class UnrealImportError(RuntimeError):
    """Raised when the Unreal script cannot complete a step, with a next action."""


def fbx_import_options() -> dict:
    """Options that keep pre-converted Unreal-centimeter vertices unchanged."""
    return {
        "convert_scene": False,
        "convert_scene_unit": False,
        "import_uniform_scale": 1.0,
        "auto_generate_collision": False,
        "generate_lightmap_uvs": False,
        "combine_meshes": True,
        "import_materials": False,
    }


def transform_channel_values(frame: dict) -> list[float]:
    """Nine floats for a Movie Scene 3D transform section.

    Channel order matches Unreal's ``MovieScene3DTransformSection``:
    location X Y Z, then rotation roll, pitch, yaw, then scale X Y Z.
    """
    location = frame["location"]
    euler = frame["rotation_euler_deg"]
    scale = frame.get("scale", (1.0, 1.0, 1.0))
    return [
        float(location[0]),
        float(location[1]),
        float(location[2]),
        float(euler["roll"]),
        float(euler["pitch"]),
        float(euler["yaw"]),
        float(scale[0]),
        float(scale[1]),
        float(scale[2]),
    ]


def plan_import(
    manifest: dict,
    *,
    nanite: bool,
    generate_lods: bool,
    placement: str,
    content_root: str = "/Game/Infinigen",
) -> dict:
    validate_manifest(manifest)
    if placement not in _PLACEMENTS:
        raise UnrealImportError(
            f"Unknown placement {placement!r}. Use one of: {', '.join(_PLACEMENTS)}. "
            "Copies are placed on an instanced component, not as one actor per copy."
        )
    by_asset: dict[str, list] = {}
    for instance in manifest["instances"]:
        by_asset.setdefault(instance["asset_id"], []).append(instance)
    groups = []
    for asset in manifest["assets"]:
        instances = by_asset.get(asset["id"], [])
        groups.append(
            {
                "asset_id": asset["id"],
                "asset": asset,
                "count": len(instances),
                "instances": instances,
            }
        )
    cameras = []
    for camera in manifest["cameras"]:
        cameras.append(
            {
                "name": camera["name"],
                "sensor_width_mm": camera["sensor_width_mm"],
                "sensor_height_mm": camera["sensor_height_mm"],
                "frames": camera["frames"],
                "channels": [
                    transform_channel_values(frame) for frame in camera["frames"]
                ],
            }
        )
    return {
        "content_root": content_root.rstrip("/"),
        "nanite": bool(nanite),
        "generate_lods": bool(generate_lods),
        "placement": placement,
        "fbx_options": fbx_import_options(),
        "instance_groups": groups,
        "mesh_imports": [asset["file"] for asset in manifest["assets"]],
        "landscape": manifest.get("terrain"),
        "cameras": cameras,
        "lights": manifest["lights"],
        "fps": manifest["fps"],
        "frame_start": manifest["frame_start"],
        "frame_end": manifest["frame_end"],
        "mesh_format": manifest["mesh_format"],
    }


def format_landscape_instructions(terrain: dict, heightmap_path: Path) -> str:
    """Manual Landscape Mode steps, used when the Python API cannot create one."""
    components = terrain["components"]
    scale = terrain["scale"]
    location = terrain["location"]
    resolution = terrain["resolution"]
    return (
        "Unreal Editor Python cannot create a Landscape component grid in stock UE 5.8.\n"
        "spawn_actor_from_class(Landscape) returns a LandscapePlaceholder, and ALandscape::Import "
        "is not exposed to the unreal module.\n\n"
        "The rest of the import can still run. To build the Landscape itself:\n"
        "1. Open Landscape mode (Shift+2) and choose Import from File.\n"
        f"2. Heightmap File: {heightmap_path}\n"
        f"3. Heightmap Resolution: {resolution[0]} x {resolution[1]}\n"
        f"4. Section Size: {terrain['quads_per_section']} x {terrain['quads_per_section']} quads\n"
        f"5. Sections Per Component: {terrain['sections_per_component']} x {terrain['sections_per_component']}\n"
        f"6. Number of Components: {components['x']} x {components['y']}\n"
        f"7. Scale: X={_num(scale['x'])} Y={_num(scale['y'])} Z={_num(scale['z'])}\n"
        f"8. Location: X={_num(location[0])} Y={_num(location[1])} Z={_num(location[2])}\n"
        "9. Name the Landscape actor InfinigenLandscape.\n\n"
        "The heightmap is 16-bit. Row 0 is minimum Unreal Y. "
        "Height in centimeters relative to Location Z is (pixel - 32768) * Scale.Z / 128. "
        "Enter the Scale above and do not let Unreal recalculate Z.\n"
        "Run this script again after the actor exists; it will find InfinigenLandscape and keep going."
    )


def _num(value: float) -> str:
    return f"{float(value):.6g}"
