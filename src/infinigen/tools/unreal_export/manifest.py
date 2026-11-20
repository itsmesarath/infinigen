# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Read and write the JSON manifest that the Unreal importer consumes."""

from __future__ import annotations

import json
from pathlib import Path

from infinigen.tools.unreal_export.heightmap import is_legal_landscape_resolution

MANIFEST_VERSION = 1

UNITS = {
    "distance": "centimeters",
    "up_axis": "Z",
    "handedness": "left",
    "axes": {"x": "forward", "y": "right", "z": "up"},
    "source": "blender_z_up_right_handed_meters",
    "conversion": "unreal_x = blender_y * 100; unreal_y = blender_x * 100; unreal_z = blender_z * 100",
}

_EULER_KEYS = ("pitch", "yaw", "roll")


class ManifestError(ValueError):
    """The manifest is missing data the Unreal importer needs."""


def build_manifest(
    *,
    assets: list[dict],
    instances: list[dict],
    terrain: dict | None,
    cameras: list[dict],
    lights: list[dict],
    fps: float,
    frame_start: int,
    frame_end: int,
    mesh_format: str,
    warnings: list[str] | None = None,
) -> dict:
    manifest = {
        "version": MANIFEST_VERSION,
        "units": dict(UNITS),
        "mesh_format": mesh_format,
        "mesh_space": "unreal_cm_z_up_left_handed",
        "fbx_import": {
            "convert_scene": False,
            "convert_scene_unit": False,
            "import_uniform_scale": 1.0,
        },
        "fps": float(fps),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "assets": assets,
        "instances": instances,
        "terrain": terrain,
        "cameras": cameras,
        "lights": lights,
        "warnings": list(warnings or []),
    }
    validate_manifest(manifest)
    return manifest


def load_manifest(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"Manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path} is not valid JSON: {exc}") from exc
    validate_manifest(manifest)
    return manifest


def write_manifest(path: Path, manifest: dict) -> Path:
    validate_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def validate_manifest(manifest: dict) -> None:
    if not isinstance(manifest, dict):
        raise ManifestError("Manifest must be a JSON object.")
    if manifest.get("version") != MANIFEST_VERSION:
        raise ManifestError(
            f"Unsupported manifest version {manifest.get('version')!r}. This importer reads version {MANIFEST_VERSION}."
        )
    units = manifest.get("units") or {}
    if (
        units.get("distance") != "centimeters"
        or units.get("up_axis") != "Z"
        or units.get("handedness") != "left"
    ):
        raise ManifestError("Manifest units must be centimeters, Z-up and left-handed.")
    for key in (
        "assets",
        "instances",
        "cameras",
        "lights",
        "fps",
        "frame_start",
        "frame_end",
        "mesh_format",
    ):
        if key not in manifest:
            raise ManifestError(f"Manifest is missing '{key}'.")

    asset_ids = []
    for asset in manifest["assets"]:
        _require(asset, ("id", "name", "file"), "asset")
        _require_relative(asset["file"], "asset file")
        textures = asset.get("textures") or {}
        for label, texture_path in textures.items():
            if texture_path:
                _require_relative(texture_path, f"texture {label}")
        asset_ids.append(asset["id"])
    if len(asset_ids) != len(set(asset_ids)):
        raise ManifestError("Asset ids must be unique.")
    known = set(asset_ids)

    for instance in manifest["instances"]:
        _require(
            instance,
            (
                "asset_id",
                "name",
                "location",
                "rotation_quat",
                "rotation_euler_deg",
                "scale",
            ),
            "instance",
        )
        if instance["asset_id"] not in known:
            raise ManifestError(
                f"Instance '{instance.get('name')}' references missing asset '{instance['asset_id']}'."
            )
        _vector(instance["location"], 3, "instance location")
        _vector(instance["rotation_quat"], 4, "instance rotation_quat")
        _vector(instance["scale"], 3, "instance scale")
        _euler(instance["rotation_euler_deg"], "instance rotation")

    terrain = manifest.get("terrain")
    if terrain is not None:
        _require(
            terrain, ("name", "heightmap", "resolution", "scale", "location"), "terrain"
        )
        _require_relative(terrain["heightmap"], "heightmap")
        if terrain.get("heightmap_raw"):
            _require_relative(terrain["heightmap_raw"], "heightmap_raw")
        if terrain.get("preview_mesh"):
            _require_relative(terrain["preview_mesh"], "preview_mesh")
        resolution = terrain["resolution"]
        if (
            not isinstance(resolution, (list, tuple))
            or len(resolution) != 2
            or int(resolution[0]) != int(resolution[1])
            or not is_legal_landscape_resolution(int(resolution[0]))
        ):
            raise ManifestError(
                f"Terrain resolution {resolution} is not a square size Unreal Landscape accepts. "
                "Use a size such as 127, 253, 505, 1009, 2017 or 4033."
            )
        _vector(terrain["location"], 3, "terrain location")
        scale = terrain["scale"]
        for axis in ("x", "y", "z"):
            if axis not in scale:
                raise ManifestError(f"Terrain scale is missing '{axis}'.")

    for camera in manifest["cameras"]:
        _require(
            camera, ("name", "sensor_width_mm", "sensor_height_mm", "frames"), "camera"
        )
        if not camera["frames"]:
            raise ManifestError(f"Camera '{camera['name']}' has no frames.")
        for frame in camera["frames"]:
            _require(
                frame,
                (
                    "frame",
                    "location",
                    "rotation_quat",
                    "rotation_euler_deg",
                    "focal_length_mm",
                ),
                f"camera '{camera['name']}' frame",
            )
            _vector(frame["location"], 3, "camera location")
            _euler(frame["rotation_euler_deg"], "camera rotation")

    for light in manifest["lights"]:
        _require(
            light,
            (
                "name",
                "source",
                "kind",
                "direction",
                "rotation_euler_deg",
                "intensity_lux",
                "color",
            ),
            "light",
        )
        _vector(light["direction"], 3, "light direction")
        _vector(light["color"], 3, "light color")
        _euler(light["rotation_euler_deg"], "light rotation")


def _require(record: dict, keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in record]
    if missing:
        raise ManifestError(f"{label} is missing {', '.join(missing)}.")


def _require_relative(path_value: str, label: str) -> None:
    if not isinstance(path_value, str) or not path_value:
        raise ManifestError(f"{label} path is empty.")
    if path_value.startswith(("/", "\\")) or (
        len(path_value) > 1 and path_value[1] == ":"
    ):
        raise ManifestError(
            f"{label} must be a path relative to the manifest, not an absolute path ({path_value})."
        )


def _vector(value, length: int, label: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ManifestError(f"{label} must have {length} numbers.")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ManifestError(f"{label} must be numeric.")


def _euler(value, label: str) -> None:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be an object with pitch, yaw and roll.")
    missing = [key for key in _EULER_KEYS if key not in value]
    if missing:
        raise ManifestError(f"{label} is missing {', '.join(missing)}.")
