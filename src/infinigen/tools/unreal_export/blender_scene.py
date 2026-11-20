# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Export the open Blender scene as unique meshes plus an Unreal manifest.

Run from Blender's Python (the ``bpy`` module or ``blender -b``):

    python -m infinigen.tools.unreal_export --blend scene.blend --output export_dir

Each unique mesh is written once, in Unreal centimeters. Copies are records in
``manifest.json``. Terrain becomes a 16-bit heightmap instead of one giant mesh.
"""

from __future__ import annotations

import hashlib
import logging
import traceback
from pathlib import Path

import bpy
import numpy as np
from PIL import Image

from infinigen.tools.unreal_export.coords import (
    blender_camera_to_unreal,
    blender_matrix_to_unreal,
    convert_points_to_unreal,
)
from infinigen.tools.unreal_export.heightmap import (
    terrain_record_from_mesh,
    write_heightmap_png,
    write_heightmap_raw,
)
from infinigen.tools.unreal_export.lights import (
    WATT_PER_SQ_M_TO_LUX,
    directional_light_from_direction,
    nishita_light,
    sun_lamp_travel_direction_blender,
)
from infinigen.tools.unreal_export.manifest import build_manifest, write_manifest

logger = logging.getLogger("infinigen.unreal_export")

DEFAULT_TERRAIN_NAMES = {
    "opaqueterain",
    "collectivetransparentterrain",
    "terrain",
    "landscape",
    "ground",
    "landtiles",
}
_SKIP_TOKENS = ("atmosphere", "kolecloud")
_PREVIEW_MAX_VERTS = 253


def export_blend_file(blend_path: Path, output_dir: Path, **kwargs) -> Path:
    """Open ``blend_path`` and export it. The call replaces the current Blender file."""
    blend_path = Path(blend_path)
    if not blend_path.is_file():
        raise FileNotFoundError(f"Blend file not found: {blend_path}")
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    return export_current_scene(output_dir, **kwargs)


def export_current_scene(
    output_dir: Path,
    *,
    mesh_format: str = "fbx",
    texture_resolution: int = 1024,
    heightmap_resolution: int = 505,
    bake: bool = True,
    bake_device: str = "CPU",
    terrain_names: list[str] | None = None,
    frame_step: int = 1,
) -> Path:
    """Export whatever file is currently open in Blender."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_log(output_dir / "export.log")
    if frame_step < 1:
        raise ValueError("--frame-step must be a positive integer.")
    mesh_format = mesh_format.lower().lstrip(".")
    if mesh_format == "usd":
        mesh_format = "usdc"
    if mesh_format not in {"fbx", "usdc", "usda"}:
        raise ValueError(
            f"Unsupported mesh format {mesh_format!r}. Use fbx, usd, usdc or usda."
        )

    scene = bpy.context.scene
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    gathered = _gather(depsgraph, set(terrain_names or []))
    warnings = list(gathered["warnings"])

    assets = []
    instances = []
    for index, group in enumerate(gathered["groups"], start=1):
        asset_id = f"asset_{index:04d}_{_safe_name(group['name'])}"
        asset_dir = output_dir / "assets"
        texture_dir = output_dir / "textures"
        asset_dir.mkdir(parents=True, exist_ok=True)
        texture_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = asset_dir / f"{asset_id}.{mesh_format}"
        textures = _export_unique_mesh(
            group,
            mesh_path,
            texture_dir,
            asset_id,
            mesh_format=mesh_format,
            texture_resolution=texture_resolution,
            bake=bake,
            bake_device=bake_device,
            warnings=warnings,
        )
        assets.append(
            {
                "id": asset_id,
                "name": group["name"],
                "file": mesh_path.relative_to(output_dir).as_posix(),
                "textures": textures,
                "baked": bool(
                    bake
                    and textures.get("base_color")
                    and "DIFFUSE" in textures.get("base_color", "")
                ),
                "vertex_count": int(len(group["vertices"])),
                "instance_count": len(group["instances"]),
            }
        )
        for instance in group["instances"]:
            converted = blender_matrix_to_unreal(instance["matrix"])
            if converted["shear"] > 1e-3:
                warnings.append(
                    f"Instance '{instance['name']}' has shear {converted['shear']:.4f}, "
                    "which Unreal cannot represent. Scale and rotation were used and the shear was dropped."
                )
            instances.append(
                {
                    "asset_id": asset_id,
                    "name": instance["name"],
                    "location": converted["location"],
                    "rotation_quat": converted["rotation_quat"],
                    "rotation_euler_deg": converted["rotation_euler_deg"],
                    "scale": converted["scale"],
                }
            )

    terrain = None
    if gathered["terrain_vertices"]:
        record = terrain_record_from_mesh(
            np.vstack(gathered["terrain_vertices"]),
            _offset_faces(gathered["terrain_faces"]),
            heightmap_resolution,
            gathered["terrain_name"],
        )
        encoded = record.pop("_encoded")
        heights = record.pop("_heights_m")
        png_path = output_dir / "terrain" / "heightmap.png"
        raw_path = output_dir / "terrain" / "heightmap.r16"
        preview_path = output_dir / "terrain" / "terrain_preview.fbx"
        write_heightmap_png(png_path, encoded)
        write_heightmap_raw(raw_path, encoded)
        _export_preview_mesh(heights, record, preview_path)
        record["heightmap"] = png_path.relative_to(output_dir).as_posix()
        record["heightmap_raw"] = raw_path.relative_to(output_dir).as_posix()
        record["preview_mesh"] = preview_path.relative_to(output_dir).as_posix()
        record["preview_mesh_space"] = "unreal_world_cm"
        if record["requested_resolution"] != record["resolution"][0]:
            warnings.append(
                f"Requested heightmap resolution {record['requested_resolution']} is not a Landscape size. "
                f"Wrote {record['resolution'][0]} instead."
            )
        terrain = record
    else:
        warnings.append(
            "No terrain mesh was found. Name the ground 'OpaqueTerrain' (Infinigen's default) "
            "or pass --terrain-name. Cameras, lights and instanced meshes were still exported."
        )

    frames = list(range(scene.frame_start, scene.frame_end + 1, frame_step))
    if not frames:
        frames = [scene.frame_current]
    cameras = _export_cameras(scene, frames)
    lights = _export_lights(scene, frames)
    fps = scene.render.fps / scene.render.fps_base
    manifest = build_manifest(
        assets=assets,
        instances=instances,
        terrain=terrain,
        cameras=cameras,
        lights=lights,
        fps=fps,
        frame_start=frames[0],
        frame_end=frames[-1],
        mesh_format=mesh_format,
        warnings=warnings,
    )
    path = write_manifest(output_dir / "manifest.json", manifest)
    logger.info(
        "Wrote %s (%d unique assets, %d instances)",
        path,
        len(assets),
        len(instances),
    )
    return path


def _gather(depsgraph, extra_terrain_names: set[str]) -> dict:
    cache: dict[int, dict | None] = {}
    groups: dict[str, dict] = {}
    terrain_vertices = []
    terrain_faces = []
    terrain_name = None
    warnings = []
    seen_real = set()

    for inst in depsgraph.object_instances:
        if not inst.show_self:
            continue
        obj = inst.object
        if obj is None or obj.type != "MESH":
            continue
        if _skip_name(obj.name):
            continue
        if obj.hide_render and not inst.is_instance:
            continue
        pointer = obj.original.data.as_pointer()
        if pointer not in cache:
            cache[pointer] = _local_mesh(obj, depsgraph)
        mesh = cache[pointer]
        if mesh is None:
            continue
        matrix = np.array(inst.matrix_world, dtype=np.float64)
        if _is_terrain(obj.name, extra_terrain_names):
            world = _transform_points(mesh["vertices"], matrix)
            offset = sum(len(block) for block in terrain_vertices)
            terrain_vertices.append(world)
            terrain_faces.append(
                [[index + offset for index in face] for face in mesh["faces"]]
            )
            terrain_name = terrain_name or obj.name
            continue
        # A real object is also visited once as a non-instance. Geometry-node
        # instances of that same object carry their own matrices; keep both only
        # when this entry is an actual placed copy.
        identity = (pointer, inst.is_instance, tuple(np.round(matrix[:, 3], 5)))
        if identity in seen_real:
            continue
        seen_real.add(identity)
        fingerprint = mesh["fingerprint"]
        group = groups.get(fingerprint)
        if group is None:
            group = {
                "name": obj.name,
                "vertices": mesh["vertices"],
                "faces": mesh["faces"],
                "source": obj.original,
                "base_color": _base_color(obj.original),
                "instances": [],
            }
            groups[fingerprint] = group
        group["instances"].append(
            {
                "name": f"{obj.name}_{len(group['instances']):04d}",
                "matrix": matrix,
            }
        )

    ordered = sorted(groups.values(), key=lambda group: group["name"])
    if not ordered and not terrain_vertices:
        warnings.append("The scene has no visible meshes to export.")
    return {
        "groups": ordered,
        "terrain_vertices": terrain_vertices,
        "terrain_faces": terrain_faces,
        "terrain_name": terrain_name or "Terrain",
        "warnings": warnings,
    }


def _local_mesh(obj, depsgraph) -> dict | None:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        if len(mesh.vertices) == 0 or len(mesh.polygons) == 0:
            return None
        flat = [0.0] * (len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", flat)
        vertices = np.asarray(flat, dtype=np.float64).reshape(-1, 3)
        faces = []
        for polygon in mesh.polygons:
            start = polygon.loop_start
            faces.append(
                [
                    mesh.loops[start + offset].vertex_index
                    for offset in range(polygon.loop_total)
                ]
            )
        return {
            "vertices": vertices,
            "faces": faces,
            "fingerprint": _fingerprint(vertices, faces),
        }
    finally:
        evaluated.to_mesh_clear()


def _export_unique_mesh(
    group,
    mesh_path: Path,
    texture_dir: Path,
    asset_id: str,
    *,
    mesh_format: str,
    texture_resolution: int,
    bake: bool,
    bake_device: str,
    warnings: list[str],
) -> dict:
    # Reflection from the axis change reverses winding. Flip faces so normals stay outward.
    faces = [list(reversed(face)) for face in group["faces"]]
    vertices = convert_points_to_unreal(group["vertices"])
    export_object = _create_mesh_object(asset_id, vertices, faces)
    textures: dict[str, str | None] = {
        "base_color": None,
        "roughness": None,
        "normal": None,
        "metallic": None,
    }
    try:
        _smart_uv(export_object)
        source = group["source"]
        if source.data.materials:
            export_object.data.materials.append(source.data.materials[0])
        if bake and export_object.data.materials:
            baked = _try_bake(
                export_object,
                texture_dir,
                asset_id,
                texture_resolution,
                bake_device,
                warnings,
            )
            textures.update(baked)
        if not textures.get("base_color"):
            color_path = texture_dir / f"{asset_id}_basecolor.png"
            _write_color_png(color_path, group["base_color"])
            _assign_image_material(export_object, color_path, f"{asset_id}_Mat")
            textures["base_color"] = color_path.relative_to(
                mesh_path.parents[1]
            ).as_posix()
        _export_file(export_object, mesh_path, mesh_format)
    finally:
        mesh_data = export_object.data
        bpy.data.objects.remove(export_object, do_unlink=True)
        if mesh_data.users == 0:
            bpy.data.meshes.remove(mesh_data)
    return textures


def _try_bake(
    obj,
    texture_dir: Path,
    asset_id: str,
    resolution: int,
    device: str,
    warnings: list[str],
) -> dict:
    try:
        from infinigen.tools.export import bake_object
    except ImportError as exc:
        warnings.append(
            "Texture baking needs infinigen.tools.export (and its imports: gin, trimesh). "
            f"Wrote a solid base color instead. Import error: {exc}"
        )
        return {}
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = device
    scene.cycles.samples = 1
    obj.hide_render = False
    obj.hide_viewport = False
    try:
        bake_object(obj, texture_dir, resolution, False, export_name=asset_id)
    except Exception as exc:  # noqa: BLE001 - bake failures should not lose the mesh export
        warnings.append(
            f"Baking '{asset_id}' failed ({exc}). Wrote a solid base color instead."
        )
        logger.debug(traceback.format_exc())
        return {}
    # bake_pass saves PNGs directly into the destination directory.
    found = {}
    for bake_type, key in (
        ("DIFFUSE", "base_color"),
        ("ROUGHNESS", "roughness"),
        ("NORMAL", "normal"),
        ("METAL", "metallic"),
        ("TRANSMISSION", "transmission"),
    ):
        candidate = texture_dir / f"{asset_id}_{bake_type}.png"
        if candidate.is_file():
            found[key] = candidate.relative_to(texture_dir.parent).as_posix()
    return found


def _export_preview_mesh(heights_m: np.ndarray, record: dict, path: Path) -> None:
    rows, columns = heights_m.shape
    stride = max(1, int(np.ceil(max(rows, columns) / _PREVIEW_MAX_VERTS)))
    row_ids = list(range(0, rows, stride))
    column_ids = list(range(0, columns, stride))
    if row_ids[-1] != rows - 1:
        row_ids.append(rows - 1)
    if column_ids[-1] != columns - 1:
        column_ids.append(columns - 1)
    location = record["location"]
    scale = record["scale"]
    vertices = []
    for row in row_ids:
        for column in column_ids:
            vertices.append(
                (
                    location[0] + column * scale["x"],
                    location[1] + row * scale["y"],
                    float(heights_m[row, column]) * 100.0,
                )
            )
    faces = []
    width = len(column_ids)
    for row in range(len(row_ids) - 1):
        for column in range(width - 1):
            origin = row * width + column
            faces.append([origin, origin + 1, origin + width + 1, origin + width])
    export_object = _create_mesh_object("InfinigenTerrainPreview", vertices, faces)
    try:
        _export_file(export_object, path, "fbx")
    finally:
        mesh_data = export_object.data
        bpy.data.objects.remove(export_object, do_unlink=True)
        if mesh_data.users == 0:
            bpy.data.meshes.remove(mesh_data)


def _export_cameras(scene, frames: list[int]) -> list[dict]:
    cameras = []
    objects = [
        obj for obj in scene.objects if obj.type == "CAMERA" and not obj.hide_render
    ]
    if not objects and scene.camera is not None:
        objects = [scene.camera]
    for obj in objects:
        samples = []
        for frame in frames:
            scene.frame_set(frame)
            location, quat, euler = blender_camera_to_unreal(
                np.array(obj.matrix_world, dtype=np.float64)
            )
            samples.append(
                {
                    "frame": int(frame),
                    "location": location,
                    "rotation_quat": quat,
                    "rotation_euler_deg": euler,
                    "focal_length_mm": float(obj.data.lens),
                }
            )
        cameras.append(
            {
                "name": obj.name,
                "sensor_width_mm": float(obj.data.sensor_width),
                "sensor_height_mm": float(obj.data.sensor_height),
                "frames": samples,
            }
        )
    return cameras


def _export_lights(scene, frames: list[int]) -> list[dict]:
    lights = []
    for obj in scene.objects:
        if obj.type != "LIGHT" or obj.data.type != "SUN":
            continue
        samples = []
        for frame in frames:
            scene.frame_set(frame)
            matrix = np.array(obj.matrix_world, dtype=np.float64)
            travel = sun_lamp_travel_direction_blender(matrix)
            color = tuple(float(channel) for channel in obj.data.color)
            light = directional_light_from_direction(
                travel,
                float(obj.data.energy) * WATT_PER_SQ_M_TO_LUX,
                color,
                obj.name,
                "sun_lamp",
            )
            light["frame"] = int(frame)
            samples.append(light)
        merged = dict(samples[0])
        merged.pop("frame", None)
        if len(samples) > 1 and _light_samples_differ(samples):
            merged["frames"] = samples
        lights.append(merged)

    world = scene.world
    sky, background = _find_nishita(
        world.node_tree if world and world.use_nodes else None
    )
    if sky is not None:
        samples = []
        for frame in frames:
            scene.frame_set(frame)
            strength = 1.0
            if background is not None and background.inputs.get("Strength") is not None:
                strength = float(background.inputs["Strength"].default_value)
            light = nishita_light(
                float(sky.sun_elevation),
                float(sky.sun_rotation),
                float(sky.sun_intensity),
                strength,
            )
            light["frame"] = int(frame)
            samples.append(light)
        merged = dict(samples[0])
        merged.pop("frame", None)
        if len(samples) > 1 and _light_samples_differ(samples):
            merged["frames"] = samples
        lights.append(merged)
    return lights


def _find_nishita(node_tree):
    if node_tree is None:
        return None, None
    sky = None
    background = None
    for node in node_tree.nodes:
        if node.type == "TEX_SKY" and getattr(node, "sky_type", "") == "NISHITA":
            sky = node
        elif node.type == "BACKGROUND":
            background = node
        elif node.type == "GROUP" and getattr(node, "node_tree", None):
            child_sky, child_background = _find_nishita(node.node_tree)
            sky = sky or child_sky
            background = background or child_background
    return sky, background


def _light_samples_differ(samples: list[dict]) -> bool:
    first = samples[0]
    for sample in samples[1:]:
        if (
            sample["direction"] != first["direction"]
            or sample["intensity_lux"] != first["intensity_lux"]
        ):
            return True
    return False


def _create_mesh_object(name: str, vertices, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(
        [tuple(map(float, vertex)) for vertex in vertices],
        [],
        [list(map(int, face)) for face in faces],
    )
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _smart_uv(obj) -> None:
    if obj.data.uv_layers:
        return
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    obj.data.uv_layers.new(name="UVMap")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    try:
        bpy.ops.uv.smart_project(angle_limit=0.7)
    except RuntimeError:
        logger.info("UV unwrap failed for %s", obj.name)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


def _assign_image_material(obj, image_path: Path, name: str) -> None:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        material.node_tree.links.new(
            nodes["Material Output"].inputs["Surface"], bsdf.outputs["BSDF"]
        )
    image_node = nodes.new("ShaderNodeTexImage")
    image_node.image = bpy.data.images.load(str(image_path))
    material.node_tree.links.new(bsdf.inputs["Base Color"], image_node.outputs["Color"])
    obj.data.materials.clear()
    obj.data.materials.append(material)


def _export_file(obj, path: Path, mesh_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.hide_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    if mesh_format == "fbx":
        # Vertices are already Unreal centimeters. These flags keep Blender from
        # applying a second axis conversion or a meters-to-centimeters scale.
        bpy.ops.export_scene.fbx(
            filepath=str(path),
            use_selection=True,
            global_scale=1.0,
            apply_unit_scale=False,
            apply_scale_options="FBX_SCALE_UNITS",
            use_space_transform=False,
            bake_space_transform=False,
            axis_forward="Y",
            axis_up="Z",
            use_mesh_modifiers=False,
            mesh_smooth_type="FACE",
            add_leaf_bones=False,
            path_mode="COPY",
            embed_textures=False,
            object_types={"MESH"},
            bake_anim=False,
        )
    else:
        bpy.ops.wm.usd_export(
            filepath=str(path),
            selected_objects_only=True,
            export_animation=False,
            export_materials=True,
            convert_orientation=False,
            generate_preview_surface=True,
        )


def _write_color_png(path: Path, color) -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :] = [
        int(round(max(0.0, min(1.0, float(channel))) * 255.0)) for channel in color[:3]
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)


def _base_color(obj) -> tuple[float, float, float, float]:
    materials = getattr(obj.data, "materials", None)
    if not materials:
        return (0.8, 0.8, 0.8, 1.0)
    material = materials[0]
    if (
        material
        and material.use_nodes
        and material.node_tree
        and material.node_tree.nodes.get("Principled BSDF")
    ):
        value = (
            material.node_tree.nodes["Principled BSDF"]
            .inputs["Base Color"]
            .default_value
        )
        return tuple(float(channel) for channel in value)
    return (0.8, 0.8, 0.8, 1.0)


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (matrix @ homogeneous.T).T[:, :3]


def _offset_faces(blocks: list[list[list[int]]]) -> list[list[int]]:
    return [face for block in blocks for face in block]


def _fingerprint(vertices: np.ndarray, faces) -> str:
    digest = hashlib.sha1()
    quantized = np.round(np.asarray(vertices, dtype=np.float64), 5).astype(np.float32)
    digest.update(quantized.tobytes())
    for face in faces:
        digest.update(np.asarray(face, dtype=np.int32).tobytes())
        digest.update(b"|")
    return digest.hexdigest()[:16]


def _is_terrain(name: str, extra: set[str]) -> bool:
    key = name.lower().split(".")[0]
    names = set(DEFAULT_TERRAIN_NAMES)
    names.update(item.lower() for item in extra)
    return key in names or key.endswith("terrain")


def _skip_name(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in _SKIP_TOKENS)


def _safe_name(name: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in name
    )
    return cleaned.strip("_")[:40] or "asset"


def _configure_log(path: Path) -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, mode="w")
    file_handler.setFormatter(formatter)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream)
