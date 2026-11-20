# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""End-to-end export of a synthetic scene. A full Infinigen nature solve is not required."""

import math

import bpy
import numpy as np
import pytest
from mathutils import Euler

from infinigen.tools.unreal_export.blender_scene import export_current_scene
from infinigen.tools.unreal_export.lights import (
    directional_light_from_direction,
    sun_lamp_travel_direction_blender,
)
from infinigen.tools.unreal_export.manifest import load_manifest


def test_synthetic_scene_exports_one_mesh_per_unique_asset(tmp_path):
    _build_scene()
    manifest_path = export_current_scene(
        tmp_path,
        mesh_format="fbx",
        bake=False,
        heightmap_resolution=127,
        frame_step=1,
    )
    manifest = load_manifest(manifest_path)

    assert len(manifest["assets"]) == 1
    assert manifest["assets"][0]["instance_count"] == 2
    assert len(manifest["instances"]) == 2
    assert manifest["instances"][0]["asset_id"] == manifest["instances"][1]["asset_id"]
    assert (tmp_path / manifest["assets"][0]["file"]).is_file()
    assert (tmp_path / manifest["assets"][0]["textures"]["base_color"]).is_file()

    locations = {
        tuple(np.round(instance["location"], 3)) for instance in manifest["instances"]
    }
    # Blender (0, 0, 0) and (3, 0, 0) meters -> Unreal (0, 0, 0) and (0, 300, 0) cm.
    assert (0.0, 0.0, 0.0) in locations
    assert (0.0, 300.0, 0.0) in locations

    terrain = manifest["terrain"]
    assert terrain["resolution"] == [127, 127]
    assert terrain["sections_per_component"] == 2
    assert terrain["quads_per_section"] == 63
    assert (tmp_path / terrain["heightmap"]).is_file()
    assert (tmp_path / terrain["heightmap_raw"]).stat().st_size == 127 * 127 * 2
    from PIL import Image

    height = np.array(Image.open(tmp_path / terrain["heightmap"]))
    # z = 0.1 * Blender X, and row 0 is minimum Blender X, so the first row is lower.
    assert float(height[0].mean()) < float(height[-1].mean())

    camera = manifest["cameras"][0]
    assert camera["sensor_width_mm"] == 36.0
    assert len(camera["frames"]) == 2
    assert camera["frames"][0]["focal_length_mm"] == 35.0
    np.testing.assert_allclose(
        camera["frames"][0]["location"], [-500.0, 0.0, 200.0], atol=1e-3
    )
    np.testing.assert_allclose(
        camera["frames"][1]["location"], [-500.0, 100.0, 200.0], atol=1e-3
    )
    assert manifest["fps"] == 24.0

    sun = next(light for light in manifest["lights"] if light["source"] == "sun_lamp")
    assert sun["intensity_lux"] == 500.0
    expected = directional_light_from_direction(
        sun_lamp_travel_direction_blender(
            np.array(bpy.data.objects["Sun"].matrix_world)
        ),
        500.0,
        (1.0, 1.0, 1.0),
        "Sun",
        "sun_lamp",
    )
    np.testing.assert_allclose(sun["direction"], expected["direction"], atol=1e-5)

    _assert_fbx_roundtrip(
        tmp_path / manifest["assets"][0]["file"], expected_vertex=(25.0, 100.0, 50.0)
    )
    _assert_fbx_roundtrip_bounds(
        tmp_path / terrain["preview_mesh"],
        min_z=-50.0,
        max_z=50.0,
    )


def test_geometry_nodes_instances_share_one_asset(tmp_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    prototype = bpy.context.active_object
    prototype.name = "Pebble"
    for collection in list(prototype.users_collection):
        collection.objects.unlink(prototype)

    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
    scatter = bpy.context.active_object
    scatter.name = "PebbleScatter"
    modifier = scatter.modifiers.new("Scatter", "NODES")
    tree = bpy.data.node_groups.new("PebbleScatter", "GeometryNodeTree")
    modifier.node_group = tree
    tree.interface.new_socket(
        name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    mesh_line = tree.nodes.new("GeometryNodeMeshLine")
    mesh_line.inputs["Count"].default_value = 4
    mesh_line.inputs["Offset"].default_value = (2.0, 0.0, 0.0)
    object_info = tree.nodes.new("GeometryNodeObjectInfo")
    object_info.inputs["Object"].default_value = prototype
    object_info.inputs["As Instance"].default_value = True
    instance = tree.nodes.new("GeometryNodeInstanceOnPoints")
    output = tree.nodes.new("NodeGroupOutput")
    tree.links.new(mesh_line.outputs["Mesh"], instance.inputs["Points"])
    tree.links.new(object_info.outputs["Geometry"], instance.inputs["Instance"])
    tree.links.new(instance.outputs["Instances"], output.inputs["Geometry"])

    manifest = load_manifest(
        export_current_scene(tmp_path, bake=False, heightmap_resolution=127)
    )
    pebble_assets = [asset for asset in manifest["assets"] if "Pebble" in asset["name"]]
    assert len(pebble_assets) == 1
    assert pebble_assets[0]["instance_count"] == 4
    assert manifest["terrain"] is None
    assert any("No terrain" in warning for warning in manifest["warnings"])


def test_cycles_bake_uses_existing_bake_path(tmp_path):
    """Cycles bake goes through infinigen.tools.export.bake_object when that import works."""
    pytest.importorskip("gin")
    pytest.importorskip("trimesh")
    try:
        from infinigen.tools.export import bake_object
    except ImportError as exc:
        pytest.skip(f"bake_object is not importable in this environment: {exc}")
    assert bake_object is not None

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.active_object
    material = bpy.data.materials.new("BakeMat")
    material.use_nodes = True
    cube.data.materials.append(material)

    manifest = load_manifest(
        export_current_scene(
            tmp_path,
            bake=True,
            bake_device="CPU",
            texture_resolution=32,
            heightmap_resolution=127,
        )
    )
    asset = manifest["assets"][0]
    assert asset["baked"] is True
    expected = {
        "base_color": "DIFFUSE",
        "roughness": "ROUGHNESS",
        "normal": "NORMAL",
    }
    for key, bake_type in expected.items():
        relative = asset["textures"][key]
        assert relative.endswith(f"_{bake_type}.png")
        written = tmp_path / relative
        assert written.is_file()
        assert written.stat().st_size > 0


def _build_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    rock = bpy.context.active_object
    rock.name = "Rock"
    rock.data.vertices[0].co = (1.0, 0.25, 0.5)
    material = bpy.data.materials.new("RockMat")
    material.use_nodes = True
    material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
        0.2,
        0.4,
        0.6,
        1.0,
    )
    rock.data.materials.append(material)

    bpy.ops.object.select_all(action="DESELECT")
    rock.select_set(True)
    bpy.context.view_layer.objects.active = rock
    bpy.ops.object.duplicate(linked=True)
    copy = bpy.context.active_object
    copy.location = (3.0, 0.0, 0.0)

    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=4, y_subdivisions=4, size=10, location=(0, 0, 0)
    )
    terrain = bpy.context.active_object
    terrain.name = "OpaqueTerrain"
    for vertex in terrain.data.vertices:
        vertex.co.z = 0.1 * vertex.co.x

    bpy.ops.object.camera_add(location=(0.0, -5.0, 2.0))
    camera = bpy.context.active_object
    camera.data.lens = 35.0
    camera.data.sensor_width = 36.0
    camera.data.sensor_height = 24.0
    camera.rotation_euler = Euler((math.radians(70.0), 0.0, 0.0), "XYZ")
    bpy.context.scene.camera = camera
    camera.keyframe_insert(data_path="location", frame=1)
    camera.location.x = 1.0
    camera.keyframe_insert(data_path="location", frame=2)
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 2
    bpy.context.scene.render.fps = 24

    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 10.0))
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.data.energy = 5.0
    sun.rotation_euler = Euler((0.0, math.radians(45.0), 0.0), "XYZ")


def _assert_fbx_roundtrip(path, expected_vertex):
    before = {obj.name for obj in bpy.data.objects}
    bpy.ops.import_scene.fbx(filepath=str(path))
    imported = [
        obj for obj in bpy.data.objects if obj.name not in before and obj.type == "MESH"
    ]
    assert imported, f"FBX import produced no mesh for {path}"
    vertices = np.array(
        [tuple(vertex.co) for obj in imported for vertex in obj.data.vertices]
    )
    distances = np.linalg.norm(vertices - np.array(expected_vertex), axis=1)
    assert distances.min() < 1e-3
    # The axis reflection plus the reversed winding should keep the cube's normals outward.
    outward = []
    for obj in imported:
        center = np.mean([tuple(vertex.co) for vertex in obj.data.vertices], axis=0)
        for polygon in obj.data.polygons:
            face_center = np.array(polygon.center)
            outward.append(
                float(np.dot(np.array(polygon.normal), face_center - center))
            )
    assert np.mean(outward) > 0.0
    for obj in imported:
        bpy.data.objects.remove(obj, do_unlink=True)


def _assert_fbx_roundtrip_bounds(path, min_z, max_z):
    before = {obj.name for obj in bpy.data.objects}
    bpy.ops.import_scene.fbx(filepath=str(path))
    imported = [
        obj for obj in bpy.data.objects if obj.name not in before and obj.type == "MESH"
    ]
    assert imported
    z_values = [vertex.co.z for obj in imported for vertex in obj.data.vertices]
    assert min(z_values) == pytest_approx(min_z)
    assert max(z_values) == pytest_approx(max_z)
    for obj in imported:
        bpy.data.objects.remove(obj, do_unlink=True)


def pytest_approx(value, tolerance=1e-2):
    class _Approx:
        def __eq__(self, other):
            return abs(float(other) - value) <= tolerance

        def __repr__(self):
            return f"approx {value}"

    return _Approx()
