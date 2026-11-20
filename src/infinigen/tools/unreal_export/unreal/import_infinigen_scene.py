# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Build an Infinigen export inside Unreal Editor 5.8.

Enable Edit → Plugins → Python Editor Script Plugin, then either:

    py "<repo>/src/infinigen/tools/unreal_export/unreal/import_infinigen_scene.py" --manifest "D:/export/manifest.json"

or Tools → Execute Python Script and pass the same arguments.

Meshes in the export are already Unreal centimeters (Z-up, left-handed). This
script imports them with scene-axis and scene-unit conversion turned off.
Copies are placed on one instanced component per unique mesh.

Stock Unreal Python cannot create a Landscape component grid. When that is the
case the script places everything else, imports a preview terrain mesh, and
raises UnrealImportError with the exact Landscape Mode settings to type in.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _run(args)
    except Exception as exc:  # noqa: BLE001 - the editor log should show one clear error
        if exc.__class__.__name__ in {
            "ManifestError",
            "UnrealImportError",
            "FileNotFoundError",
        }:
            print(f"\nERROR: {exc}\n", file=sys.stderr)
            return 2
        traceback.print_exc()
        print(
            "\nERROR: The Unreal import failed. The traceback above is the direct cause. "
            "Fix that and run the script again; already imported assets are replaced.\n",
            file=sys.stderr,
        )
        return 2
    return 0


def _run(args) -> None:
    support = _load_support()
    manifest_path = Path(args.manifest).resolve()
    try:
        manifest = support.load_manifest(manifest_path)
    except support.ManifestError:
        raise
    plan = support.plan_import(
        manifest,
        nanite=not args.no_nanite,
        generate_lods=args.generate_lods,
        placement=args.placement,
        content_root=args.content_root,
    )
    try:
        import unreal
    except ImportError as exc:
        raise support.UnrealImportError(
            "This script must be run inside Unreal Editor 5.8 with the Python Editor Script Plugin enabled. "
            "The 'unreal' module was not found, so nothing was imported.\n"
            "In the editor: Edit → Plugins → Python Editor Script Plugin → restart, then run:\n"
            f'py "{Path(__file__).resolve()}" --manifest "{manifest_path}"'
        ) from exc

    root = manifest_path.parent
    content_root = plan["content_root"]
    _ensure_directory(unreal, content_root)
    _clear_previous(unreal, keep_labels={args.landscape_actor})

    mesh_by_asset = {}
    for group in plan["instance_groups"]:
        asset = group["asset"]
        mesh_file = _existing(root, asset["file"])
        destination = f"{content_root}/Meshes"
        mesh = _import_static_mesh(
            unreal,
            mesh_file,
            destination,
            asset["id"],
            plan["fbx_options"],
            plan["mesh_format"],
        )
        _set_nanite(unreal, mesh, plan["nanite"])
        if plan["generate_lods"]:
            _generate_lods(unreal, mesh)
        material = _create_material(unreal, content_root, asset, root)
        if material is not None:
            try:
                mesh.set_material(0, material)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"Could not assign M_{asset['id']} automatically ({exc}). "
                    "Assign that material on the static mesh in the editor."
                )
        unreal.EditorAssetLibrary.save_loaded_asset(mesh)
        mesh_by_asset[asset["id"]] = mesh
        if group["count"] == 0:
            continue
        _place_group(unreal, group, mesh, plan["placement"])

    _place_lights(unreal, plan["lights"])
    _place_cameras_and_sequence(unreal, plan, content_root)
    landscape_error = _place_terrain(unreal, plan, root, args.landscape_actor)
    _save_level(unreal)
    if landscape_error:
        raise support.UnrealImportError(landscape_error)
    print(
        "Infinigen import finished. Open the InfinigenShot level sequence and render it with Movie Render Queue."
    )


def _import_static_mesh(
    unreal,
    filename: Path,
    destination: str,
    asset_name: str,
    options: dict,
    mesh_format: str,
):
    _ensure_directory(unreal, destination)
    task = unreal.AssetImportTask()
    task.filename = str(filename)
    task.destination_path = destination
    task.destination_name = asset_name
    task.automated = True
    task.replace_existing = True
    task.save = True
    if mesh_format == "fbx":
        task.options = _fbx_options(unreal, options)
    elif hasattr(unreal, "UsdStageImportOptions"):
        usd_options = unreal.UsdStageImportOptions()
        # The file is already in Unreal centimeters. Block a second axis conversion when the property exists.
        for prop, value in (
            ("import_actors", False),
            ("import_geometry", True),
            (
                "existing_asset_policy",
                getattr(unreal, "UsdExistingAssetPolicy", None)
                and unreal.UsdExistingAssetPolicy.REPLACE,
            ),
        ):
            if value is not None and hasattr(usd_options, prop):
                setattr(usd_options, prop, value)
        task.options = usd_options
    else:
        raise _error(
            unreal,
            f"Mesh format {mesh_format!r} needs the USD importer plugin, which is not loaded. Re-export with --format fbx.",
        )
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    paths = list(task.imported_object_paths or [])
    if not paths:
        loaded = unreal.EditorAssetLibrary.load_asset(f"{destination}/{asset_name}")
        if loaded:
            return _as_static_mesh(unreal, loaded, filename)
        raise _error(
            unreal,
            f"Unreal did not import a mesh from {filename}. "
            "Confirm the Interchange/FBX importer is enabled and that the file is not empty.",
        )
    asset = unreal.EditorAssetLibrary.load_asset(paths[0])
    return _as_static_mesh(unreal, asset, filename)


def _fbx_options(unreal, options: dict):
    ui = unreal.FbxImportUI()
    ui.import_mesh = True
    ui.import_as_skeletal = False
    ui.import_materials = False
    ui.import_textures = False
    ui.import_animations = False
    data = ui.static_mesh_import_data
    _set(
        data,
        "combine_meshes",
        options["combine_meshes"],
        "FbxStaticMeshImportData.combine_meshes",
    )
    _set(
        data,
        "auto_generate_collision",
        options["auto_generate_collision"],
        "auto_generate_collision",
    )
    _set(
        data,
        "generate_lightmap_u_vs",
        options["generate_lightmap_uvs"],
        "generate_lightmap_u_vs",
    )
    _set(data, "convert_scene", options["convert_scene"], "convert_scene")
    _set(
        data, "convert_scene_unit", options["convert_scene_unit"], "convert_scene_unit"
    )
    _set(
        data,
        "import_uniform_scale",
        options["import_uniform_scale"],
        "import_uniform_scale",
    )
    return ui


def _as_static_mesh(unreal, asset, filename: Path):
    if isinstance(asset, unreal.StaticMesh):
        return asset
    raise _error(
        unreal,
        f"Import of {filename.name} produced {asset.get_class().get_name()} instead of a StaticMesh. "
        "Re-export that asset as FBX and import again.",
    )


def _set_nanite(unreal, mesh, enabled: bool) -> None:
    try:
        settings = mesh.get_editor_property("nanite_settings")
        settings.enabled = bool(enabled)
        mesh.set_editor_property("nanite_settings", settings)
    except Exception as exc:  # noqa: BLE001
        raise _error(
            unreal,
            "Could not change Nanite on the imported mesh "
            f"({exc}). Re-run with --no-nanite, or enable Nanite on the static mesh by hand.",
        ) from exc


def _generate_lods(unreal, mesh) -> None:
    subsystem = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    if not hasattr(subsystem, "set_lod_count"):
        raise _error(
            unreal,
            "LOD generation needs StaticMeshEditorSubsystem.set_lod_count, which this editor build does not expose. "
            "Re-run without --generate-lods and add LODs in the Static Mesh editor (LOD 1 at 50%, LOD 2 at 25%, LOD 3 at 12.5%).",
        )
    subsystem.set_lod_count(mesh, 4)
    library = getattr(unreal, "EditorStaticMeshLibrary", None)
    if library is None or not hasattr(unreal, "MeshReductionSettings"):
        print(
            "Set the LOD count to 4. Reduction settings are not exposed; set triangle percentages in the mesh editor."
        )
        return
    for lod, percent in ((1, 0.5), (2, 0.25), (3, 0.125)):
        settings = unreal.MeshReductionSettings()
        settings.percent_triangles = percent
        library.set_lod_reduction_settings(mesh, lod, settings)


def _create_material(unreal, content_root: str, asset: dict, root: Path):
    textures = asset.get("textures") or {}
    if not any(textures.values()):
        return None
    destination = f"{content_root}/Materials"
    _ensure_directory(unreal, destination)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    material = tools.create_asset(
        f"M_{asset['id']}", destination, unreal.Material, unreal.MaterialFactoryNew()
    )
    library = unreal.MaterialEditingLibrary
    row = 0
    for key, prop_name, sampler in (
        ("base_color", "MP_BASE_COLOR", None),
        ("roughness", "MP_ROUGHNESS", None),
        ("metallic", "MP_METALLIC", None),
        ("normal", "MP_NORMAL", "SAMPLERTYPE_NORMAL"),
    ):
        relative = textures.get(key)
        if not relative:
            continue
        texture_file = _existing(root, relative)
        texture = _import_texture(
            unreal,
            texture_file,
            f"{content_root}/Textures",
            f"T_{asset['id']}_{key}",
            srgb=key == "base_color",
        )
        expression = library.create_material_expression(
            material, unreal.MaterialExpressionTextureSample, -400, row
        )
        expression.texture = texture
        if sampler and hasattr(unreal, "MaterialSamplerType"):
            expression.sampler_type = getattr(unreal.MaterialSamplerType, sampler)
        prop = getattr(unreal.MaterialProperty, prop_name)
        library.connect_material_property(expression, "RGB", prop)
        row += 280
    library.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material)
    return material


def _import_texture(
    unreal, filename: Path, destination: str, asset_name: str, srgb: bool
):
    _ensure_directory(unreal, destination)
    task = unreal.AssetImportTask()
    task.filename = str(filename)
    task.destination_path = destination
    task.destination_name = asset_name
    task.automated = True
    task.replace_existing = True
    task.save = True
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    paths = list(task.imported_object_paths or [])
    asset_path = paths[0] if paths else f"{destination}/{asset_name}"
    texture = unreal.EditorAssetLibrary.load_asset(asset_path)
    if texture is None:
        raise _error(unreal, f"Could not import texture {filename}.")
    if hasattr(texture, "srgb"):
        texture.srgb = bool(srgb)
    return texture


def _place_group(unreal, group: dict, mesh, placement: str) -> None:
    if placement == "foliage":
        _place_foliage(unreal, group, mesh)
        return
    component_class = (
        unreal.HierarchicalInstancedStaticMeshComponent
        if placement == "hism"
        else unreal.InstancedStaticMeshComponent
    )
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = subsystem.spawn_actor_from_class(unreal.Actor, unreal.Vector(0.0, 0.0, 0.0))
    actor.set_actor_label(f"Infinigen_{group['asset_id']}")
    _tag(actor)
    component = actor.add_component_by_class(
        component_class, False, unreal.Transform(), False
    )
    component.set_static_mesh(mesh)
    transforms = [
        _instance_transform(unreal, instance) for instance in group["instances"]
    ]
    if hasattr(component, "add_instances"):
        component.add_instances(transforms, True, True)
    else:
        for transform in transforms:
            component.add_instance(transform)


def _place_foliage(unreal, group: dict, mesh) -> None:
    if not hasattr(unreal, "FoliageType_InstancedStaticMesh"):
        raise _error(
            unreal,
            "Foliage placement needs FoliageType_InstancedStaticMesh, which this editor does not expose. "
            "Re-run with --placement hism. That still places every copy on one instanced component.",
        )
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    destination = "/Game/Infinigen/Foliage"
    _ensure_directory(unreal, destination)
    factory = getattr(unreal, "FoliageType_InstancedStaticMeshFactory", None)
    if factory is None:
        raise _error(
            unreal,
            "FoliageType_InstancedStaticMeshFactory is not exposed. Re-run with --placement hism.",
        )
    foliage_type = tools.create_asset(
        f"FT_{group['asset_id']}",
        destination,
        unreal.FoliageType_InstancedStaticMesh,
        factory(),
    )
    foliage_type.set_editor_property("mesh", mesh)
    world = _editor_world(unreal)
    actor_class = getattr(unreal, "InstancedFoliageActor", None)
    getter = (
        getattr(actor_class, "get_instanced_foliage_actor_for_current_level", None)
        if actor_class
        else None
    )
    if getter is None:
        raise _error(
            unreal,
            "InstancedFoliageActor.get_instanced_foliage_actor_for_current_level is not available. "
            "The foliage type asset was created. Paint with the Foliage tool, or re-run with --placement hism.",
        )
    foliage_actor = getter(world)
    if not hasattr(foliage_actor, "add_instances"):
        raise _error(
            unreal,
            "This editor cannot add foliage instances from Python (no add_instances on the foliage actor). "
            f"The foliage type is {foliage_type.get_path_name()}. Re-run with --placement hism to place the copies.",
        )
    foliage_actor.add_instances(
        foliage_type,
        [_instance_transform(unreal, instance) for instance in group["instances"]],
    )


def _place_lights(unreal, lights: list[dict]) -> None:
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for index, light in enumerate(lights):
        rotator = _rotator(unreal, light["rotation_euler_deg"])
        actor = subsystem.spawn_actor_from_class(
            unreal.DirectionalLight, unreal.Vector(0.0, 0.0, 500.0), rotator
        )
        actor.set_actor_label(f"Infinigen_{light['name']}")
        _tag(actor)
        component = actor.light_component
        component.set_editor_property("intensity", float(light["intensity_lux"]))
        component.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
        color = light["color"]
        if hasattr(unreal, "LinearColor"):
            component.set_editor_property(
                "light_color", unreal.LinearColor(color[0], color[1], color[2], 1.0)
            )
        if index == 0:
            _set_if(component, "atmosphere_sun_light", True)
    if lights and hasattr(unreal, "SkyAtmosphere"):
        sky = subsystem.spawn_actor_from_class(
            unreal.SkyAtmosphere, unreal.Vector(0.0, 0.0, 0.0)
        )
        sky.set_actor_label("InfinigenSkyAtmosphere")
        _tag(sky)
    if lights and hasattr(unreal, "SkyLight"):
        sky_light = subsystem.spawn_actor_from_class(
            unreal.SkyLight, unreal.Vector(0.0, 0.0, 0.0)
        )
        sky_light.set_actor_label("InfinigenSkyLight")
        _tag(sky_light)
        component = sky_light.light_component
        _set_if(component, "real_time_capture", True)
        _set_if(component, "mobility", unreal.ComponentMobility.MOVABLE)
        if hasattr(component, "recapture_sky"):
            component.recapture_sky()


def _place_cameras_and_sequence(unreal, plan: dict, content_root: str) -> None:
    if not plan["cameras"]:
        print("The manifest has no cameras, so no level sequence was created.")
        return
    camera_info = plan["cameras"][0]
    first = camera_info["frames"][0]
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = subsystem.spawn_actor_from_class(
        unreal.CineCameraActor,
        unreal.Vector(*first["location"]),
        _rotator(unreal, first["rotation_euler_deg"]),
    )
    actor.set_actor_label(f"Infinigen_{camera_info['name']}")
    _tag(actor)
    component = actor.get_cine_camera_component()
    component.current_focal_length = float(first["focal_length_mm"])
    filmback = component.filmback
    filmback.sensor_width = float(camera_info["sensor_width_mm"])
    filmback.sensor_height = float(camera_info["sensor_height_mm"])
    component.filmback = filmback

    tools = unreal.AssetToolsHelpers.get_asset_tools()
    sequence_path = f"{content_root}/InfinigenShot"
    if unreal.EditorAssetLibrary.does_asset_exist(sequence_path):
        unreal.EditorAssetLibrary.delete_asset(sequence_path)
    sequence = tools.create_asset(
        "InfinigenShot",
        content_root,
        unreal.LevelSequence,
        unreal.LevelSequenceFactoryNew(),
    )
    fps = float(plan["fps"])
    if abs(fps - round(fps)) < 1e-3:
        sequence.set_display_rate(unreal.FrameRate(int(round(fps)), 1))
    else:
        sequence.set_display_rate(unreal.FrameRate(int(round(fps * 1000)), 1000))
    sequence.set_playback_start(int(plan["frame_start"]))
    sequence.set_playback_end(int(plan["frame_end"]))
    binding = sequence.add_possessable(actor)
    track = binding.add_track(unreal.MovieScene3DTransformTrack)
    section = track.add_section()
    section.set_range(int(plan["frame_start"]), int(plan["frame_end"]) + 1)
    channels = list(section.get_all_channels())
    if len(channels) < 9:
        raise _error(
            unreal,
            f"The transform section exposed {len(channels)} channels; 9 are required "
            "(location XYZ, roll, pitch, yaw, scale XYZ). "
            f"Camera '{camera_info['name']}' was created with focal length {first['focal_length_mm']} mm. "
            "Key the transform by hand from manifest.json cameras[].frames.",
        )
    for frame, values in zip(camera_info["frames"], camera_info["channels"]):
        frame_number = unreal.FrameNumber(int(frame["frame"]))
        for channel, value in zip(channels[:9], values):
            channel.add_key(frame_number, float(value))
    _key_focal_length(unreal, sequence, component, camera_info)
    unreal.EditorAssetLibrary.save_loaded_asset(sequence)
    _queue_movie_render(unreal, sequence)


def _key_focal_length(unreal, sequence, component, camera_info: dict) -> None:
    focals = [float(frame["focal_length_mm"]) for frame in camera_info["frames"]]
    if len(set(focals)) == 1:
        return
    try:
        binding = sequence.add_possessable(component)
        track = binding.add_track(unreal.MovieSceneFloatTrack)
        track.set_property_name_and_path("CurrentFocalLength", "CurrentFocalLength")
        section = track.add_section()
        section.set_range(
            camera_info["frames"][0]["frame"], camera_info["frames"][-1]["frame"] + 1
        )
        channel = section.get_all_channels()[0]
        for frame in camera_info["frames"]:
            channel.add_key(
                unreal.FrameNumber(int(frame["frame"])), float(frame["focal_length_mm"])
            )
    except Exception as exc:  # noqa: BLE001
        print(
            "Could not key focal length on the cine camera "
            f"({exc}). The component uses {focals[0]} mm. "
            "If the manifest frames change focal length, key CurrentFocalLength on the camera component."
        )


def _queue_movie_render(unreal, sequence) -> None:
    if not hasattr(unreal, "MoviePipelineQueueSubsystem"):
        print(
            "Movie Render Queue is not enabled. The level sequence is /Game/Infinigen/InfinigenShot. "
            "Enable the Movie Render Queue plugin, restart, and add that sequence to the queue."
        )
        return
    try:
        subsystem = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
        queue = subsystem.get_queue()
        job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        job.job_name = "InfinigenShot"
        job.sequence = unreal.SoftObjectPath(sequence.get_path_name())
        world = _editor_world(unreal)
        job.map = unreal.SoftObjectPath(world.get_path_name())
        print(
            f"Added Movie Render Queue job 'InfinigenShot' for {sequence.get_path_name()}."
        )
    except Exception as exc:  # noqa: BLE001
        print(
            "The level sequence was created, but the Movie Render Queue job was not "
            f"({exc}). Open Window → Cinematics → Movie Render Queue and add /Game/Infinigen/InfinigenShot."
        )


def _place_terrain(unreal, plan: dict, root: Path, landscape_name: str) -> str | None:
    terrain = plan["landscape"]
    if not terrain:
        print("Manifest has no terrain. Skipping the Landscape.")
        return None
    heightmap = _existing(root, terrain["heightmap"])
    existing = _find_labeled(unreal, landscape_name)
    if existing is not None:
        _apply_landscape_transform(unreal, existing, terrain)
        print(
            f"Using existing Landscape '{landscape_name}'. Height data was left as imported."
        )
        return None
    created = _try_known_landscape_factory(unreal, terrain)
    if created is not None:
        created.set_actor_label(landscape_name)
        _tag(created)
        _apply_landscape_transform(unreal, created, terrain)
        if not _paint_heightmap(unreal, created, heightmap):
            return (
                support_instructions(unreal, terrain, heightmap)
                + "\nA Landscape actor was spawned but its height could not be loaded from Python. "
                "Import the heightmap file with the settings above onto that actor."
            )
        return None
    preview = terrain.get("preview_mesh")
    if preview:
        preview_file = root / preview
        if preview_file.is_file():
            mesh = _import_static_mesh(
                unreal,
                preview_file,
                f"{plan['content_root']}/Terrain",
                "InfinigenTerrainPreview",
                plan["fbx_options"],
                "fbx",
            )
            _set_nanite(unreal, mesh, plan["nanite"])
            subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
            actor = subsystem.spawn_actor_from_class(
                unreal.StaticMeshActor, unreal.Vector(0.0, 0.0, 0.0)
            )
            actor.set_actor_label("InfinigenTerrainPreview")
            _tag(actor)
            actor.static_mesh_component.set_static_mesh(mesh)
            print(
                "Placed InfinigenTerrainPreview from the heightmap. It is not a Landscape."
            )
    return support_instructions(unreal, terrain, heightmap)


def support_instructions(unreal, terrain: dict, heightmap: Path) -> str:
    # Local import keeps this function usable if a caller wants the text before unreal exists.
    from infinigen.tools.unreal_export.placement import format_landscape_instructions

    del unreal
    return format_landscape_instructions(terrain, heightmap)


def _try_known_landscape_factory(unreal, terrain: dict):
    library = getattr(unreal, "PythonLandscapeLib", None)
    if library is None or not hasattr(library, "create_landscape"):
        return None
    transform = unreal.Transform()
    transform.translation = unreal.Vector(*terrain["location"])
    transform.scale3d = unreal.Vector(
        terrain["scale"]["x"], terrain["scale"]["y"], terrain["scale"]["z"]
    )
    try:
        return library.create_landscape(
            transform,
            int(terrain["quads_per_section"]),
            int(terrain["sections_per_component"]),
            int(terrain["components"]["x"]),
            int(terrain["components"]["y"]),
        )
    except TypeError as exc:
        print(
            f"PythonLandscapeLib.create_landscape was found but rejected the arguments ({exc})."
        )
        return None


def _paint_heightmap(unreal, landscape, heightmap: Path) -> bool:
    if not hasattr(landscape, "landscape_import_heightmap_from_render_target"):
        return False
    world = _editor_world(unreal)
    texture = _import_texture(
        unreal, heightmap, "/Game/Infinigen/Terrain", "InfinigenHeightmap", srgb=False
    )
    resolution = (
        texture.blueprint_get_size_x()
        if hasattr(texture, "blueprint_get_size_x")
        else None
    )
    if not resolution:
        return False
    render_target = unreal.RenderingLibrary.create_render_target2d(
        world,
        int(resolution),
        int(texture.blueprint_get_size_y()),
        unreal.TextureRenderTargetFormat.RTF_RGBA8,
    )
    unreal.RenderingLibrary.draw_texture_to_render_target(world, render_target, texture)
    return bool(
        landscape.landscape_import_heightmap_from_render_target(render_target, False)
    )


def _apply_landscape_transform(unreal, actor, terrain: dict) -> None:
    actor.set_actor_location(unreal.Vector(*terrain["location"]), False, False)
    actor.set_actor_scale3d(
        unreal.Vector(
            terrain["scale"]["x"], terrain["scale"]["y"], terrain["scale"]["z"]
        )
    )
    actor.set_actor_rotation(
        _rotator(
            unreal, terrain.get("rotation_euler_deg", {"pitch": 0, "yaw": 0, "roll": 0})
        ),
        False,
    )


def _instance_transform(unreal, instance: dict):
    transform = unreal.Transform()
    transform.translation = unreal.Vector(*instance["location"])
    quat = instance["rotation_quat"]
    transform.rotation = unreal.Quat(quat[0], quat[1], quat[2], quat[3])
    transform.scale3d = unreal.Vector(*instance["scale"])
    return transform


def _rotator(unreal, euler: dict):
    try:
        return unreal.Rotator(
            pitch=float(euler["pitch"]),
            yaw=float(euler["yaw"]),
            roll=float(euler["roll"]),
        )
    except TypeError as exc:
        raise _error(
            unreal,
            "Could not build an unreal.Rotator from "
            f"pitch={euler['pitch']}, yaw={euler['yaw']}, roll={euler['roll']} ({exc}).",
        ) from exc


def _clear_previous(unreal, keep_labels: set[str]) -> None:
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for actor in list(subsystem.get_all_level_actors()):
        tags = {str(tag) for tag in actor.tags}
        if "InfinigenExport" in tags and actor.get_actor_label() not in keep_labels:
            subsystem.destroy_actor(actor)


def _find_labeled(unreal, label: str):
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for actor in subsystem.get_all_level_actors():
        if actor.get_actor_label() == label:
            return actor
    return None


def _tag(actor) -> None:
    tags = list(actor.tags)
    tags.append("InfinigenExport")
    actor.tags = tags


def _ensure_directory(unreal, path: str) -> None:
    if not unreal.EditorAssetLibrary.does_directory_exist(path):
        unreal.EditorAssetLibrary.make_directory(path)


def _editor_world(unreal):
    if hasattr(unreal, "UnrealEditorSubsystem"):
        return unreal.get_editor_subsystem(
            unreal.UnrealEditorSubsystem
        ).get_editor_world()
    return unreal.EditorLevelLibrary.get_editor_world()


def _save_level(unreal) -> None:
    try:
        unreal.EditorLevelLibrary.save_current_level()
    except Exception as exc:  # noqa: BLE001
        print(
            f"Could not save the current level automatically ({exc}). Save it from File → Save Current Level."
        )


def _existing(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_file():
        raise _missing_file(relative, path)
    return path


def _missing_file(relative: str, path: Path):
    from infinigen.tools.unreal_export.placement import UnrealImportError

    return UnrealImportError(
        f"Manifest references '{relative}', but {path} does not exist. "
        "Keep the manifest next to the assets/ and terrain/ folders from the Blender export, "
        "and pass that manifest path to --manifest."
    )


def _set(obj, name: str, value, label: str) -> None:
    if not hasattr(obj, name):
        raise _error(
            None,
            f"This Unreal build has no '{label}'. The importer needs it set to {value!r} "
            "so pre-converted centimeter meshes are not transformed a second time. "
            "Set that option in the FBX import dialog and import the meshes by hand, then re-run with the meshes already in /Game/Infinigen.",
        )
    setattr(obj, name, value)


def _set_if(obj, name: str, value) -> None:
    if hasattr(obj, name):
        try:
            obj.set_editor_property(name, value)
        except Exception:
            setattr(obj, name, value)


def _error(unreal, message: str):
    del unreal
    from infinigen.tools.unreal_export.placement import UnrealImportError

    return UnrealImportError(message)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import an Infinigen Unreal export into the open level."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest.json written by the Blender exporter.",
    )
    parser.add_argument(
        "--content-root",
        default="/Game/Infinigen",
        help="Unreal content folder for imported assets.",
    )
    parser.add_argument(
        "--no-nanite",
        action="store_true",
        help="Leave Nanite disabled. Default is Nanite on, for PC film work.",
    )
    parser.add_argument(
        "--generate-lods",
        action="store_true",
        help="Build 4 LODs (100/50/25/12.5 percent) for a non-Nanite mobile path.",
    )
    parser.add_argument(
        "--placement", default="hism", choices=["hism", "ism", "foliage"]
    )
    parser.add_argument(
        "--landscape-actor",
        default="InfinigenLandscape",
        help="Existing Landscape actor to reuse. Created landscapes are given this label.",
    )
    return parser


def _load_support():
    try:
        from infinigen.tools.unreal_export.manifest import ManifestError, load_manifest
        from infinigen.tools.unreal_export.placement import (
            UnrealImportError,
            plan_import,
        )
    except ImportError:
        _install_package_on_path()
        from infinigen.tools.unreal_export.manifest import ManifestError, load_manifest
        from infinigen.tools.unreal_export.placement import (
            UnrealImportError,
            plan_import,
        )

    class Support:
        pass

    support = Support()
    support.ManifestError = ManifestError
    support.UnrealImportError = UnrealImportError
    support.load_manifest = load_manifest
    support.plan_import = plan_import
    return support


def _install_package_on_path() -> None:
    """Make ``infinigen.tools.unreal_export`` importable when only this folder was copied."""
    import types

    root = Path(__file__).resolve().parents[1]
    for parent in [root, *root.parents]:
        if (parent / "infinigen" / "tools" / "unreal_export" / "manifest.py").is_file():
            sys.path.insert(0, str(parent))
            return
        if (
            parent / "src" / "infinigen" / "tools" / "unreal_export" / "manifest.py"
        ).is_file():
            sys.path.insert(0, str(parent / "src"))
            return

    def namespace(name: str, path: Path):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__package__ = name
        sys.modules[name] = module
        return module

    infinigen_pkg = namespace("infinigen", root.parent.parent)
    tools_pkg = namespace("infinigen.tools", root.parent)
    export_pkg = namespace("infinigen.tools.unreal_export", root)
    infinigen_pkg.tools = tools_pkg
    tools_pkg.unreal_export = export_pkg


if __name__ == "__main__":
    raise SystemExit(main())
