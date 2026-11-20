# Infinigen → Unreal Engine 5.8

Export a generated nature scene so Unreal can rebuild it without turning every copy of a tree, rock or plant into its own mesh. The Blender side writes each unique asset once, plus a manifest of every instance. The Unreal side places those instances on instanced components, builds a Landscape from a 16-bit heightmap, and creates a cine camera level sequence for Movie Render Queue.

Infinigen runs in WSL or Linux. Unreal Editor runs on Windows. The hand-off is a folder of files.

## What gets written

```
export_dir/
  manifest.json              instance transforms, camera, sun, landscape scale
  assets/asset_0001_Tree.fbx one file per unique mesh, already in Unreal centimeters
  textures/                  baked maps, or a solid base color when baking is off
  terrain/heightmap.png      16-bit Landscape heightmap
  terrain/heightmap.r16      same data as little-endian raw, no header
  terrain/terrain_preview.fbx coarse mesh used only if Python cannot spawn a Landscape
```

Coordinates in the manifest are Unreal's: centimeters, Z-up, left-handed, +X forward, +Y right. The conversion from Blender meters is:

```
unreal_x = blender_y * 100
unreal_y = blender_x * 100
unreal_z = blender_z * 100
```

Mesh vertices are converted the same way before export. The Unreal script imports FBX with **Convert Scene** and **Convert Scene Unit** off. Turning those on will scale and rotate the meshes a second time, and the instances will not line up.

A Blender camera looks down local -Z. The manifest rotation is an Unreal cine camera, which looks down local +X. Sun direction is the direction light travels (into the scene), and it is the +X axis of the directional light.

Height samples use Unreal's Landscape encoding. Pixel 32768 is the landscape actor's Z. Relative height in centimeters is `(pixel - 32768) * scale.z / 128`. Row 0 of the PNG is minimum Unreal Y. Column 0 is minimum Unreal X. The manifest `terrain.scale` and `terrain.location` are the values to type into Landscape mode.

## 1. Export from WSL or Linux

Use the same Python that has `bpy` (Infinigen's `bpy==4.2.0` environment).

```bash
cd /path/to/infinigen
export PYTHONPATH=src

python -m infinigen.tools.unreal_export \
  --blend outputs/nature/scene.blend \
  --output /mnt/d/infinigen_unreal/scene01 \
  --format fbx \
  --heightmap-resolution 505 \
  --resolution 1024
```

`--output` should be a Windows drive mounted in WSL (`/mnt/d/...`) or a folder you copy later. Do not point it at a Linux-only path unless you will copy the whole folder across.

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--format fbx` | Default. This is the path the Unreal script imports. `usd` / `usdc` / `usda` are also written in the same centimeter space. |
| `--heightmap-resolution 505` | Target Landscape size. Values Unreal rejects (anything other than `components * sections * quads + 1`) snap to the nearest legal size. 505 is 4×4 components, 2×2 sections, 63 quads. |
| `--resolution 1024` | Bake resolution for base color, roughness, normal and metallic. |
| `--no-bake` | Skip Cycles and write one solid color from the Principled BSDF. Use this when you have no GPU and only need layout. |
| `--bake-device CPU` | Default. Baking calls `infinigen.tools.export.bake_object`, which needs the Infinigen install (`gin`, `trimesh`). |
| `--terrain-name OpaqueTerrain` | Extra object names to rasterize as the Landscape. `OpaqueTerrain`, `terrain`, `ground` and `landtiles` are already recognized. |
| `--frame-step 2` | Sample every Nth frame of the camera and sun. |

The command prints the path of `manifest.json`.

### What is instanced

Geometry-node scatters and linked duplicates that share a mesh are one asset. Each copy is a transform in the manifest. Leaves that Infinigen instances inside a tree stay instances of the leaf mesh, which is the small file you want. Two trees that were generated as different meshes stay two assets; the exporter does not merge different geometry.

Terrain is not exported as that mesh. A heightmap cannot represent caves or overhangs. Those meshes are exported as ordinary assets if their names are not terrain names. Water is a mesh, not part of the heightmap.

Atmosphere and cloud objects are skipped.

## 2. Open the folder on Windows

If you wrote straight to `/mnt/d/infinigen_unreal/scene01`, the Windows path is `D:\infinigen_unreal\scene01`.

If the export stayed inside the Linux filesystem, copy it out:

```bash
cp -a outputs/unreal /mnt/d/infinigen_unreal/scene01
```

From Windows you can also reach a WSL path as `\\wsl$\Ubuntu\home\<user>\infinigen\outputs\unreal`. Prefer a normal drive letter for the FBX importer. It is unreliable with `\\wsl$` paths.

Keep `manifest.json` next to `assets/`, `textures/` and `terrain/`. Paths inside the manifest are relative to the manifest.

The Unreal script lives in this repo, not in the export folder. On Windows, clone the fork (or copy `src/infinigen/tools/unreal_export`) so the script can import the manifest parser. A full copy of the folder still works if the `infinigen/tools/unreal_export` parents are not present; the script registers that package itself.

## 3. Import in Unreal Editor 5.8

1. Create or open a project. An empty Basic level is easier than a World Partition template, because Landscape import and Python actor spawning are more predictable there.
2. Edit → Plugins. Enable **Python Editor Script Plugin**, **Editor Scripting Utilities**, and **Movie Render Queue**. Restart the editor.
3. In the Output Log, switch the command dropdown to Python and run (one line):

```text
py "C:/src/infinigen/src/infinigen/tools/unreal_export/unreal/import_infinigen_scene.py" --manifest "D:/infinigen_unreal/scene01/manifest.json"
```

Or from `UnrealEditor-Cmd.exe`:

```text
UnrealEditor-Cmd.exe MyProject.uproject -ExecutePythonScript="C:/src/infinigen/src/infinigen/tools/unreal_export/unreal/import_infinigen_scene.py --manifest D:/infinigen_unreal/scene01/manifest.json"
```

Flags:

| Flag | Meaning |
| --- | --- |
| *(default)* | Nanite on, for PC film work. |
| `--no-nanite` | Leave Nanite off. |
| `--generate-lods` | Four LODs at 100%, 50%, 25% and 12.5% triangles, for a mobile path. |
| `--placement hism` | Default. One Hierarchical Instanced Static Mesh component per unique asset. `ism` is the non-hierarchical version. `foliage` builds a foliage type when that API exists, and tells you to use `hism` when it does not. |
| `--content-root /Game/Infinigen` | Where meshes, materials and the sequence are saved. |
| `--landscape-actor InfinigenLandscape` | Actor label to reuse on a second run. |

The script deletes actors it tagged `InfinigenExport` and replaces assets under the content root, so running it again is safe. It will not delete a Landscape you already named `InfinigenLandscape`.

### Landscape

Unreal's Python API does not expose `ALandscape::Import`. `spawn_actor_from_class(Landscape)` returns a placeholder with no components. The script still imports meshes, lights and the camera. It also places `InfinigenTerrainPreview` (a static mesh of the heightmap) and then stops with a message like:

```text
Section Size: 63 x 63 quads
Sections Per Component: 2 x 2
Number of Components: 4 x 4
Scale: X=... Y=... Z=...
```

Do this once:

1. Shift+2 to open Landscape mode, then **Import from File**.
2. Use the heightmap path, section size, component count, scale and location from that message. The numbers are also in `manifest.json` under `terrain`.
3. Do not let the importer invent a new Z scale. `scale.z` is what makes `(pixel - 32768) * scale.z / 128` match the exported heights.
4. Name the actor `InfinigenLandscape`.

Run the script again. It finds that actor, sets the transform from the manifest, and leaves the height data you imported.

If a third-party `PythonLandscapeLib.create_landscape` is installed, the script uses it and then tries to push the heightmap through a render target. The file import above is the accurate path. Render-target import quantizes differently.

### Level sequence and Movie Render Queue

The script spawns a `CineCameraActor`, sets the filmback (sensor width and height in mm) and focal length, and saves `/Game/Infinigen/InfinigenShot`. Transform keys use the manifest frame numbers. Channel order is location X Y Z, roll, pitch, yaw, scale.

When the Movie Render Queue plugin is enabled, a job named `InfinigenShot` is added to the current queue, pointed at that sequence and the open level. Open Window → Cinematics → Movie Render Queue, pick the output, and render. If the plugin was disabled, the sequence is still there; enable the plugin and add the sequence by hand. The log says so when that happens.

### Materials

Procedural Infinigen shaders do not survive FBX. Baking (the default) uses the existing `bake_object` path and writes base color, roughness, normal and metallic PNGs. The script builds an Unreal material from whichever of those files exist. `--no-bake` writes a constant base color instead. Transmission, sheen and coat are not reconstructed.

## Example, end to end

On the Linux / WSL side, after a scene has been generated:

```bash
python -m infinigen.tools.unreal_export \
  --blend ~/infinigen/outputs/scene.blend \
  --output /mnt/d/infinigen_unreal/scene01 \
  --format fbx \
  --heightmap-resolution 505 \
  --resolution 512 \
  --bake-device CPU
```

On Windows, in the Unreal Output Log (Python):

```text
py "C:/src/infinigen/src/infinigen/tools/unreal_export/unreal/import_infinigen_scene.py" --manifest "D:/infinigen_unreal/scene01/manifest.json" --generate-lods
```

Drop `--generate-lods` for a Nanite film scene. Add `--no-nanite` only when you also want the mobile LODs and no Nanite.

## Tests

Unit tests cover the manifest, the axis conversion, heightmap resolution and encoding, light direction, and the Unreal script's failure messages. They do not need Unreal or a GPU:

```bash
python -m pytest tests/infinigen/tools/unreal_export
```

`test_blender_export.py` builds a small scene in `bpy` (two linked rocks, a sloped terrain, a keyed camera, a sun, and a geometry-node scatter) and runs the exporter. A full Infinigen solve is not required for that test. A third test bakes a Principled cube through `infinigen.tools.export.bake_object` on Cycles CPU when `gin` and `trimesh` import; otherwise it skips.

The Unreal Editor script cannot run in the Linux test environment. Its planning code (instance grouping, FBX options, transform-track channel order, Landscape instructions) is tested without the `unreal` module. Running the script outside the editor exits with a message that names the Python Editor Script Plugin and the manifest path, instead of a raw `ImportError`.

## Limits

- Heightmaps are a single surface. Caves and overhangs stay as meshes.
- Identical realized copies are merged when their local vertices match to 1e-5 m. Different procedural variations stay separate assets.
- Shear on an instance is dropped and reported in `manifest.warnings`. Unreal instances are rotation plus scale.
- Animated deformation, particles and hair are not exported. Camera and sun transforms are sampled per frame.
- USD files use the same vertex space, but Unreal's USD importer may apply its own axis conversion. Use FBX unless you have confirmed the USD import options leave the centimeters alone.
