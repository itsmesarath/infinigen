# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Command line for the Blender-side Unreal export."""

from __future__ import annotations

import argparse
from pathlib import Path

from infinigen.tools.unreal_export.blender_scene import (
    export_blend_file,
    export_current_scene,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a generated Infinigen scene for Unreal Engine 5.8. "
            "Unique meshes are written once. Copies, the terrain heightmap, cameras and the sun "
            "go into manifest.json in Unreal's centimeter Z-up coordinates."
        )
    )
    parser.add_argument(
        "--blend",
        type=Path,
        help="scene.blend to open. Omit to export the current file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for the manifest, meshes and heightmap.",
    )
    parser.add_argument(
        "--format",
        default="fbx",
        choices=["fbx", "usd", "usdc", "usda"],
        help="Mesh container. FBX is the path the Unreal script imports with axis conversion disabled.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Baked texture resolution. Ignored with --no-bake.",
    )
    parser.add_argument(
        "--heightmap-resolution",
        type=int,
        default=505,
        help="Target Landscape heightmap size. Illegal sizes snap to the nearest size Unreal accepts.",
    )
    parser.add_argument(
        "--no-bake",
        action="store_true",
        help="Skip Cycles baking and write a solid base color from the Principled BSDF.",
    )
    parser.add_argument("--bake-device", default="CPU", choices=["CPU", "GPU"])
    parser.add_argument(
        "--terrain-name",
        action="append",
        default=None,
        help="Extra object name to treat as terrain. Repeat to add more. OpaqueTerrain is always included.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Sample every Nth frame of the camera and sun.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    kwargs = dict(
        mesh_format=args.format,
        texture_resolution=args.resolution,
        heightmap_resolution=args.heightmap_resolution,
        bake=not args.no_bake,
        bake_device=args.bake_device,
        terrain_names=args.terrain_name,
        frame_step=args.frame_step,
    )
    if args.blend:
        manifest = export_blend_file(args.blend, args.output, **kwargs)
    else:
        manifest = export_current_scene(args.output, **kwargs)
    print(manifest)
    return 0
