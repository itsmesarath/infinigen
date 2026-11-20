# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Rasterize a terrain mesh to a 16-bit heightmap Unreal Landscape can import.

Unreal Landscape resolutions are ``components * sections_per_component * quads_per_section + 1``.
Quads per section are 7, 15, 31, 63, 127 or 255, and a component has 1 or 2 sections per side.

Height samples are stored as uint16. Pixel 32768 is the landscape actor's Z. A sample's
height in centimeters relative to that actor is ``(pixel - 32768) * scale_z / 128``.

Row 0 of the image is landscape vertex Y=0 (minimum Unreal Y, which is minimum Blender X).
Column 0 is landscape vertex X=0 (minimum Unreal X, which is minimum Blender Y).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

QUADS_PER_SECTION = (7, 15, 31, 63, 127, 255)
SECTIONS_PER_COMPONENT = (1, 2)
MAX_RESOLUTION = 8129
DEFAULT_Z_SCALE = 100.0
MID_PIXEL = 32768.0
HEIGHT_UNITS_PER_PIXEL = 128.0


def iter_landscape_layouts() -> list[dict]:
    layouts = []
    for quads in QUADS_PER_SECTION:
        for sections in SECTIONS_PER_COMPONENT:
            step = quads * sections
            components = 1
            while True:
                resolution = components * step + 1
                if resolution > MAX_RESOLUTION:
                    break
                layouts.append(
                    {
                        "resolution": resolution,
                        "components": components,
                        "sections_per_component": sections,
                        "quads_per_section": quads,
                    }
                )
                components += 1
    return layouts


def is_legal_landscape_resolution(resolution: int) -> bool:
    return any(
        layout["resolution"] == int(resolution) for layout in iter_landscape_layouts()
    )


def choose_landscape_layout(requested_resolution: int) -> dict:
    """Pick a square Landscape layout.

    An exact legal size is kept. Anything else snaps to the nearest legal size.
    When several component arrangements share a size, prefer 2 sections of 63 quads,
    which is the arrangement Epic recommends for imported heightmaps.
    """
    requested = int(requested_resolution)
    if requested < 8:
        raise ValueError(
            f"Heightmap resolution {requested} is too small. Unreal Landscape needs at least 8."
        )
    layouts = iter_landscape_layouts()

    def rank(layout: dict) -> tuple:
        preferred = (
            0
            if layout["sections_per_component"] == 2
            and layout["quads_per_section"] == 63
            else 1
        )
        return (preferred, layout["components"], -layout["quads_per_section"])

    exact = [layout for layout in layouts if layout["resolution"] == requested]
    if exact:
        chosen = sorted(exact, key=rank)[0]
    else:
        chosen = min(
            layouts,
            key=lambda layout: (abs(layout["resolution"] - requested), rank(layout)),
        )
    result = dict(chosen)
    result["requested_resolution"] = requested
    return result


def choose_z_scale(heights_m: np.ndarray, margin: float = 1.02) -> tuple[float, float]:
    """Return ``(mid_height_m, scale_z)`` so the samples fit in the uint16 range.

    Flat terrain keeps Unreal's default Z scale of 100, which still encodes the
    constant height as pixel 32768 placed at the actor location.
    """
    finite = np.asarray(heights_m, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Heightmap has no finite samples.")
    mid = 0.5 * (float(finite.min()) + float(finite.max()))
    max_abs_cm = float(np.max(np.abs(finite - mid)) * 100.0)
    if max_abs_cm < 1e-6:
        return mid, DEFAULT_Z_SCALE
    scale = max(max_abs_cm * HEIGHT_UNITS_PER_PIXEL / 32767.0 * margin, 1.0)
    return mid, float(scale)


def encode_height(
    heights_m: np.ndarray,
    mid_m: float | None = None,
    scale_z: float | None = None,
    margin: float = 1.02,
) -> tuple[np.ndarray, float, float]:
    heights = np.asarray(heights_m, dtype=np.float64)
    if mid_m is None or scale_z is None:
        mid_m, scale_z = choose_z_scale(heights, margin=margin)
    relative_cm = (heights - mid_m) * 100.0
    raw = np.rint(relative_cm * HEIGHT_UNITS_PER_PIXEL / scale_z + MID_PIXEL)
    encoded = np.clip(raw, 0, 65535).astype(np.uint16)
    return encoded, float(mid_m), float(scale_z)


def decode_height(encoded: np.ndarray, mid_m: float, scale_z: float) -> np.ndarray:
    pixels = np.asarray(encoded, dtype=np.float64)
    relative_cm = (pixels - MID_PIXEL) * scale_z / HEIGHT_UNITS_PER_PIXEL
    return mid_m + relative_cm / 100.0


def height_grid_from_mesh(
    vertices: np.ndarray, faces, resolution: int
) -> tuple[np.ndarray, dict]:
    """Rasterize a Blender-space mesh to a square height grid in meters.

    ``grid[row, column]`` samples Blender X along rows and Blender Y along columns,
    which is Unreal Y and Unreal X after the axis conversion. Overlapping triangles
    keep the highest sample, so a cave under the ground does not punch a hole in
    the landscape. Uncovered pixels are filled from their neighbors.
    """
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Terrain mesh needs an (N, 3) vertex array.")
    resolution = int(resolution)
    if resolution < 2:
        raise ValueError("Heightmap resolution must be at least 2.")
    min_xy = points[:, :2].min(axis=0)
    max_xy = points[:, :2].max(axis=0)
    size = max_xy - min_xy
    if np.any(size <= 1e-8):
        raise ValueError(
            "Terrain mesh has no area in the XY plane, so it cannot become a Landscape heightmap."
        )
    grid = np.full((resolution, resolution), np.nan, dtype=np.float64)
    triangles = list(_iter_triangles(faces))
    if not triangles:
        raise ValueError("Terrain mesh has no triangles to rasterize into a heightmap.")
    for triangle in triangles:
        _rasterize_triangle(grid, points[list(triangle)], min_xy, size, resolution)
    if not np.any(np.isfinite(grid)):
        raise ValueError("Terrain rasterization produced no height samples.")
    _fill_holes(grid)
    return grid, {
        "min": [float(value) for value in min_xy],
        "max": [float(value) for value in max_xy],
    }


def terrain_record_from_mesh(
    vertices: np.ndarray, faces, resolution: int, name: str
) -> dict:
    """Build the manifest terrain block plus the encoded grids used to write files."""
    layout = choose_landscape_layout(resolution)
    sampled, bounds = height_grid_from_mesh(vertices, faces, layout["resolution"])
    encoded, mid_m, scale_z = encode_height(sampled)
    min_xy = bounds["min"]
    max_xy = bounds["max"]
    quads = layout["resolution"] - 1
    # Unreal X follows Blender Y, Unreal Y follows Blender X.
    scale_x = (max_xy[1] - min_xy[1]) * 100.0 / quads
    scale_y = (max_xy[0] - min_xy[0]) * 100.0 / quads
    return {
        "name": name,
        "resolution": [layout["resolution"], layout["resolution"]],
        "requested_resolution": layout["requested_resolution"],
        "components": {"x": layout["components"], "y": layout["components"]},
        "sections_per_component": layout["sections_per_component"],
        "quads_per_section": layout["quads_per_section"],
        "scale": {"x": float(scale_x), "y": float(scale_y), "z": float(scale_z)},
        "location": [
            float(min_xy[1] * 100.0),
            float(min_xy[0] * 100.0),
            float(mid_m * 100.0),
        ],
        "rotation_euler_deg": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        "height_mid_m": float(mid_m),
        "row_order": "row 0 is landscape Y=0 (minimum Unreal Y / minimum Blender X)",
        "sample_axes": "column i samples Unreal +X (Blender +Y); row j samples Unreal +Y (Blender +X)",
        "z_encoding": "(uint16 - 32768) * scale.z / 128 = centimeters relative to location.z",
        "_encoded": encoded,
        "_heights_m": sampled,
    }


def write_heightmap_png(path: Path, values: np.ndarray) -> None:
    """Write a 16-bit greyscale PNG. Values are linear; no gamma is applied."""
    import struct
    import zlib

    data = np.asarray(values, dtype=np.uint16)
    if data.ndim != 2:
        raise ValueError("Heightmap PNG data must be a 2D array.")
    height, width = data.shape
    raw_rows = data.astype(">u2").tobytes()
    stride = width * 2
    filtered = b"".join(
        b"\x00" + raw_rows[row * stride : (row + 1) * stride] for row in range(height)
    )

    def chunk(tag: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", checksum)
        )

    header = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(filtered))
        + chunk(b"IEND", b"")
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def write_heightmap_raw(path: Path, values: np.ndarray) -> None:
    """Little-endian uint16, row-major, no header. This is Unreal's ``.r16`` layout."""
    data = np.asarray(values, dtype="<u2")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.tofile(path)


def _iter_triangles(faces):
    array = np.asarray(faces)
    if array.ndim == 2 and array.shape[1] == 3:
        for face in array:
            yield (int(face[0]), int(face[1]), int(face[2]))
        return
    for face in faces:
        indices = [int(index) for index in face]
        if len(indices) < 3:
            continue
        for offset in range(1, len(indices) - 1):
            yield (indices[0], indices[offset], indices[offset + 1])


def _rasterize_triangle(grid, triangle, min_xy, size, resolution):
    # (column, row) pixel coordinates. Column follows Blender Y, row follows Blender X.
    coords = []
    for point in triangle:
        column = (point[1] - min_xy[1]) / size[1] * (resolution - 1)
        row = (point[0] - min_xy[0]) / size[0] * (resolution - 1)
        coords.append((column, row))
    columns = [coord[0] for coord in coords]
    rows = [coord[1] for coord in coords]
    min_column = max(int(math.floor(min(columns))), 0)
    max_column = min(int(math.ceil(max(columns))), resolution - 1)
    min_row = max(int(math.floor(min(rows))), 0)
    max_row = min(int(math.ceil(max(rows))), resolution - 1)
    if min_column > max_column or min_row > max_row:
        return
    column_index = np.arange(min_column, max_column + 1)
    row_index = np.arange(min_row, max_row + 1)
    column_grid, row_grid = np.meshgrid(column_index, row_index)
    origin = np.array(coords[0])
    edge_u = np.array(coords[1]) - origin
    edge_v = np.array(coords[2]) - origin
    denominator = edge_u[0] * edge_v[1] - edge_v[0] * edge_u[1]
    if abs(denominator) < 1e-12:
        return
    delta_x = column_grid - origin[0]
    delta_y = row_grid - origin[1]
    beta = (delta_x * edge_v[1] - edge_v[0] * delta_y) / denominator
    gamma = (edge_u[0] * delta_y - delta_x * edge_u[1]) / denominator
    alpha = 1.0 - beta - gamma
    mask = (alpha >= -1e-5) & (beta >= -1e-5) & (gamma >= -1e-5)
    if not np.any(mask):
        return
    height = alpha * triangle[0, 2] + beta * triangle[1, 2] + gamma * triangle[2, 2]
    current = grid[min_row : max_row + 1, min_column : max_column + 1]
    candidate = np.where(mask, height, np.nan)
    grid[min_row : max_row + 1, min_column : max_column + 1] = np.fmax(
        current, candidate
    )
    missing = ~np.isfinite(current) & mask
    if np.any(missing):
        written = grid[min_row : max_row + 1, min_column : max_column + 1]
        written[missing] = candidate[missing]


def _fill_holes(grid: np.ndarray) -> None:
    for _ in range(max(grid.shape)):
        if np.all(np.isfinite(grid)):
            return
        padded = np.pad(grid, 1, constant_values=np.nan)
        neighbors = []
        for row_shift in (-1, 0, 1):
            for column_shift in (-1, 0, 1):
                if row_shift == 0 and column_shift == 0:
                    continue
                neighbors.append(
                    padded[
                        1 + row_shift : 1 + row_shift + grid.shape[0],
                        1 + column_shift : 1 + column_shift + grid.shape[1],
                    ]
                )
        stacked = np.stack(neighbors, axis=0)
        count = np.sum(np.isfinite(stacked), axis=0)
        total = np.nansum(stacked, axis=0)
        holes = ~np.isfinite(grid) & (count > 0)
        grid[holes] = total[holes] / count[holes]
    if not np.all(np.isfinite(grid)):
        grid[~np.isfinite(grid)] = float(np.nanmin(grid))
