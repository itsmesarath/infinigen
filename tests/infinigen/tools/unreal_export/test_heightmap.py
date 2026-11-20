# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import numpy as np

from infinigen.tools.unreal_export.heightmap import (
    choose_landscape_layout,
    decode_height,
    encode_height,
    height_grid_from_mesh,
    is_legal_landscape_resolution,
    write_heightmap_png,
    write_heightmap_raw,
)


def test_recommended_landscape_sizes_are_legal():
    for resolution in (127, 253, 505, 1009, 2017, 4033):
        layout = choose_landscape_layout(resolution)
        assert layout["resolution"] == resolution
        assert is_legal_landscape_resolution(resolution)
        quads = (
            layout["components"]
            * layout["sections_per_component"]
            * layout["quads_per_section"]
        )
        assert quads + 1 == resolution
        assert layout["sections_per_component"] in (1, 2)
        assert layout["quads_per_section"] in (7, 15, 31, 63, 127, 255)


def test_non_legal_request_snaps_to_nearest_legal_size():
    layout = choose_landscape_layout(100)
    assert layout["resolution"] == 99
    assert is_legal_landscape_resolution(layout["resolution"])
    assert not is_legal_landscape_resolution(100)
    assert layout["requested_resolution"] == 100


def test_height_encoding_roundtrip_uses_unreal_scale():
    heights = np.array([-2.0, 0.0, 3.5])
    encoded, mid_m, scale_z = encode_height(heights)
    decoded = decode_height(encoded, mid_m, scale_z)
    # 16-bit quantization over the used range stays well under a centimeter.
    assert np.max(np.abs(decoded - heights)) < 0.01
    assert encoded.min() >= 0
    assert encoded.max() <= 65535
    # The midpoint of the range lands on the landscape actor height (pixel 32768).
    mid_encoded = encode_height(np.array([mid_m]), mid_m=mid_m, scale_z=scale_z)[0]
    assert int(np.asarray(mid_encoded).reshape(-1)[0]) == 32768


def test_flat_terrain_uses_default_z_scale_and_mid_gray():
    encoded, mid_m, scale_z = encode_height(np.array([4.0, 4.0, 4.0]))
    assert mid_m == 4.0
    assert scale_z == 100.0
    assert np.all(encoded == 32768)


def test_sloped_mesh_samples_blender_x_along_landscape_rows():
    # Two triangles covering the square [-10, 10] x [-10, 10], height = 2 * blender X.
    vertices = np.array(
        [
            [-10.0, -10.0, -20.0],
            [10.0, -10.0, 20.0],
            [10.0, 10.0, 20.0],
            [-10.0, 10.0, -20.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    grid, bounds = height_grid_from_mesh(vertices, faces, resolution=5)
    assert grid.shape == (5, 5)
    # Column i is constant in Blender X; row j follows Blender X from min to max.
    assert bounds["min"][0] == -10.0
    assert bounds["max"][0] == 10.0
    np.testing.assert_allclose(grid[0, :], -20.0, atol=1e-6)
    np.testing.assert_allclose(grid[-1, :], 20.0, atol=1e-6)
    np.testing.assert_allclose(grid[2, :], 0.0, atol=1e-6)


def test_png_and_raw_roundtrip_preserve_uint16(tmp_path):
    values = np.arange(16, dtype=np.uint16).reshape(4, 4) * 4000
    png = tmp_path / "height.png"
    raw = tmp_path / "height.r16"
    write_heightmap_png(png, values)
    write_heightmap_raw(raw, values)
    from PIL import Image

    loaded = np.array(Image.open(png))
    assert loaded.dtype == np.uint16
    np.testing.assert_array_equal(loaded, values)
    raw_loaded = np.fromfile(raw, dtype="<u2").reshape(values.shape)
    np.testing.assert_array_equal(raw_loaded, values)
