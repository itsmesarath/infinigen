# Copyright (C) 2026, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import json

from infinigen.tools.unreal_export.manifest import build_manifest, write_manifest
from infinigen.tools.unreal_export.unreal.import_infinigen_scene import main


def _write(path):
    manifest = build_manifest(
        assets=[],
        instances=[],
        terrain=None,
        cameras=[],
        lights=[],
        fps=24,
        frame_start=1,
        frame_end=1,
        mesh_format="fbx",
    )
    write_manifest(path, manifest)


def test_script_explains_that_it_must_run_inside_unreal(tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    _write(manifest)
    assert main(["--manifest", str(manifest)]) == 2
    error = capsys.readouterr().err
    assert "Unreal Editor" in error
    assert "Python Editor Script Plugin" in error
    assert str(manifest) in error


def test_script_explains_a_missing_manifest(tmp_path, capsys):
    missing = tmp_path / "missing.json"
    assert main(["--manifest", str(missing)]) == 2
    error = capsys.readouterr().err
    assert "Manifest not found" in error
    assert "missing.json" in error


def test_script_rejects_bad_json(tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{not json")
    assert main(["--manifest", str(manifest)]) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_manifest_paths_stay_relative_for_the_windows_handoff(tmp_path):
    manifest = build_manifest(
        assets=[
            {
                "id": "asset_0001_Rock",
                "name": "Rock",
                "file": "assets/asset_0001_Rock.fbx",
                "textures": {},
                "baked": False,
            }
        ],
        instances=[],
        terrain=None,
        cameras=[],
        lights=[],
        fps=24,
        frame_start=1,
        frame_end=1,
        mesh_format="fbx",
    )
    path = write_manifest(tmp_path / "manifest.json", manifest)
    payload = json.loads(path.read_text())
    assert "\\" not in payload["assets"][0]["file"]
    assert not payload["assets"][0]["file"].startswith("/")
