"""Integration checks using actual composed USD payloads, without PhysX."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("pxr.Usd")

from tools.audit_simple_pnp_scenes import ROOT, audit_flat_scene, audit_scene
from tools.generate_simple_pnp_tasks import validate_scene_geometry


def test_actual_plasticpail_collision_and_origin_are_detected():
    row, stage = audit_scene(ROOT / "assets/scenes/simple_pnp/pnp_242_plasticpail_a01_15.usda",
                             "plasticpail_a01")
    assert stage
    assert row["footprint_separation_m"] == pytest.approx(-0.03227, abs=1e-5)
    assert row["bottom_gaps_m"]["plasticpail_a01"] == pytest.approx(0.06392, abs=1e-5)
    lower, upper = (row["bounds"]["grey_bin"][key] for key in ("lower", "upper"))
    assert [b-a for a,b in zip(lower, upper)] == pytest.approx([.42, .28, .105], abs=1e-6)


def test_safe_layout_passes_static_checks():
    row, _stage = audit_scene(ROOT / "assets/scenes/simple_pnp/pnp_037_chocolate_pudding_mix_15.usda",
                              "chocolate_pudding_mix")
    assert row["issues"] == []


def test_new_task_generation_refuses_bad_geometry():
    with pytest.raises(ValueError, match="Scene geometry failed.*footprint clearance"):
        validate_scene_geometry([{"scene": "pnp_242_plasticpail_a01_15.usda",
                                  "target": {"name": "plasticpail_a01"}}])


@pytest.fixture
def flat_scene(tmp_path):
    from pxr import Usd, UsdGeom

    path = tmp_path / "custom.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/world").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    for name, scale, pos in (("desk", (.5, .5, .05), (0, 0, -.05)),
                             ("target", (.04, .04, .04), (-.2, 0, .042)),
                             ("destination", (.08, .08, .04), (.2, 0, .042))):
        cube = UsdGeom.Cube.Define(stage, f"/world/{name}")
        cube.CreateSizeAttr(2)
        cube.AddTranslateOp().Set(pos)
        cube.AddScaleOp().Set(scale)
    stage.GetRootLayer().Save()
    return path


def test_arbitrary_flat_scene_and_report_hashes(flat_scene, tmp_path):
    report = tmp_path / "geometry.json"
    result = subprocess.run([sys.executable, str(ROOT / "tools/audit_simple_pnp_scenes.py"),
                             "--scene", str(flat_scene), "--objects", "target", "destination",
                             "--support", "desk", "--report", str(report)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(report.read_text())
    assert data["status"] == "passed" and data["issues"] == []
    assert data["objects"] == ["target", "destination"]
    assert data["asset_layer_sha256"][str(flat_scene)] == hashlib.sha256(flat_scene.read_bytes()).hexdigest()


def test_missing_payload_fails_even_with_other_visible_geometry(flat_scene):
    from pxr import Usd

    stage = Usd.Stage.Open(str(flat_scene))
    stage.GetPrimAtPath("/world/target").GetPayloads().AddPayload("missing_payload.usd")
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="composition errors"):
        audit_flat_scene(flat_scene, ["target", "destination"], support="desk")


def test_flat_audit_does_not_accept_an_intended_stack(flat_scene):
    from pxr import Usd

    stage = Usd.Stage.Open(str(flat_scene))
    stage.GetPrimAtPath("/world/target").GetAttribute("xformOp:translate").Set((.2, 0, .124))
    stage.GetRootLayer().Save()
    row, _stage = audit_flat_scene(flat_scene, ["target", "destination"], support="desk")
    assert any("footprint clearance" in reason for reason in row["issues"])
    assert any("bottom gap" in reason for reason in row["issues"])


@pytest.mark.parametrize("objects", [[], ["target", "target"], ["desk", "target"], ["unknown"]])
def test_invalid_object_selection_fails(flat_scene, objects):
    with pytest.raises(ValueError):
        audit_flat_scene(flat_scene, objects, support="desk")
