"""CPU regression checks for oversized bins, edge overhang and tipped ready poses."""

import math

import numpy as np
import pytest

from robolab.core.scenes.placement import (
    bounds_corners, flat_layout_issues, footprint_separation, tilt_change_deg,
)


def box(lower, upper, x=0, y=0, yaw=0):
    angle = math.radians(yaw) / 2
    return bounds_corners(lower, upper, [x, y, 0, math.cos(angle), 0, 0, math.sin(angle)])


TABLE = box([0.1971368, -0.5, -0.7], [0.8971368, 0.5, 0])


def test_plasticpail_15_uses_real_bin_footprint():
    # Measured composed bounds, not the manifest's 0.23 x 0.17 assumed bin.
    pail = box([0.34725905, -0.26529765, 0], [0.48846276, -0.12409376, 0.12384])
    real_bin = box([-0.21, -0.14, 0], [0.21, 0.14, 0.105], 0.664248957, -0.294793676)
    assumed_bin = box([-0.115, -0.085, 0], [0.115, 0.085, 0.13], 0.664248957, -0.294793676)
    assert footprint_separation(pail, real_bin) < 0
    assert not flat_layout_issues({"pail": pail, "bin": assumed_bin}, TABLE)
    assert any("footprint clearance" in reason for reason in
               flat_layout_issues({"pail": pail, "bin": real_bin}, TABLE))


def test_center_on_table_does_not_allow_overhang():
    overhang = box([-0.08, -0.08, 0], [0.08, 0.08, 0.1], x=0.22)
    assert any("outside table" in r for r in flat_layout_issues({"object": overhang}, TABLE))


def test_rotated_rectangles_with_overlapping_aabbs_can_be_separate():
    first = box([-0.1, -0.01, 0], [0.1, 0.01, 0.1], yaw=45)
    second = first + [-0.04, 0.04, 0]
    assert np.all(first.max(axis=0)[:2] > second.min(axis=0)[:2])
    assert footprint_separation(first, second) > 0.03


def test_clearance_includes_space_for_approach():
    first = box([0.3, -0.1, 0], [0.4, 0, 0.1])
    second = first + [0.11, 0, 0]
    assert flat_layout_issues({"a": first, "b": second}, TABLE)
    assert not flat_layout_issues({"a": first, "b": second + [0.02, 0, 0]}, TABLE)


def test_tipped_stationary_pudding_is_not_an_acceptable_start():
    recorded = [-0.08175399, -0.70442981, -0.06905077, 0.70166004]
    assert tilt_change_deg([1, 0, 0, 0], recorded) == pytest.approx(90.1, abs=0.1)
    assert tilt_change_deg([1, 0, 0, 0], [0, 0, 0, -1]) == pytest.approx(0)
    horizontal = [math.sqrt(.5), math.sqrt(.5), 0, 0]
    assert tilt_change_deg(horizontal, horizontal) == pytest.approx(0)


def test_missing_or_nonfinite_geometry_fails_closed():
    bad = TABLE.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        flat_layout_issues({"bad": bad}, TABLE)


def test_spatial_solver_includes_rotated_rectangle_corners():
    from robolab.scene_gen.llm_scene_gen import SpatialSolver, ObjectState

    solver = SpatialSolver((-2, 2, -2, 2), collision_margin=0)
    states = {"a": ObjectState("a", x=0, y=0, yaw=45),
              "b": ObjectState("b", x=1.2, y=0, yaw=45)}
    assert solver._check_collisions(states, {n: (1, 1, .1) for n in states}) == [("a", "b")]


def test_spatial_bounds_check_is_read_only_and_oversize_fails():
    from robolab.scene_gen.llm_scene_gen import SpatialSolver, ObjectState

    solver = SpatialSolver((0, 1, 0, 1), collision_margin=0)
    states = {"a": ObjectState("a", x=1.1, y=.5, yaw=0)}
    dims = {"a": (.1, .1, .1)}
    assert not solver._check_table_bounds(states, dims)
    assert states["a"].x == 1.1
    assert not solver._optimize_placement(states, {"a": (2, 2, .1)}, 1)
    assert not solver._optimize_placement(states, dims, 1, fixed_objects=["a"])


def test_clamping_cannot_introduce_an_unchecked_collision():
    from robolab.scene_gen.llm_scene_gen import SpatialSolver, ObjectState

    solver = SpatialSolver((0, 2, 0, 2), collision_margin=0)
    states = {"a": ObjectState("a", x=2.3, y=1, yaw=0),
              "b": ObjectState("b", x=1.7, y=1, yaw=0)}
    dims = {n: (.4, .1, .1) for n in states}
    accepted = solver._optimize_placement(states, dims, 1)
    assert not accepted or (not solver._check_collisions(states, dims)
                            and solver._check_table_bounds(states, dims))


def test_runtime_preflights_all_objects_before_writing_poses():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from robolab.core.omy_teleop.preparation import PreparationSettings
    from robolab.core.omy_teleop.preparation_runtime import ScenePreparation

    preparation = ScenePreparation.__new__(ScenePreparation)
    preparation.flat_table = True
    preparation.settings = PreparationSettings()
    preparation.world = SimpleNamespace(clear_contact_cache=Mock())
    preparation.names = ["table", "target", "grey_bin"]
    preparation.env = SimpleNamespace(scene=SimpleNamespace(
        rigid_objects={name: Mock() for name in preparation.names}, write_data_to_sim=Mock()))
    points = {"table": TABLE, "target": box([.4, -.1, 0], [.5, 0, .1]),
              "grey_bin": box([.45, -.1, 0], [.65, .1, .1])}
    preparation._bounds = points.__getitem__
    with pytest.raises(ValueError, match="Cannot prepare layout.*footprint clearance"):
        preparation._correct_placement()
    preparation.env.scene.write_data_to_sim.assert_not_called()
    for asset in preparation.env.scene.rigid_objects.values():
        asset.write_root_pose_to_sim.assert_not_called()


def test_runtime_refuses_tipped_pose_even_when_stability_gate_is_ready():
    from types import SimpleNamespace as NS
    from unittest.mock import Mock
    import torch
    from robolab.core.omy_teleop.preparation import PreparationSettings
    from robolab.core.omy_teleop.preparation_runtime import ScenePreparation

    preparation = ScenePreparation.__new__(ScenePreparation)
    preparation.flat_table = True
    preparation.settings = PreparationSettings()
    preparation.setup_error = None
    preparation.started = 0
    preparation.action = None
    preparation.placement = []
    preparation.names = ["table", "target"]
    preparation.initial_orientations = {"target": [1, 0, 0, 0]}
    poses = {"table": [0, 0, 0, 1, 0, 0, 0],
             "target": [.45, 0, .05, math.sqrt(.5), math.sqrt(.5), 0, 0]}
    assets = {name: NS(data=NS(root_pose_w=torch.tensor([pose]), root_vel_w=torch.zeros(1, 6)))
              for name, pose in poses.items()}
    sensors = {"omy_prepare_"+name: NS(data=NS(net_forces_w=torch.tensor([[[0., 0., 1.]]])))
               for name in poses}
    env = NS(_sim_step_counter=0, physics_dt=.5, cfg=NS(decimation=1, sim=NS(render_interval=2)),
             action_manager=Mock(), sim=Mock(), scene=NS(rigid_objects=assets, sensors=sensors,
             write_data_to_sim=Mock(), update=Mock()), reset_eval_state=Mock())
    preparation.env = env
    preparation.world = NS(clear_contact_cache=Mock(), is_supported_on_surface=Mock(return_value=True))
    preparation.gate = NS(observe=Mock(return_value={"status": "ready", "reasons": [],
                                                    "bodies": {name: {} for name in poses}}))
    points = {"table": TABLE, "target": box([.4, -.05, 0], [.5, .05, .1])}
    preparation._bounds = points.__getitem__
    report = preparation.advance()
    assert report["status"] == "failed"
    assert any("tipped 90.0" in reason for reason in report["reasons"])
    env.reset_eval_state.assert_not_called()
    assert preparation.advance()["status"] == "failed"  # Cannot later settle into acceptance.
