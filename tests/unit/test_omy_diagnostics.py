"""Substep fault capture without simulator, headset, or serial hardware."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from robolab.core.omy_teleop.diagnostics import (
    DiagnosticFailure, RuntimeDiagnostics, enable_input_diagnostics, joint_violations, world_anchor,
)
from robolab.core.omy_teleop.options import build_parser, validate_args
from robolab.core.omy_teleop.performance import PerformanceMonitor


def test_guard_checks_actual_limits_and_passive_joint_nonfinite_values():
    names = ["panda_joint1", "left_inner_finger_joint"]
    limits = np.array([[-2, 2], [-3, 3]])
    assert joint_violations(names, [2.1, 0], [0, 0], limits, 0.2) == []
    violations = joint_violations(names, [2.21, 0], [0, np.nan], limits, 0.2)
    assert "panda_joint1" in violations[0] and "outside" in violations[0]
    assert "left_inner_finger_joint" in violations[1] and "nonfinite" in violations[1]


def test_world_anchor_uses_link_frame_and_wxyz_quaternion():
    angle = np.pi / 4
    pose = [1, 2, 3, np.cos(angle), 0, 0, np.sin(angle)]
    np.testing.assert_allclose(world_anchor(pose, [1, 0, 0]), [1, 3, 3], atol=1e-12)


def test_leader_observer_does_not_reread_or_change_serial_errors():
    raw = {"joint_1.pos": 0.4, "gripper.pos": 0.0}
    calls = []

    def read():
        calls.append("serial")
        if len(calls) > 1:
            raise ConnectionError("sync read failed")
        return raw

    latest_calls = []
    device = SimpleNamespace(cfg=SimpleNamespace(source="serial", stale_s=0.5),
                             _serial=SimpleNamespace(read=read),
                             _latest=lambda: latest_calls.append(1) or (raw, 0.6))
    enable_input_diagnostics(device)
    enable_input_diagnostics(device)  # observer must not be installed twice
    assert device._serial.read() is raw
    with pytest.raises(ConnectionError, match="sync read failed"):
        device._serial.read()
    result, age = device._latest()
    assert result is raw and age == 0.6 and latest_calls == [1] and len(calls) == 2
    assert device.diagnostic_input["stale"]
    assert device.diagnostic_input["serial"]["read_calls"] == 2
    assert device.diagnostic_input["serial"]["read_errors"] == 1
    raw["joint_1.pos"] = 99
    assert device.diagnostic_input["raw"]["joint_1.pos"] == 0.4


def test_tcp_observer_distinguishes_missing_sample_from_epoch_sized_age():
    device = SimpleNamespace(cfg=SimpleNamespace(source="tcp", stale_s=0.5),
                             _latest=lambda: (None, 1e9))
    enable_input_diagnostics(device)
    assert device._latest() == (None, 1e9)
    assert device.diagnostic_input == {"raw": None, "age_s": None, "stale": True, "serial": None}


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_invalid_guard_tolerance_rejected(value):
    parser = build_parser()
    with pytest.raises(SystemExit):
        validate_args(parser, parser.parse_args(["--diagnostics", "--diagnostic-joint-tolerance", value]))


def fixture_env():
    names = [f"panda_joint{i}" for i in range(1, 8)] + ["finger_joint", "right_outer_knuckle_joint",
             "right_inner_finger_joint", "right_inner_finger_knuckle_joint",
             "left_inner_finger_knuckle_joint", "left_inner_finger_joint"]
    positions = np.zeros((1, len(names)))
    poses = np.array([[[0, 0, 0, 1, 0, 0, 0], [0, 0.02, 0, 1, 0, 0, 0]]], dtype=float)
    data = SimpleNamespace(joint_pos=positions, joint_vel=positions.copy(),
                           joint_pos_limits=np.tile([-3, 3], (1, len(names), 1)),
                           joint_vel_limits=np.ones_like(positions),
                           default_joint_stiffness=np.ones_like(positions),
                           default_joint_damping=np.ones_like(positions),
                           joint_effort_limits=np.ones_like(positions),
                           body_link_pose_w=poses, body_link_state_w=np.concatenate((poses, np.zeros((1, 2, 6))), axis=-1))
    robot = SimpleNamespace(joint_names=names, body_names=["base", "finger"], data=data,
                            root_physx_view=SimpleNamespace(link_paths=[["/robot/base", "/robot/finger"]],
                                                           prim_paths=["/robot"]))
    obj = SimpleNamespace(data=SimpleNamespace(root_state_w=np.zeros((1, 13))),
                          root_physx_view=SimpleNamespace(prim_paths=["/banana"]))
    contact = SimpleNamespace(cfg=SimpleNamespace(filter_prim_paths_expr=["banana"]),
                              data=SimpleNamespace(net_forces_w=np.zeros((1, 1, 3)),
                                                   force_matrix_w=np.zeros((1, 1, 1, 3))))

    class Scene(dict):
        rigid_objects = {"banana": obj}
        sensors = {"gripper__banana": contact}

        def update(self, dt):
            return "updated"

    env = SimpleNamespace(scene=Scene(robot=robot), _sim_step_counter=2, physics_dt=1 / 120, step_dt=1 / 60,
                          sim=SimpleNamespace(get_initial_stage=lambda: None))
    return env


def install(diagnostics, monkeypatch):
    monkeypatch.setattr("robolab.core.omy_teleop.diagnostics.inspect_stage", lambda *args: {
        "joints": [{"path": "/robot/finger_joint", "body0": ["/robot/base"], "body1": ["/robot/finger"],
                    "local_pos0": [0, 0.02, 0], "local_pos1": [0, 0, 0]}], "colliders": [],
    })
    diagnostics.install()


def test_substep_failure_persists_first_fault_and_partial_control(tmp_path, monkeypatch):
    env = fixture_env()
    timing = PerformanceMonitor()
    diagnostics = RuntimeDiagnostics(env, tmp_path, timing)
    original_update = env.scene.update
    install(diagnostics, monkeypatch)
    device = SimpleNamespace(enable_input_diagnostics=lambda: None,
                             diagnostic_input={"raw": {"gripper.pos": 0}, "age_s": 0.01, "stale": False})
    diagnostics.start_episode(0, device)
    timing.begin_step()
    diagnostics.begin_control(np.array([0] * 7 + [1]), 1)
    assert env.scene.update(1 / 120) == "updated"
    env._sim_step_counter = 3
    env.scene["robot"].data.joint_pos[0, -1] = 4.0  # passive finger fails first
    with pytest.raises(DiagnosticFailure, match="left_inner_finger_joint"):
        env.scene.update(1 / 120)
    diagnostics.end_control(incomplete=True)
    diagnostics.close()
    assert env.scene.update == original_update
    metadata = json.loads((tmp_path / "omy_diagnostic_metadata.json").read_text())
    assert len(metadata["joint_names"]) == 13
    assert "banana" in metadata["object_collision_audits"]
    rows = [json.loads(line) for line in (tmp_path / "omy_diagnostics.jsonl").read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["reset_state", "physics_state", "physics_state", "control_timing"]
    assert rows[1]["action"][-1] == 1 and rows[1]["leader"]["raw"]["gripper.pos"] == 0
    assert rows[1]["joint_pos_rad"][-1] == 0  # captures must not alias live tensors
    assert rows[-1]["incomplete"] and rows[-2]["physics_step"] == 3
    summary = json.loads((tmp_path / "omy_diagnostic_summary.json").read_text())
    assert summary["first_failure"]["physics_step"] == 3 and summary["records"] == 4


def test_nonfinite_object_snapshot_is_saved_as_strict_json_before_stop(tmp_path, monkeypatch):
    env = fixture_env()
    diagnostics = RuntimeDiagnostics(env, tmp_path, PerformanceMonitor())
    install(diagnostics, monkeypatch)
    env.scene.rigid_objects["banana"].data.root_state_w[0, 7] = np.nan
    with pytest.raises(DiagnosticFailure, match="banana: nonfinite"):
        diagnostics.capture("reset_state")
    diagnostics.close()
    row = json.loads((tmp_path / "omy_diagnostics.jsonl").read_text())
    assert row["objects"]["banana"]["root_state_w"][7] == "nan"
