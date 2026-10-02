"""XR + L100 launch/lifecycle tests without Isaac, SteamVR, or serial access."""

import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from robolab.core.omy_teleop.xr import (
    XRControlSession, build_parser, launcher_options, preflight, validate_args,
)


def args_for(*argv):
    parser = build_parser()
    args = parser.parse_args(list(argv))
    validate_args(parser, args)
    return args


def test_shared_calibration_viewpoint_and_launcher():
    args = args_for("--source", "tcp", "--headless")
    assert Path(args.calib).name == "omy_calib_robolab.json"
    assert not args.auto_zero and not args.record and not args.stream
    assert args.eye == (-0.25, 0.0, 0.95) and args.lookat == (0.53, 0.0, 0.48)
    assert args.render_interval == 2
    options = launcher_options(args)
    assert options["xr"] and options["video"] and not options["enable_cameras"]
    assert options["device"] == "cpu" and options["device_explicit"]
    assert "--enable omni.kit.xr.profile.vr" in options["kit_args"]
    assert "--/rtx-transient/resourcemanager/enableTextureStreaming=false" in options["kit_args"]
    args = args_for("--device", "cuda:1", "--render-gpu", "1", "--record", "--record-images")
    options = launcher_options(args)
    assert options["enable_cameras"] and options["device"] == "cuda:1"
    assert "/renderer/activeGpu=1" in options["kit_args"]


@pytest.mark.parametrize("argv", [
    ["--stream"], ["--render-interval", "4"], ["--record-images"],
    ["--xr-timeout", "0"], ["--device", "cuda:bad"], ["--calib", "/missing.json"],
    ["--xr-resolution-multiplier", "nan"], ["--xr-resolution-multiplier", "0"],
    ["--xr-resolution-multiplier", "3"], ["--desktop", "--xr-resolution-multiplier", "0.8"],
])
def test_invalid_combinations(argv):
    with pytest.raises(SystemExit):
        args_for(*argv)


def test_preflight_read_only_serial_and_dependency_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.1.0")
    args = args_for("--desktop", "--port", str(tmp_path / "missing_port"))
    assert "Leader port not found" in " ".join(preflight(args)["errors"])
    port = tmp_path / "port"
    port.touch()
    args.port = str(port)
    assert preflight(args)["errors"] == []
    monkeypatch.setattr(os, "access", lambda *args: False)
    assert "dialout" in " ".join(preflight(args)["errors"])
    args.source = "tcp"
    assert preflight(args)["errors"] == []

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert "requirements-omy" in " ".join(preflight(args)["errors"])


def test_resolution_multiplier_metadata():
    args = args_for("--xr-resolution-multiplier", "0.8")
    session = XRControlSession(args, {"mode": "steamvr", "runtime_json": "/runtime.json"})
    assert session.metadata["xr_resolution_multiplier_override"] == 0.8
    assert args_for().xr_resolution_multiplier is None


class FakeView:
    def __init__(self):
        self.active = self.ever_active = False
        self.observed = False
        self.closed = False

    def start(self):
        pass

    def update(self):
        changed = self.observed != self.active
        self.observed = self.active
        self.ever_active |= self.active
        return changed

    def close(self):
        self.closed = True


def session_for(view, *, clock=lambda: 0.0, desktop=False):
    args = args_for(*(["--desktop"] if desktop else []))
    report = {"mode": "desktop" if desktop else "steamvr", "runtime_json": "/runtime.json"}
    return XRControlSession(args, report, view_factory=lambda *args: view, clock=clock)


def test_wait_activation_disconnect_reconnect_and_cleanup():
    view = FakeView()
    session = session_for(view)
    app = SimpleNamespace(is_running=lambda: True)
    renders = []

    def render():
        renders.append(1)
        view.active = True

    session.start()
    assert not session.ready()
    assert session.wait_for_display(app, render)
    assert len(renders) == 2  # activation + processing the scheduled camera pose
    assert session.metadata["xr_display_seen"]
    view.active = False
    assert not session.ready()
    assert session.wait_for_display(app, render)
    assert session.metadata["display_waits"] == 2
    session.close(render)
    assert view.closed and session.view is None


def test_timeout_and_app_close_do_not_enable_control():
    view = FakeView()
    times = iter([0.0, 61.0])
    session = session_for(view, clock=lambda: next(times))
    session.start()
    with pytest.raises(RuntimeError, match="XR display inactive"):
        session.wait_for_display(SimpleNamespace(is_running=lambda: True), lambda: None)
    assert not session.metadata["xr_display_seen"]
    session = session_for(view)
    session.start()
    assert not session.wait_for_display(SimpleNamespace(is_running=lambda: False), lambda: None)


def test_desktop_bypasses_xr_but_retains_viewpoint():
    session = session_for(None, desktop=True)
    cfg = SimpleNamespace(viewer=SimpleNamespace())
    session.configure(cfg)
    assert cfg.viewer.eye == session.args.eye
    session.start()
    assert session.ready() and session.view is None
    assert session.wait_for_display(SimpleNamespace(is_running=lambda: True), lambda: pytest.fail("unexpected wait"))
    assert not session.metadata["xr_display_seen"]


@pytest.mark.parametrize("outcome, expected", [(0, 0), (130, 130), (RuntimeError("test failure"), 1)])
def test_cli_restores_runtime_and_sets_exit_status(monkeypatch, capsys, outcome, expected):
    calls = []
    app = SimpleNamespace(app=SimpleNamespace(post_quit=lambda status: calls.append(status)),
                          close=lambda: calls.append("close"))

    class Launcher:
        def __init__(self, options):
            assert os.environ["XR"] == "1"
            assert os.environ["XR_RUNTIME_JSON"] == "/steamxr.json"
            assert options["xr"] and not options["enable_cameras"]
            self.app = app

    def run(args, simulation_app, *, xr_session):
        assert simulation_app is app and isinstance(xr_session, XRControlSession)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    fake_app = ModuleType("isaaclab.app")
    fake_app.AppLauncher = Launcher
    fake_runner = ModuleType("robolab.core.omy_teleop.runner")
    fake_runner.run = run
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_app)
    monkeypatch.setitem(sys.modules, "robolab.core.omy_teleop.runner", fake_runner)
    monkeypatch.setitem(sys.modules, "cv2", ModuleType("cv2"))
    monkeypatch.setenv("XR", "previous")
    monkeypatch.setenv("XR_RUNTIME_JSON", "/previous.json")
    monkeypatch.setattr(sys, "argv", ["teleop_omy_xr.py"])
    path = Path(__file__).resolve().parents[2] / "examples/teleop_omy_xr.py"
    spec = importlib.util.spec_from_file_location("omy_xr_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "preflight", lambda args: {
        "mode": "steamvr", "runtime_json": "/steamxr.json", "errors": [],
    })
    assert module.main() == expected
    assert calls == [expected, "close"]
    assert os.environ["XR"] == "previous"
    assert os.environ["XR_RUNTIME_JSON"] == "/previous.json"
    if isinstance(outcome, Exception):
        assert "test failure" in capsys.readouterr().err


@pytest.mark.parametrize("scenario", ["reconnect", "startup_timeout", "step_failure", "diagnostic_failure"])
@pytest.mark.parametrize("record", [False, True])
def test_real_runner_gates_input_and_cleans_up_without_reset_on_reconnect(tmp_path, monkeypatch, scenario, record):
    """Execute the actual runner with fake environment/device and real XR gate."""
    import torch

    from robolab.core.omy_teleop.performance import PerformanceMonitor

    events = []
    view = FakeView()
    cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=1 / 120), decimation=2,
        viewer=SimpleNamespace(), teleop_devices=SimpleNamespace(devices={}),
        instruction="banana into bowl",
    )
    device = SimpleNamespace(
        advance=lambda: events.append("advance") or torch.zeros(8),
        reset=lambda: events.append("device_reset"),
        close=lambda: events.append("device_close"),
    )

    def create_device(*args):
        assert view.active, "input was opened before headset activation"
        events.append("device_open")
        return device

    def render():
        events.append("render")
        view.active = scenario != "startup_timeout"

    def step(action):
        assert view.active and action.shape == (1, 8)
        events.append("step")
        if scenario == "step_failure":
            raise RuntimeError("simulation failed")
        if scenario == "diagnostic_failure":
            from robolab.core.omy_teleop.diagnostics import DiagnosticFailure

            raise DiagnosticFailure("left_inner_finger_joint: outside hard limits")
        view.active = False  # simulate a display loss during every step

    dataset_attrs = {}
    recorder = SimpleNamespace(
        **{name: lambda *a, **kw: None for name in (
            "set_buffered_recording", "set_hdf5_compression", "set_flush_interval",
            "set_flush_memory_cleanup", "set_hdf5_file", "set_episode_index",
            "set_episode_seed", "set_success_to_episodes",
        )},
        cfg=SimpleNamespace(), set_dataset_attrs=dataset_attrs.update,
        export_episodes=lambda *a: events.append("export"),
    )
    env = SimpleNamespace(
        device="cpu", step_dt=1 / 60, cfg=cfg, all_terminated=False,
        sim=SimpleNamespace(render=render, pause=lambda: events.append("physics_pause")),
        recorder_manager=recorder,
        observation_manager=SimpleNamespace(compute=lambda **kwargs: {}),
        reset=lambda **kwargs: events.append("env_reset"), step=step,
        close=lambda: events.append("env_close"),
    )

    def stub(name, **attrs):
        module = ModuleType(name)
        vars(module).update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    stub("isaaclab.devices.teleop_device_factory", create_teleop_device=create_device)
    stub("isaaclab.managers", DatasetExportMode=SimpleNamespace(EXPORT_NONE=0, EXPORT_ALL=1, EXPORT_SUCCEEDED_ONLY=2),
         RecorderManagerBaseCfg=lambda **kw: SimpleNamespace(**kw))
    stub("robolab.core.environments.config", parse_env_cfg=lambda *a, **kw: cfg)
    stub("robolab.core.environments.factory", get_envs=lambda **kw: ["TaskOMYL100"])
    stub("robolab.core.environments.runtime", create_env=lambda *a, **kw: (env, cfg))
    stub("robolab.core.omy_teleop.config", ENV_POSTFIX="OMYL100",
         configure_device=lambda *a: SimpleNamespace(dt=1 / 60),
         configure_performance=lambda *a: None, register_omy_envs=lambda *a, **kw: None,
         validate_environment=lambda env: range(7))
    stub("carb", settings=SimpleNamespace(get_settings=lambda: SimpleNamespace(set_bool=lambda *a: None)))
    path = Path(__file__).resolve().parents[2] / "robolab/core/omy_teleop/runner.py"
    spec = importlib.util.spec_from_file_location("robolab.core.omy_teleop._xr_runner_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "performance_settings", lambda *a: {})
    monkeypatch.setattr(runner, "upstream_metadata", lambda: {})
    class FakeDiagnostics:
        def __init__(self, *a, **kw):
            pass

        def install(self):
            events.append("diagnostics_install")

        def start_episode(self, *a):
            events.append("diagnostics_reset")

        def begin_control(self, *a):
            events.append("diagnostics_begin")

        def end_control(self, *, incomplete=False):
            assert incomplete
            events.append("diagnostics_fault")

        def close(self):
            events.append("diagnostics_close")

    monkeypatch.setattr(runner, "RuntimeDiagnostics", FakeDiagnostics)
    monkeypatch.setattr(PerformanceMonitor, "install", lambda *a: None)
    monkeypatch.setattr(runner.robolab.constants, "set_output_dir", lambda *a: None)
    monkeypatch.setattr(runner.robolab.constants, "RECORD_IMAGE_DATA", False)
    args = args_for("--source", "tcp", "--steps", "3", "--no-realtime", "--output-dir", str(tmp_path),
                    *(["--record", "--save-failures"] if record else []),
                    *(["--diagnostics"] if scenario == "diagnostic_failure" else []))
    times = iter([0.0, 61.0])
    session = session_for(view, clock=(lambda: next(times)) if scenario == "startup_timeout" else lambda: 0.0)
    if scenario == "reconnect":
        assert runner.run(args, SimpleNamespace(is_running=lambda: True), xr_session=session) == 0
    elif scenario == "diagnostic_failure":
        assert runner.run(args, SimpleNamespace(is_running=lambda: True), xr_session=session) == 2
        assert events.index("diagnostics_fault") < events.index("physics_pause") < events.index("device_close")
        assert "diagnostics_close" in events
    else:
        with pytest.raises(RuntimeError, match="inactive" if scenario == "startup_timeout" else "simulation failed"):
            runner.run(args, SimpleNamespace(is_running=lambda: True), xr_session=session)
    if scenario == "startup_timeout":
        assert "device_open" not in events and "advance" not in events and "step" not in events
    else:
        assert events.index("render") < events.index("device_open") < events.index("advance")
        assert events.count("step") == events.count("advance") == (3 if scenario == "reconnect" else 1)
        assert events.count("device_reset") == events.count("env_reset") == 1
        assert events.index("device_close") < events.index("env_close")
        if record:
            assert events.index("export") < events.index("device_close")
    assert view.closed
    status = json.loads((tmp_path / "xr_status.json").read_text())
    assert status["completed"] == (scenario == "reconnect")
    assert status["xr_display_seen"] == (scenario != "startup_timeout")
    assert status["display_waits"] == (3 if scenario == "reconnect" else 1)
    if record:
        metadata = json.loads(dataset_attrs["robolab_omy_teleop"])
        assert metadata["action_type"] == "absolute_joint_position"
        assert metadata["calibration"]["contents"]["omy_sign"]["joint_6"] == -1.0
        assert metadata["viewing"]["mode"] == "steamvr"
