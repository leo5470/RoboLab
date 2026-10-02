"""Viewing-only preflight and native API lifecycle, without Isaac or hardware."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robolab.core.xr_view.native import NativeXRView
from robolab.core.xr_view.options import build_parser, find_runtime, launcher_options, preflight, validate_args
from robolab.core.xr_view.runner import prepare_output, render_loop


def runtime_fixture(tmp_path):
    manifest = tmp_path / "SteamVR" / "steamxr_linux64.json"
    manifest.parent.mkdir()
    (manifest.parent / "runtime.so").touch()
    manifest.write_text(json.dumps({"runtime": {"name": "SteamVR", "library_path": "runtime.so"}}))
    return manifest


def test_runtime_override_and_relative_library(tmp_path):
    manifest = runtime_fixture(tmp_path)
    assert find_runtime(manifest, environ={}) == manifest
    assert find_runtime(environ={"XR_RUNTIME_JSON": str(manifest)}) == manifest
    with pytest.raises(ValueError, match="Invalid OpenXR"):
        find_runtime(tmp_path / "missing.json", environ={"XR_RUNTIME_JSON": str(manifest)})


def test_runtime_active_symlink_resolves_library_relative_to_real_manifest(tmp_path):
    manifest = runtime_fixture(tmp_path)
    active = tmp_path / ".config/openxr/1/active_runtime.json"
    active.parent.mkdir(parents=True)
    active.symlink_to(manifest)
    assert find_runtime(environ={"XDG_CONFIG_DIRS": ""}, home=tmp_path) == manifest


@pytest.mark.parametrize("contents", ["not json", "{}", '{"runtime": {"library_path": 4}}'])
def test_bad_runtime_fails_before_simulator_import(tmp_path, contents):
    path = tmp_path / "steamxr.json"
    path.write_text(contents)
    with pytest.raises(ValueError, match="Invalid OpenXR"):
        find_runtime(path, environ={})


def test_runtime_library_missing(tmp_path):
    manifest = runtime_fixture(tmp_path)
    (manifest.parent / "runtime.so").unlink()
    with pytest.raises(ValueError, match="library not found"):
        find_runtime(manifest, environ={})


def test_cloudxr_not_silently_selected(tmp_path):
    library = tmp_path / "libcloudxr.so"
    library.touch()
    manifest = tmp_path / "runtime.json"
    manifest.write_text(json.dumps({"runtime": {"library_path": str(library)}}))
    with pytest.raises(ValueError, match="Expected a SteamVR"):
        find_runtime(manifest, environ={})


def test_preflight_and_launcher(tmp_path):
    manifest = runtime_fixture(tmp_path)
    args = build_parser().parse_args(["--runtime-json", str(manifest), "--headless"])
    assert args.eye == (-0.25, 0.0, 0.95)
    assert args.lookat == (0.53, 0.0, 0.48)
    assert preflight(args, environ={"DISPLAY": ":0"})["errors"] == []
    assert preflight(args, environ={})["errors"]
    assert preflight(args, environ={"DISPLAY": "localhost:10.0"})["errors"]
    launch = launcher_options(args)
    assert launch["xr"] and not launch["enable_cameras"]
    assert launch["video"]  # preserves the native Kit viewport, not a camera sensor
    assert "--enable omni.kit.xr.profile.vr" in launch["kit_args"]
    assert "/xr/profile/ar/enabled=false" in launch["kit_args"]
    assert "/app/settings/persistent=false" in launch["kit_args"]
    assert "--/rtx-transient/resourcemanager/enableTextureStreaming=false" in launch["kit_args"]
    assert launch["device"] == "cpu"
    args.desktop = True
    assert preflight(args, environ={})["errors"] == []
    assert not launcher_options(args)["xr"]
    assert "enableTextureStreaming" not in launcher_options(args)["kit_args"]
    assert launcher_options(args)["experience"] == "isaaclab.python.kit"
    args.desktop = False
    assert launcher_options(args)["experience"] == ""  # native XR experience selection


@pytest.mark.parametrize("argv", [
    ["--frames", "-1"], ["--render-gpu", "-1"], ["--xr-timeout", "nan"],
    ["--eye", "0", "0", "0", "--lookat", "0", "0", "1"],
    ["--eye", "inf", "0", "0"], ["--device", "cuda:bad"],
    ["--xr-resolution-multiplier", "nan"], ["--xr-resolution-multiplier", "inf"],
    ["--xr-resolution-multiplier", "0.09"], ["--xr-resolution-multiplier", "2.01"],
    ["--desktop", "--xr-resolution-multiplier", "0.8"],
])
def test_invalid_options(argv):
    parser = build_parser()
    with pytest.raises(SystemExit):
        validate_args(parser, parser.parse_args(argv))


class Settings(dict):
    def set(self, key, value):
        self[key] = value

    def destroy_item(self, key):
        self.pop(key, None)


class Core:
    def __init__(self):
        self.enabled = self.display = False
        self.profile = "vr"
        self.calls = []

    def request_enable_profile(self, name):
        self.calls.append(("enable", name))

    def request_disable_profile(self):
        self.calls.append(("disable",))

    def is_xr_enabled(self):
        return self.enabled

    def is_xr_display_enabled(self):
        return self.display

    def get_current_profile_name(self):
        return self.profile

    def schedule_set_camera(self, transform):
        self.calls.append(("camera", transform))


def test_native_api_activation_reconnect_and_cleanup():
    core = Core()
    settings = Settings({"/persistent/xr/system/openxr/runtime": "previous",
                         "/persistent/xr/profile/vr/adjustForUserHeight": True})
    view = NativeXRView(core, settings, "/runtime.json", "camera-matrix")
    view.start()
    assert settings["/persistent/xr/system/openxr/runtime"] == "custom"
    assert settings["/xr/profile/vr/tools/layout"] == "robolab_view_only"
    assert settings["/persistent/xr/profile/vr/adjustForUserHeight"] is False
    assert not view.update()
    core.enabled = True
    assert not view.update()  # profile enabled is not proof of headset display
    core.display = True
    assert view.update() and view.ever_active
    assert not view.update()
    core.display = False
    assert view.update()
    core.display = True
    assert view.update()
    assert core.calls == [("enable", "vr"), ("camera", "camera-matrix"), ("camera", "camera-matrix")]
    view.close()
    assert core.calls[-1] == ("disable",)
    assert settings["/persistent/xr/system/openxr/runtime"] == "previous"
    assert "/xr/profile/vr/tools/layout" not in settings
    assert settings["/persistent/xr/profile/vr/adjustForUserHeight"] is True
    assert settings["/xr/profile/vr/enabled"] is False


def test_unexpected_profile_rejected():
    core = Core()
    core.enabled = core.display = True
    core.profile = "ar"
    with pytest.raises(RuntimeError, match="another XR profile"):
        NativeXRView(core, Settings(), "runtime", "camera").update()


@pytest.mark.parametrize("value", [0.1, 0.8, 2.0])
def test_resolution_multiplier_valid_options(value):
    parser = build_parser()
    args = parser.parse_args(["--xr-resolution-multiplier", str(value)])
    validate_args(parser, args)
    assert args.xr_resolution_multiplier == value
    assert parser.parse_args([]).xr_resolution_multiplier is None


@pytest.mark.parametrize("previous", [None, 0.5, 1.0])
@pytest.mark.parametrize("override", [None, 0.8])
def test_resolution_multiplier_override_and_restore(previous, override):
    key = "/persistent/xr/profile/vr/render/resolutionMultiplier"
    settings = Settings({key: previous} if previous is not None else {})
    core = Core()
    view = NativeXRView(core, settings, "runtime", "camera", resolution_multiplier=override)
    view.start()
    assert settings.get(key) == (previous if override is None else override)
    core.enabled = core.display = True
    view.update()
    core.display = False
    view.update()
    core.display = True
    view.update()
    assert settings.get(key) == (previous if override is None else override)
    view.close()
    assert settings.get(key) == previous
    assert (key in settings) == (previous is not None)


def test_render_only_loop_and_no_false_xr_success():
    calls = []
    app = SimpleNamespace(is_running=lambda: True)
    assert render_loop(app, lambda: calls.append("render"), None, frames=5) == 5
    assert calls == ["render"] * 5
    view = NativeXRView(Core(), Settings(), "runtime", "camera")
    with pytest.raises(RuntimeError, match="before an XR display"):
        render_loop(app, lambda: None, view, frames=1)
    times = iter([0, 1, 4])
    with pytest.raises(RuntimeError, match="did not become active"):
        render_loop(app, lambda: None, view, timeout=2, clock=lambda: next(times))


def test_existing_output_not_overwritten(tmp_path):
    output = tmp_path / "new"
    assert prepare_output(output) == output
    (output / "keep.txt").write_text("user data")
    with pytest.raises(ValueError, match="new or empty"):
        prepare_output(output)
    assert (output / "keep.txt").read_text() == "user data"


def test_no_teleop_dependencies_or_control_in_viewer():
    root = Path(__file__).resolve().parents[2]
    for path in (root / "robolab/core/xr_view").glob("*.py"):
        source = path.read_text()
        assert "import omy_leader" not in source
        assert "env.step(" not in source
        assert "get_input_device(" not in source
