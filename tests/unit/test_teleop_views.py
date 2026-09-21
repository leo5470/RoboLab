"""Pure layout checks; GPU rendering is exercised by the collector benchmark."""

import sys
from types import SimpleNamespace

import pytest

from robolab.core.teleop.views import OperatorInset, inset_resolution, pin_viewport_resolution, wrist_parent_path


@pytest.mark.parametrize("stream, expected", [
    ((960, 540), (320, 180)),
    ((1280, 720), (426, 240)),
    ((640, 360), (212, 120)),
    ((240, 180), (80, 60)),
])
def test_inset_stays_small_and_even(stream, expected):
    assert inset_resolution(*stream) == expected
    width, height = expected
    assert width % 2 == height % 2 == 0
    assert width * height <= stream[0] * stream[1] / 9


@pytest.mark.parametrize("stream", [(64, 64), (960, 100), (959, 540), (960, 539)])
def test_invalid_stream_size_is_rejected(stream):
    with pytest.raises(ValueError, match="at least 240x180"):
        inset_resolution(*stream)


def test_toggle_stops_rendering_not_just_display():
    inset = OperatorInset.__new__(OperatorInset)
    inset.camera = "wrist"
    inset.visible = True
    inset._frame = SimpleNamespace(visible=True)
    inset._widget = SimpleNamespace(viewport_api=SimpleNamespace(updates_enabled=True))
    for expected in (False, True, False):
        inset.toggle()
        assert inset.visible == expected
        assert inset._frame.visible == expected
        assert inset.viewport_api.updates_enabled == expected


def test_close_disables_and_destroys_widget_and_is_idempotent():
    inset = OperatorInset.__new__(OperatorInset)
    calls = []
    api = SimpleNamespace(updates_enabled=True)

    def destroy():
        assert not api.updates_enabled
        calls.append("destroy")

    frame = SimpleNamespace(visible=True, clear=lambda: calls.append("clear"))
    inset._frame = frame
    inset._widget = SimpleNamespace(viewport_api=api, destroy=destroy)
    inset._stage = None
    inset._camera_path = None
    inset.visible = True
    inset.close()
    inset.close()
    assert calls == ["destroy", "clear"]
    assert not frame.visible
    assert not inset.visible
    assert inset.viewport_api is None


def test_pin_overrides_saved_texture_size_and_ui_auto_resize(monkeypatch):
    api = SimpleNamespace(resolution=(1280, 720), fill_frame=True, resolution_scale=0.5)

    class Widget:
        expand_viewport = True
        fill_frame = True

        @property
        def resolution(self):
            return api.resolution

        @resolution.setter
        def resolution(self, value):
            api.resolution = value

    widget = Widget()
    window = SimpleNamespace(viewport_widget=widget, viewport_api=api)
    monkeypatch.setitem(sys.modules, "omni.kit.viewport.utility",
                        SimpleNamespace(get_active_viewport_window=lambda: window))
    assert pin_viewport_resolution((960, 540)) == (960, 540)
    assert not widget.expand_viewport
    assert not widget.fill_frame
    assert not api.fill_frame
    assert api.resolution_scale == 1.0


def test_wrist_mount_uses_env_path_without_reusing_the_policy_camera():
    assert wrist_parent_path(
        "{ENV_REGEX_NS}/robot/Gripper/base_link/wrist_cam", "/World/envs/env_0"
    ) == "/World/envs/env_0/robot/Gripper/base_link"
    assert wrist_parent_path("/Robot/hand/wrist_cam", "/World/envs/env_0") == "/Robot/hand"


@pytest.mark.parametrize("path", ["relative/wrist_cam", "{UNKNOWN}/wrist_cam", "/wrist_cam"])
def test_invalid_wrist_mount_is_rejected(path):
    with pytest.raises(ValueError, match="Cannot resolve"):
        wrist_parent_path(path, "/World/envs/env_0")
