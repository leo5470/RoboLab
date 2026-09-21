from dataclasses import replace

import numpy as np
import pytest

from robolab.core.teleop.lifecycle import LeaderControls
from robolab.core.teleop.local_controls import LocalControls
from robolab.core.teleop.retarget import PoseRetargeter, RetargetConfig
from robolab.core.teleop.so101_controller import LeaderController
from robolab.core.teleop.so101_source import LeaderSample, StaleSample


class Source:
    def __init__(self):
        self.sample = LeaderSample("test", 0, 0.0, 0.0, np.zeros(5), 100.0, np.zeros(3), np.array([1.0, 0, 0, 0]))
        self.stale = False

    def current(self):
        if self.stale:
            raise StaleSample("stale")
        return self.sample


def test_stale_data_requires_explicit_resume_and_new_references():
    now = [0.0]
    source = Source()
    controls = LeaderControls()
    controller = LeaderController(source, PoseRetargeter(RetargetConfig(filter_alpha=1)), controls, lambda: now[0])
    controls.request_start()
    controller.begin([0, 0, 0], [1, 0, 0, 0])
    assert controller.action([0, 0, 0], [1, 0, 0, 0]) is not None
    source.stale = True
    now[0] = 1.0
    assert controller.action([0, 0, 0], [1, 0, 0, 0]) is None
    source.stale = False
    source.sample = replace(source.sample, sequence=1, position=np.array([1.0, 2, 3]), read_start=2.0, read_end=2.0)
    now[0] = 2.0
    assert controller.action([0, 0, 0], [1, 0, 0, 0]) is None
    controls.request_start()
    np.testing.assert_array_equal(controller.action([0, 0, 0], [1, 0, 0, 0]), 0)
    assert controller.resumed
    assert controller.diagnostics["pause_count"][0] == 1
    assert controller.diagnostics["paused_seconds"][0] == 1.0
    assert controller.mapper.generation == 2


def test_clutch_gripper_mismatch_never_releases_object():
    source = Source()
    controls = LeaderControls()
    controller = LeaderController(source, PoseRetargeter(RetargetConfig()), controls, lambda: 1.0)
    controls.request_start()
    controller.begin([0, 0, 0], [1, 0, 0, 0])
    source.sample = replace(source.sample, sequence=1, gripper=0.0)
    assert controller.action([0, 0, 0], [1, 0, 0, 0])[6] == 1.0
    controls.request_clutch()
    source.sample = replace(source.sample, sequence=2, gripper=100.0)
    assert controller.action([0, 0, 0], [1, 0, 0, 0]) is None
    controls.request_clutch()
    assert controller.action([0, 0, 0], [1, 0, 0, 0]) is None
    assert controls.paused and not controls.resume_requested
    source.sample = replace(source.sample, sequence=3, gripper=0.0)
    controls.request_start()
    assert controller.action([0, 0, 0], [1, 0, 0, 0])[6] == 1.0


def test_retry_and_reset_discard_old_resume():
    controls = LeaderControls()
    controls.request_clutch()
    assert controls.paused is None
    controls.request_start()
    controls.begin()
    controls.request_clutch()
    controls.request_start()
    assert controls.request_retry()
    controls.arm()
    assert not controls.resume_requested and not controls.retry_requested and not controls.start_requested
    with pytest.raises(RuntimeError):
        controls.resume()


def test_evdev_split_packets_and_repeats(tmp_path):
    path = tmp_path / "events"
    path.touch()
    events = []
    control = LocalControls({"N": lambda: events.append("N")}, str(path))

    def packet(value):
        return control.EVENT.pack(0, 0, 1, 49, value)

    try:
        control.feed(packet(1)[:5])
        assert not events
        control.feed(packet(1)[5:] + packet(1) + packet(2))
        assert events == ["N"]
        control.feed(packet(0) + packet(1))
        assert events == ["N", "N"]
        with pytest.raises(RuntimeError, match="overflow"):
            control.feed(control.EVENT.pack(0, 0, 0, 3, 0))
    finally:
        control.close()


def test_stop_received_during_reset_is_not_lost():
    controls = LeaderControls()
    controls.request_stop()
    controls.arm()
    assert controls.stop_requested


@pytest.mark.parametrize("mode, code", [("fixed", 0), ("pose", 1), ("wrist-roll", 2)])
def test_orientation_mode_recording_codes(mode, code):
    controller = LeaderController(
        Source(), PoseRetargeter(RetargetConfig(orientation_mode=mode)), LeaderControls(), lambda: 1.0
    )
    controller.controls.request_start()
    controller.begin([0, 0, 0], [1, 0, 0, 0])
    assert controller.action([0, 0, 0], [1, 0, 0, 0]) is not None
    assert controller.diagnostics["orientation_mode"].tolist() == [code]
