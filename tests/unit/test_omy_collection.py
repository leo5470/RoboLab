"""Collector READY and Space transitions, independent of Isaac Sim."""

from types import SimpleNamespace

import numpy as np
import pytest

from robolab.core.omy_teleop.collection import ReadyGate, SpaceReset, build_parser, validate_args


def test_ready_requires_fresh_stable_calibrated_leader_and_movement():
    now = [0.0]
    reading = {f"joint_{i}.pos": 0.0 for i in range(1, 7)}
    reading["gripper.pos"] = 0.0
    age = [0.01]
    device = SimpleNamespace(
        cfg=SimpleNamespace(stale_s=0.5),
        _latest=lambda: (reading, age[0]),
        _mapper=SimpleNamespace(raw_target=lambda r: np.array([r[f"joint_{i}.pos"] for i in range(1, 7)] + [0.0]),
                                gripper_command=lambda x: 1.0 if x < 0.1 else -1.0),
    )
    gate = ReadyGate(device, np.zeros(7), clock=lambda: now[0])
    assert not gate.observe_waiting() and not gate.confirm()
    now[0] = 0.31
    assert gate.observe_waiting() and gate.confirm()
    assert not gate.moved()
    reading["joint_1.pos"] = 0.04
    assert gate.moved()
    age[0] = 0.6
    assert not gate.moved() and not gate.confirm()
    age[0] = 0.01
    reading["joint_1.pos"] = 0.0
    reading["gripper.pos"] = 0.12
    assert not gate.observe_waiting()


def test_space_reset_is_press_edge_and_can_be_cleared(monkeypatch):
    import sys

    callbacks = []
    input_api = SimpleNamespace(
        subscribe_to_keyboard_events=lambda keyboard, callback: callbacks.append(callback) or 7,
        unsubscribe_from_keyboard_events=lambda keyboard, sub: callbacks.append((keyboard, sub)),
    )
    key = SimpleNamespace(get_keyboard=lambda: "keyboard")
    types = SimpleNamespace(KEY_PRESS=1, KEY_RELEASE=2)
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(input=SimpleNamespace(
        acquire_input_interface=lambda: input_api, KeyboardEventType=types)))
    monkeypatch.setitem(sys.modules, "carb.input", sys.modules["carb"].input)
    monkeypatch.setitem(sys.modules, "omni", SimpleNamespace(appwindow=SimpleNamespace(
        get_default_app_window=lambda: key)))
    monkeypatch.setitem(sys.modules, "omni.appwindow", sys.modules["omni"].appwindow)
    reset = SpaceReset()
    send = lambda event_type: callbacks[0](SimpleNamespace(
        input=SimpleNamespace(name="SPACE"), type=event_type))
    send(1)
    send(1)
    assert reset.take() and not reset.take()
    send(2)
    send(1)
    reset.clear()
    assert not reset.take()
    reset.close()
    assert callbacks[-1] == ("keyboard", 7)


def test_collection_defaults_to_state_only_and_fixed_calibration():
    parser = build_parser()
    args = parser.parse_args(["--source", "tcp"])
    validate_args(parser, args)
    assert args.record and args.save_failures and not args.record_images
    assert args.num_episodes == 1000
    assert args.max_attempt_seconds == 20.0
    with pytest.raises(SystemExit):
        validate_args(parser, parser.parse_args(["--auto-zero"]))
    with pytest.raises(SystemExit):
        validate_args(parser, parser.parse_args(["--record-images"]))
    with pytest.raises(SystemExit):
        validate_args(parser, parser.parse_args(["--max-attempt-seconds", "0"]))
