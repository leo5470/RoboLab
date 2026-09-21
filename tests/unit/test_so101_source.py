"""IPC validation and actual helper-process tests; no USB or simulator."""

import json
import sys
import time
from pathlib import Path

import pytest

from robolab.core.teleop.so101_source import JOINT_NAMES, LeaderSource, SampleBuffer, SourceFault, StaleSample


def packet(**changes):
    data = dict(
        version=1,
        session_id="test",
        sequence=0,
        read_start=10.0,
        read_end=10.01,
        joint_names=list(JOINT_NAMES),
        joints_deg=[0] * 5,
        gripper=100,
        position=[0.2, 0, 0.2],
        quaternion=[1, 0, 0, 0],
    )
    data.update(changes)
    return json.dumps(data).encode()


def test_duplicate_and_reordered_samples_do_not_refresh_age():
    buffer = SampleBuffer(0.2)
    buffer.accept(packet(sequence=2), 10.02)
    assert not buffer.accept(packet(sequence=1, read_start=10.1, read_end=10.11), 10.12)
    assert not buffer.accept(packet(sequence=2, read_start=10.1, read_end=10.11), 10.12)
    assert buffer.current(10.19).sequence == 2
    with pytest.raises(StaleSample):
        buffer.current(10.21)


@pytest.mark.parametrize(
    "changes",
    [
        {"quaternion": [0, 0, 0, 0]},
        {"position": [float("nan"), 0, 0]},
        {"joints_deg": [0] * 6},
        {"read_start": 11},
        {"read_end": 11},
        {"sequence": -1},
        {"sequence": True},
        {"version": 2},
        {"gripper": float("inf")},
        {"joint_names": list(reversed(JOINT_NAMES))},
    ],
)
def test_invalid_data_latches_fault(changes):
    buffer = SampleBuffer()
    buffer.accept(packet(**changes), 10.02)
    with pytest.raises(SourceFault):
        buffer.current(10.03)
    buffer.accept(packet(sequence=3), 10.04)
    with pytest.raises(SourceFault):
        buffer.current(10.05)


def test_session_restart_is_a_fault():
    buffer = SampleBuffer()
    buffer.accept(packet(), 10.02)
    buffer.accept(packet(session_id="restart", sequence=1), 10.03)
    with pytest.raises(SourceFault, match="session"):
        buffer.current(10.04)


def wait_for_sample(source, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return source.current()
        except StaleSample:
            time.sleep(0.01)
    pytest.fail("Helper did not produce a fresh sample")


def test_real_mock_helper_ipc_and_cleanup():
    source = LeaderSource(sys.executable, ["--mock", "--read-hz", "60"])
    directory = Path(source.directory.name)
    try:
        first = wait_for_sample(source)
        assert source.metadata["mock"]
        time.sleep(0.10)
        latest = source.current()
        assert latest.sequence > first.sequence
        assert latest.gripper == 100
        assert source.clock() - latest.read_start < 0.2
        source.process.kill()
        source.process.wait(timeout=3)
        with pytest.raises(SourceFault, match="exited"):
            source.current()
    finally:
        source.close()
    assert source.process.poll() is not None
    assert not directory.exists()
    source.close()


def test_start_failure_releases_ipc():
    with pytest.raises(OSError):
        LeaderSource("/definitely/missing/python", ["--mock"])
