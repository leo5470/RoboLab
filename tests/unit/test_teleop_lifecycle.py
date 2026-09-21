"""Pure control-state tests; no simulator or live stream is needed."""

import pytest

from robolab.core.teleop.lifecycle import AttemptControls


def test_idle_retry_does_not_reject_the_next_attempt():
    controls = AttemptControls()
    assert not controls.request_retry()
    controls.request_start()
    # Also ignore R received in the same input batch as N, before begin().
    assert not controls.request_retry()
    controls.begin()
    assert controls.running
    assert not controls.retry_requested


def test_active_retry_then_reset_requires_a_fresh_start():
    controls = AttemptControls()
    controls.request_start()
    controls.begin()
    assert controls.request_retry()
    controls.arm()
    assert not controls.running
    assert not controls.start_requested
    assert not controls.retry_requested
    controls.request_start()
    controls.begin()
    assert controls.running
    assert not controls.retry_requested


def test_timeout_can_be_followed_by_multiple_new_attempts():
    controls = AttemptControls()
    for _ in range(4):
        controls.request_start()
        controls.begin()
        controls.request_start()  # N during recording must not start a later attempt
        controls.arm()  # Collector resets/arms after timeout or success
        assert not controls.running
        assert not controls.start_requested
        assert not controls.retry_requested


def test_recording_requires_an_explicit_start():
    controls = AttemptControls()
    with pytest.raises(RuntimeError, match="armed and started"):
        controls.begin()
