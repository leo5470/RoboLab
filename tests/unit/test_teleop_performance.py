"""Test timing attribution and missing stream telemetry without Isaac Sim."""

from types import SimpleNamespace

import pytest

from robolab.core.teleop.performance import LoopTiming, RealtimePacer, StreamQuality, worker_settings


def test_wifi_cpu_worker_defaults_are_session_only(monkeypatch):
    monkeypatch.setattr("os.sched_getaffinity", lambda _: set(range(40)))
    assert worker_settings("wifi", "cpu") == {
        "/plugins/carb.tasking.plugin/threadCount": 16,
        "/plugins/omni.tbb.globalcontrol/maxThreadCount": 16,
        "/persistent/physics/numThreads": 2,
        "/app/settings/persistent": "false",
    }
    monkeypatch.setattr("os.sched_getaffinity", lambda _: {1, 3, 5, 7})
    assert worker_settings("wifi", "cpu")["/plugins/carb.tasking.plugin/threadCount"] == 4


def test_worker_overrides_and_stock_profiles():
    assert worker_settings("standard", "cpu") == {}
    assert worker_settings("wifi", "cuda:0") == {}
    assert worker_settings("wifi", "cpu", 0, 0) == {}
    assert worker_settings("wifi", "cpu", 0, 4) == {
        "/persistent/physics/numThreads": 4, "/app/settings/persistent": "false",
    }
    with pytest.raises(ValueError, match="non-negative"):
        worker_settings("wifi", "cpu", -1)


def test_timing_splits_physics_render_other_and_counts_deadline_misses(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("robolab.core.teleop.performance.time.perf_counter", lambda: now[0])

    def render():
        now[0] += .010

    def step(render=False):
        now[0] += .020
        if render:
            sim.render()

    sim = SimpleNamespace(render=render, step=step)
    timing = LoopTiming()
    timing.install(sim)
    sim.step(render=True)
    timing.record_step(.050)
    sim.step()
    sim.render()
    timing.record_step(.080)
    summary = timing.summary(.0667)
    assert summary["physics_ms_mean"] == pytest.approx(20)
    assert summary["render_ms_mean"] == pytest.approx(10)
    assert summary["other_ms_mean"] == pytest.approx(35)
    assert summary["step_ms_max"] == pytest.approx(80)
    assert summary["over_budget_steps"] == 1
    timing.clear()
    assert timing.summary() == {}
    assert timing.physics_seconds == 0
    timing.close()
    assert sim.step is step
    assert sim.render is render


def test_timing_records_and_restores_on_exception(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("robolab.core.teleop.performance.time.perf_counter", lambda: now[0])
    def step():
        now[0] += .03
        raise ValueError("physics error")
    sim = SimpleNamespace(render=lambda: None, step=step)
    timing = LoopTiming()
    timing.install(sim)
    with pytest.raises(ValueError, match="physics error"):
        sim.step()
    assert timing.physics_seconds == pytest.approx(.03)
    timing.close()
    assert sim.step is step


def test_missing_qos_is_not_reported_as_zero_latency_or_zero_size(monkeypatch):
    monkeypatch.setattr("robolab.core.teleop.performance.time.perf_counter", lambda: 10)
    quality = StreamQuality.__new__(StreamQuality)
    quality.latest = (9, dict(rtt_ms=0, bitrate_mbps=12.5, width=0, height=0))
    summary = quality.summary()
    assert "RTT unavailable" in summary
    assert "0x0" not in summary
    assert "bitrate target 12.5 Mbps" in summary
    quality.latest = (9, dict(rtt_ms=25, bitrate_mbps=8.0, width=960, height=540))
    assert "RTT 25 ms" in quality.summary()
    assert "requested size 960x540" in quality.summary()
    quality.latest = None
    assert "awaiting" in quality.summary()


def test_pacer_does_not_burst_actions_after_overrun():
    now = [0.0]
    def sleep(duration):
        now[0] += duration
    pacer = RealtimePacer(.1, clock=lambda: now[0], sleep=sleep)
    pacer.wait()
    now[0] = .4
    pacer.wait()
    pacer.wait()
    assert now[0] == pytest.approx(.5)


def test_region_means_roll_over_with_the_step_window():
    timing = LoopTiming(capacity=2)
    for physics in (.01, .02, .04):
        timing.physics_seconds += physics
        timing.render_seconds += .01
        timing.renders += 2
        timing.record_step(physics + .03)
    summary = timing.summary(.0667)
    assert summary["physics_ms_mean"] == pytest.approx(30)
    assert summary["render_ms_mean"] == pytest.approx(10)
    assert summary["render_call_ms_mean"] == pytest.approx(5)
    assert summary["other_ms_mean"] == pytest.approx(20)
    assert summary["over_budget_steps"] == 1
