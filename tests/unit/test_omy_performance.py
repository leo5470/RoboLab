"""OMY performance controls without importing Isaac Sim or another collector."""

import json
from types import SimpleNamespace

import pytest

from robolab.core.omy_teleop.options import build_parser, validate_args
from robolab.core.omy_teleop.contact_profile import selected_contact_sensors
from robolab.core.omy_teleop.performance import (
    PerformanceMonitor, RenderPacer, capture_due, prepare_launcher, render_interval,
)
from scripts.compare_omy_performance import load_run


@pytest.mark.parametrize("argv, expected", [
    ([], 2), (["--stream"], 4), (["--stream", "--video-every", "1"], 2),
    (["--stream", "--video-every", "3"], 6),
    (["--stream", "--record", "--record-images"], 2),
    (["--stream", "--render-interval", "2"], 2),
    (["--stream", "--render-interval", "4", "--video-every", "4"], 4),
])
def test_render_cadence(argv, expected):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    assert render_interval(args) == expected


@pytest.mark.parametrize("hz, renders_per_command", [(15, 4), (30, 2), (60, 1)])
def test_xr_render_phase_stays_at_two_physics_steps(hz, renders_per_command):
    parser = build_parser()
    args = parser.parse_args(["--control-hz", str(hz), "--render-interval", "2"])
    validate_args(parser, args)
    assert render_interval(args) == 2
    assert (120 // hz) // render_interval(args) == renders_per_command


def test_required_contacts_keep_task_dependencies_and_other_tasks_full():
    banana = ["gripper__all_objs", "gripper__banana", "gripper__bowl", "gripper__table",
              "banana__bowl", "banana__table", "bowl__table"]
    mode, retained = selected_contact_sensors("BananaInBowlTask", banana, "required")
    assert mode == "required"
    assert retained == set(banana) - {"bowl__table"}
    animals = ["gripper__all_objs", "gripper__lizard_figurine", "gripper__grey_bin",
               "lizard_figurine__grey_bin", "table__grey_bin"]
    mode, retained = selected_contact_sensors("AnimalsInBinTask", animals, "required")
    assert mode == "required" and retained == set(animals[:3]) | {"table__grey_bin"}
    assert selected_contact_sensors("OtherTask", banana, "required") == ("full", set(banana))
    assert selected_contact_sensors("BananaInBowlTask", banana, "full") == ("full", set(banana))


def test_comparison_reads_contact_profile_and_batch_counters(tmp_path):
    (tmp_path / "omy_performance.json").write_text(json.dumps({
        "settings": {"contact_sensor_profile": "required", "contact_sensor_count": 16,
                     "physics_device": "cuda:0", "kit": {"/renderer/activeGpu": 0}},
        "last_window": {"control_hz": 12.0, "real_time_factor": 0.2,
                        "work_ms_p95": 100.0, "region_ms_mean": {"physics": 20.0}},
        "contact_cache": {"computed": 20, "reused": 7, "fallback": 2},
    }))
    (tmp_path / "env_cfg.json").write_text(json.dumps({
        "_task_name": "AnimalsInBinTask", "sim": {"gravity": [0, 0, -9.81],
        "dt": 1 / 120, "render_interval": 2}, "decimation": 2,
    }))
    row = load_run(tmp_path)
    assert (row["profile"], row["sensors"]) == ("required", 16)
    assert (row["batch_computed"], row["batch_reused"], row["batch_fallback"]) == (20, 7, 2)


def test_cpu_launch_defaults_and_explicit_overrides(monkeypatch):
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(range(20)))
    args = SimpleNamespace(performance_profile="cpu", device="cuda:0", kit_args="--/custom/setting=true")
    prepare_launcher(args)
    assert args.device == "cpu"
    assert "--/plugins/carb.tasking.plugin/threadCount=8" in args.kit_args
    assert "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8" in args.kit_args
    assert "--/persistent/physics/numThreads=2" in args.kit_args
    assert "--/renderer/multiGpu/enabled=false" in args.kit_args
    assert "--/app/settings/persistent=false" in args.kit_args
    assert args.kit_args.endswith("--/custom/setting=true")
    args = SimpleNamespace(performance_profile="cpu", device="cpu", device_explicit=True,
                           kit_args="--/plugins/carb.tasking.plugin/threadCount=4")
    prepare_launcher(args)
    assert args.kit_args.endswith("--/plugins/carb.tasking.plugin/threadCount=4")
    args = SimpleNamespace(performance_profile="cpu", device="cuda:1", device_explicit=True, kit_args="custom")
    prepare_launcher(args)
    assert args.device == "cuda:1" and args.kit_args == "custom"


def test_standard_profile_is_unchanged():
    args = SimpleNamespace(performance_profile="standard", device="cuda:1", kit_args="custom")
    before = vars(args).copy()
    prepare_launcher(args)
    assert vars(args) == before


def test_decimated_stream_has_one_pacer_and_keeps_explicit_kit_override():
    args = build_parser().parse_args(["--stream"])
    args.kit_args = "--/app/runLoops/main/rateLimitEnabled=true"
    prepare_launcher(args)
    assert args.kit_args.startswith("--/app/runLoops/main/rateLimitEnabled=false")
    assert args.kit_args.endswith("--/app/runLoops/main/rateLimitEnabled=true")
    args = build_parser().parse_args(["--stream", "--record", "--record-images"])
    prepare_launcher(args)
    assert not hasattr(args, "kit_args")
    args = build_parser().parse_args(["--control-hz", "15"])
    prepare_launcher(args)
    assert "--/app/runLoops/main/rateLimitEnabled=false" in args.kit_args


def test_worker_counts_respect_cpu_affinity(monkeypatch):
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: {1})
    args = SimpleNamespace(performance_profile="cpu")
    prepare_launcher(args)
    assert "threadCount=1" in args.kit_args
    assert "numThreads=1" in args.kit_args


@pytest.mark.parametrize("reset_physics_step", [0, 2, 6, 10])
def test_capture_follows_actual_render_phase_across_odd_resets(reset_physics_step):
    emitted = []
    for step in range(reset_physics_step + 2, reset_physics_step + 14, 2):
        rendered = int(step % 4 == 0)
        if capture_due(step, decimation=2, video_every=2, renders_before=10, renders_after=10 + rendered):
            emitted.append(step)
    assert emitted
    assert all(step % 4 == 0 for step in emitted)
    assert all(b - a == 4 for a, b in zip(emitted, emitted[1:]))
    assert not capture_due(4, decimation=2, video_every=2, renders_before=10, renders_after=10)


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def test_low_rate_render_pacer_spaces_frames_and_restores_renderer():
    clock = Clock()
    rendered = []

    class Sim:
        def render(self):
            rendered.append(clock.now)

    sim = Sim()
    pacer = RenderPacer(sim, 1 / 60, clock=clock, sleep=lambda seconds: setattr(clock, "now", clock.now + seconds))
    sim.render()  # READY rendering is never paced.
    with pacer.step():
        for _ in range(4):
            sim.render()
    assert rendered[0] == 0
    assert rendered[1:] == pytest.approx([1 / 60, 2 / 60, 3 / 60, 4 / 60])
    pacer.close()
    assert "render" not in vars(sim)


def test_render_interval_metrics_and_pacing_excluded_from_work():
    clock = Clock()
    monitor = PerformanceMonitor(clock=clock)

    class Sim:
        def render(self):
            clock.now += 0.002

    sim = Sim()
    monitor.wrap(sim, "render", "render")
    monitor.begin_step()
    sim.render()
    with monitor.measure("pacing"):
        clock.now += 0.014
    sim.render()
    monitor.end_step(work_end=clock.now)
    summary = monitor.summary(1 / 30)
    assert summary["render_interval_ms_mean"] == pytest.approx(16)
    assert summary["render_interval_ms_p95"] == pytest.approx(16)
    assert summary["work_ms_mean"] == pytest.approx(4)
    assert summary["region_ms_mean"]["pacing"] == pytest.approx(14)
    monitor.close()


def test_exclusive_timings_restore_wrappers_and_preserve_exceptions():
    clock = Clock()
    monitor = PerformanceMonitor(clock=clock)

    class Sim:
        def render(self):
            clock.now += 0.004
            return 7

        def step(self):
            clock.now += 0.003
            assert self.render() == 7
            raise ValueError("preserved")

    sim = Sim()
    monitor.wrap(sim, "render", "render")
    monitor.wrap(sim, "step", "physics")
    with pytest.raises(ValueError, match="preserved"):
        sim.step()
    assert monitor.seconds["physics"] == pytest.approx(0.003)
    assert monitor.seconds["render"] == pytest.approx(0.004)
    assert monitor.calls == {"physics": 1, "render": 1}
    monitor.close()
    monitor.close()
    assert vars(sim) == {}
    assert sim.render() == 7
    assert monitor.calls["render"] == 1


def test_bounded_window_rates_include_pacing_but_work_does_not():
    clock = Clock()
    monitor = PerformanceMonitor(capacity=2, clock=clock)
    for index in range(3):
        monitor.begin_step()
        with monitor.measure("physics"):
            clock.now += 0.003
        if index % 2 == 0:
            with monitor.measure("render"):
                clock.now += 0.004
        work_end = clock.now
        clock.now = (index + 1) * 0.02
        monitor.end_step(work_end=work_end, video_frames=int(index % 2 == 0))
    summary = monitor.summary(1 / 60)
    assert summary["window_steps"] == 2
    assert summary["control_hz"] == pytest.approx(50)
    assert summary["render_fps"] == pytest.approx(25)
    assert summary["produced_video_fps"] == pytest.approx(25)
    assert summary["work_ms_mean"] == pytest.approx(5)
    assert summary["work_ms_p95"] == pytest.approx(7)
    assert summary["region_ms_mean"]["physics"] == pytest.approx(3)
    assert summary["region_ms_mean"]["render"] == pytest.approx(2)
    assert monitor.total_steps == 3 and monitor.total_video_frames == 2
    assert "video produced" in monitor.report(1 / 60)
    monitor.reset_window()
    assert monitor.summary(1 / 60) == {}
    clock.now += 20  # startup/reset time must not enter the next window
    monitor.begin_step()
    clock.now += 0.01
    monitor.end_step(work_end=clock.now)
    assert monitor.summary(1 / 60)["control_hz"] == pytest.approx(100)
    assert monitor.total_steps == 4


def test_nonadmitted_stale_input_is_excluded_and_percentiles_are_reported():
    clock = Clock()
    monitor = PerformanceMonitor(clock=clock)
    monitor.begin_step()
    clock.now += 0.5
    monitor.end_step(work_end=clock.now, admitted=False)
    assert monitor.summary(1 / 15) == {} and monitor.total_steps == 0
    for work in (0.04, 0.06, 0.08):
        monitor.begin_step()
        clock.now += work
        monitor.end_step(work_end=clock.now)
    summary = monitor.summary(1 / 15)
    assert summary["work_ms_p50"] == pytest.approx(60)
    assert summary["work_ms_p99"] == pytest.approx(80)
    assert summary["command_interval_ms_p95"] == pytest.approx(80)
    assert summary["missed_command_budgets"] == 1


def test_render_pacer_recovers_after_a_stall_without_a_catchup_burst():
    clock = Clock()
    rendered = []
    class Sim:
        def render(self):
            rendered.append(clock.now)
    sim = Sim()
    pacer = RenderPacer(sim, 1 / 30, clock=clock, sleep=lambda delay: setattr(clock, "now", clock.now + delay))
    with pacer.step():
        sim.render()
        clock.now += 2.0
        recovery = clock.now
        sim.render()
        sim.render()
    assert rendered[1] == pytest.approx(recovery + 1 / 30)
    assert rendered[2] - rendered[1] == pytest.approx(1 / 30)
    pacer.close()
