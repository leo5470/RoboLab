"""OMY-only launch defaults, fresh-frame scheduling, and bounded host timings.

No simulator imports: these helpers can be tested without Isaac Sim. Region
times include synchronization, not isolated GPU-kernel execution. They never
insert CUDA synchronizations or change the environment's stepping logic.
"""

import math
import os
import time
from collections import Counter, deque
from contextlib import contextmanager
from functools import wraps


def prepare_launcher(args):
    """Apply the opt-in profile before AppLauncher starts Kit.

    Isaac Lab marks explicitly supplied --device arguments. Keep those, and
    preserve explicit Kit overrides by putting them after profile defaults.
    Nothing is written to environment variables or persistent preferences.
    """
    settings = {}
    if (getattr(args, "stream", False) or getattr(args, "control_hz", 60) < 60) and render_interval(args) > 2:
        # SimulationContext derives Kit's rate limit from rendering_dt. At a
        # lower control rate, an implicit Kit wait would compete with our
        # explicit control-loop pacing. Keep rendering_dt and physics_dt intact.
        settings["/app/runLoops/main/rateLimitEnabled"] = "false"
    if args.performance_profile == "cpu":
        if not getattr(args, "device_explicit", False):
            args.device = "cpu"
        if args.device == "cpu":
            try:
                available = len(os.sched_getaffinity(0))
            except (AttributeError, OSError):
                available = os.cpu_count() or 1
            settings.update({
                "/renderer/multiGpu/enabled": "false",
                "/plugins/carb.tasking.plugin/threadCount": min(8, available),
                "/plugins/omni.tbb.globalcontrol/maxThreadCount": min(8, available),
                "/persistent/physics/numThreads": min(2, available),
            })
    if not settings:
        return
    settings["/app/settings/persistent"] = "false"
    defaults = " ".join(f"--{key}={value}" for key, value in settings.items())
    args.kit_args = f"{defaults} {getattr(args, 'kit_args', '')}".strip()


def render_interval(args, decimation=None):
    if decimation is None:
        from .options import control_decimation

        decimation = control_decimation(args)
    if args.render_interval is not None:
        return args.render_interval
    return decimation * args.video_every if args.stream and not args.record_images else decimation


def capture_due(physics_step, *, decimation, video_every, renders_before, renders_after):
    """Use the simulator's actual phase, including across episode resets.

    Config validation makes render/video periods compatible. Require an actual
    render during this step as well, rather than sending a stale cached image.
    """
    return renders_after > renders_before and physics_step % (decimation * video_every) == 0


class RenderPacer:
    """Spread renders within a low-rate command while Isaac Lab steps physics.

    Only the active environment step is paced. Startup, READY, reset, and XR
    connection renders continue through the original simulator method.
    """

    def __init__(self, sim, period_s, *, monitor=None, clock=time.perf_counter, sleep=time.sleep):
        self.sim = sim
        self.period_s = period_s
        self.monitor = monitor
        self.clock = clock
        self.sleep = sleep
        self.original = sim.render
        self.had_local = "render" in vars(sim)
        self.local = vars(sim).get("render")
        self.active = False
        self.deadline = None

        @wraps(self.original)
        def paced(*args, **kwargs):
            if self.active:
                now = self.clock()
                if self.deadline is None or now - self.deadline > self.period_s:
                    self.deadline = now + self.period_s
                delay = self.deadline - now
                if delay > 0:
                    if self.monitor is None:
                        self.sleep(delay)
                    else:
                        with self.monitor.measure("pacing"):
                            self.sleep(delay)
                now = self.clock()
                self.deadline += self.period_s
            return self.original(*args, **kwargs)

        self.wrapper = paced
        sim.render = paced

    def reset(self):
        self.deadline = None

    @contextmanager
    def step(self):
        self.active = True
        try:
            yield
        finally:
            self.active = False

    def close(self):
        if self.sim.render is self.wrapper:
            if self.had_local:
                self.sim.render = self.local
            else:
                del self.sim.render
        self.active = False
        self.deadline = None


class PerformanceMonitor:
    """Exclusive region timings, with rates over a bounded rolling step window."""

    def __init__(self, capacity=300, clock=time.perf_counter):
        self.clock = clock
        self.samples = deque(maxlen=capacity)
        self._step_frame_intervals_ms = []
        self._last_render_at = None
        self._step_active = False
        self.seconds = Counter()
        self.calls = Counter()
        self._stack = []
        self._wrappers = []
        self._previous_end = None
        self.total_steps = self.total_renders = self.total_video_frames = 0

    @contextmanager
    def measure(self, name):
        entry = [self.clock(), 0.0]
        self._stack.append(entry)
        try:
            yield
        finally:
            elapsed = self.clock() - entry[0]
            self._stack.pop()
            self.seconds[name] += max(0.0, elapsed - entry[1])
            self.calls[name] += 1
            if self._stack:
                self._stack[-1][1] += elapsed

    def wrap(self, obj, method, region):
        original = getattr(obj, method)
        had_local = method in vars(obj)
        local = vars(obj).get(method)

        @wraps(original)
        def measured(*args, **kwargs):
            with self.measure(region):
                result = original(*args, **kwargs)
            if region == "render" and self._step_active:
                now = self.clock()
                if self._last_render_at is not None:
                    self._step_frame_intervals_ms.append(1000 * (now - self._last_render_at))
                self._last_render_at = now
            return result

        setattr(obj, method, measured)
        self._wrappers.append((obj, method, had_local, local, measured))

    def install(self, env):
        for obj, method, region in (
            (env.sim, "step", "physics"), (env.sim, "render", "render"),
            (env.scene, "update", "sensors"), (env.scene, "write_data_to_sim", "scene_write"),
            (env.observation_manager, "compute", "observations"),
            (env.termination_manager, "compute", "task_checks"),
            (env.recorder_manager, "record_pre_step", "recording"),
            (env.recorder_manager, "record_post_step", "recording"),
        ):
            self.wrap(obj, method, region)

    def close(self):
        for obj, method, had_local, local, wrapper in reversed(self._wrappers):
            if getattr(obj, method) is wrapper:
                if had_local:
                    setattr(obj, method, local)
                else:
                    delattr(obj, method)
        self._wrappers.clear()

    def reset_window(self):
        # Exclude startup and episode reset work from control throughput.
        self.samples.clear()
        self._previous_end = None
        self._step_frame_intervals_ms = []
        self._last_render_at = None
        self._step_active = False

    def begin_step(self):
        self._step_active = True
        self._step_frame_intervals_ms = []
        self._start = self.clock()
        self._seconds_before = self.seconds.copy()
        self._calls_before = self.calls.copy()

    def end_step(self, *, work_end, video_frames=0, admitted=True):
        if not admitted:
            self._step_active = False
            return
        now = self.clock()
        regions = dict(self.seconds - self._seconds_before)
        renders = self.calls["render"] - self._calls_before["render"]
        period = now - (self._previous_end if self._previous_end is not None else self._start)
        # Pacing occurs inside env.step for 15/30 Hz control; exclude that sleep
        # from actual host work while retaining it in the wall-rate period.
        work = max(0.0, work_end - self._start - regions.get("pacing", 0.0))
        self.samples.append({"period": period, "work": work,
                             "regions": regions, "renders": renders, "video_frames": video_frames,
                             "frame_intervals_ms": tuple(self._step_frame_intervals_ms)})
        self._previous_end = now
        self.total_steps += 1
        self.total_renders += renders
        self.total_video_frames += video_frames
        self._step_active = False

    def summary(self, step_dt):
        if not self.samples:
            return {}
        n = len(self.samples)
        elapsed = sum(sample["period"] for sample in self.samples)
        work = sorted(sample["work"] for sample in self.samples)
        periods = sorted(sample["period"] for sample in self.samples)
        regions = Counter()
        for sample in self.samples:
            regions.update(sample["regions"])
        regions["other"] = max(0.0, sum(work) - sum(value for key, value in regions.items()
                                                       if key != "pacing"))
        result = {
            "window_steps": n,
            "control_hz": n / elapsed if elapsed else 0.0,
            "render_fps": sum(s["renders"] for s in self.samples) / elapsed if elapsed else 0.0,
            "produced_video_fps": sum(s["video_frames"] for s in self.samples) / elapsed if elapsed else 0.0,
            "real_time_factor": n * step_dt / elapsed if elapsed else 0.0,
            "work_ms_mean": sum(work) * 1000 / n,
            "work_ms_p50": work[max(0, math.ceil(0.50 * n) - 1)] * 1000,
            "work_ms_p95": work[max(0, math.ceil(0.95 * n) - 1)] * 1000,
            "work_ms_p99": work[max(0, math.ceil(0.99 * n) - 1)] * 1000,
            "command_interval_ms_p50": periods[max(0, math.ceil(0.50 * n) - 1)] * 1000,
            "command_interval_ms_p95": periods[max(0, math.ceil(0.95 * n) - 1)] * 1000,
            "command_interval_ms_p99": periods[max(0, math.ceil(0.99 * n) - 1)] * 1000,
            "command_interval_ms_max": periods[-1] * 1000,
            "missed_command_budgets": sum(value > step_dt for value in periods),
            "over_budget_steps": sum(value > step_dt for value in work),
            "region_ms_mean": {key: value * 1000 / n for key, value in sorted(regions.items())},
        }
        intervals = sorted(interval for sample in self.samples for interval in sample["frame_intervals_ms"])
        if intervals:
            result["render_interval_ms_mean"] = sum(intervals) / len(intervals)
            result["render_interval_ms_p95"] = intervals[math.ceil(0.95 * len(intervals)) - 1]
            result["render_interval_ms_p99"] = intervals[math.ceil(0.99 * len(intervals)) - 1]
            result["longest_render_gap_ms"] = intervals[-1]
        return result

    def report(self, step_dt):
        s = self.summary(step_dt)
        if not s:
            return "OMY performance: no control steps completed"
        regions = ", ".join(f"{key}={value:.2f}" for key, value in s["region_ms_mean"].items())
        frame_timing = (f"; frame interval mean/p95 {s['render_interval_ms_mean']:.1f}/"
                        f"{s['render_interval_ms_p95']:.1f} ms") if "render_interval_ms_mean" in s else ""
        return (f"OMY: control {s['control_hz']:.1f}/{1 / step_dt:.0f} Hz; "
                f"render {s['render_fps']:.1f} FPS; video produced {s['produced_video_fps']:.1f} FPS; "
                f"RTF {s['real_time_factor']:.2f}; work mean/p95 {s['work_ms_mean']:.1f}/"
                f"{s['work_ms_p95']:.1f} ms; over budget {s['over_budget_steps']}/{s['window_steps']}"
                f"{frame_timing}\n"
                f"OMY host ms/control-step (exclusive, includes sync): {regions}")
