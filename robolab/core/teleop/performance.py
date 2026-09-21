"""Small, bounded timing and pacing helpers for interactive collection."""

import os
import statistics
import time
from collections import deque


def worker_settings(profile, device, kit_threads=None, physics_threads=None):
    """Session-only worker tuning for a single CPU teleoperation environment.

    Zero leaves the corresponding Kit setting untouched. Explicit Kit command
    line overrides must follow these defaults. Keep GPU/standard profiles stock.
    """
    for count in (kit_threads, physics_threads):
        if count is not None and count < 0:
            raise ValueError("worker thread counts must be non-negative")
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    tuned = profile == "wifi" and device == "cpu"
    if kit_threads is None:
        kit_threads = min(16, available) if tuned else 0
    if physics_threads is None:
        physics_threads = min(2, available) if tuned else 0
    settings = {}
    if kit_threads:
        settings["/plugins/carb.tasking.plugin/threadCount"] = kit_threads
        settings["/plugins/omni.tbb.globalcontrol/maxThreadCount"] = kit_threads
    if physics_threads:
        settings["/persistent/physics/numThreads"] = physics_threads
    if settings:
        # PhysX exposes its worker count as a persistent preference. Do not
        # silently change the user's other Isaac Sim applications or next run.
        settings["/app/settings/persistent"] = "false"
    return settings


class RealtimePacer:
    """Cap simulated time at real time without bursts of catch-up actions."""

    def __init__(self, period, clock=time.perf_counter, sleep=time.sleep):
        self.period = period
        self.clock = clock
        self.sleep = sleep
        self.deadline = clock()

    def wait(self, pump=None):
        while self.clock() < self.deadline:
            if pump is not None:
                pump()
            remaining = self.deadline - self.clock()
            if remaining > 0:
                self.sleep(min(remaining, 0.005))
        # Never replay old deadlines after a slow render/disk/network stall.
        self.deadline = self.clock() + self.period


class LoopTiming:
    def __init__(self, capacity=300):
        self.steps = deque(maxlen=capacity)
        self._regions = deque(maxlen=capacity)
        self._previous = (0.0, 0.0, 0)
        self.render_seconds = 0.0
        self.renders = 0
        self.physics_seconds = 0.0

    def install(self, sim):
        self._sim = sim
        self._original_render = sim.render
        self._original_step = sim.step

        def render(*args, **kwargs):
            start = time.perf_counter()
            try:
                return self._original_render(*args, **kwargs)
            finally:
                self.render_seconds += time.perf_counter() - start
                self.renders += 1

        sim.render = render

        def step(*args, **kwargs):
            start = time.perf_counter()
            rendered_before = self.render_seconds
            try:
                return self._original_step(*args, **kwargs)
            finally:
                # Exclude nested rendering if a caller uses step(render=True).
                self.physics_seconds += max(
                    0.0, time.perf_counter() - start - (self.render_seconds - rendered_before)
                )

        sim.step = step

    def close(self):
        if hasattr(self, "_original_render"):
            self._sim.render = self._original_render
            self._sim.step = self._original_step

    def clear(self):
        self.steps.clear()
        self._regions.clear()
        self._previous = (0.0, 0.0, 0)
        self.render_seconds = 0.0
        self.renders = 0
        self.physics_seconds = 0.0

    def record_step(self, seconds):
        """Keep elapsed and region samples aligned when the window rolls over."""
        self.steps.append(seconds)
        totals = (self.physics_seconds, self.render_seconds, self.renders)
        self._regions.append(tuple(value - previous for value, previous in zip(totals, self._previous)))
        self._previous = totals

    def summary(self, period=None):
        if not self.steps:
            return {}
        ordered = sorted(self.steps)
        physics = sum(region[0] for region in self._regions)
        render = sum(region[1] for region in self._regions)
        renders = sum(region[2] for region in self._regions)
        return {
            "step_ms_p50": statistics.median(ordered) * 1000,
            "step_ms_p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] * 1000,
            "step_ms_max": ordered[-1] * 1000,
            "over_budget_steps": sum(value > period for value in ordered) if period else 0,
            "physics_ms_mean": physics * 1000 / len(ordered),
            "other_ms_mean": max(0.0, sum(ordered) - physics - render)
            * 1000 / len(ordered),
            "render_ms_mean": render * 1000 / len(ordered),
            "render_call_ms_mean": render * 1000 / max(1, renders),
        }


class StreamQuality:
    """Read actual WebRTC QoS rather than estimating LAN latency from step rate."""

    def __init__(self):
        self.latest = None
        self.interface = None
        self.handle = None
        try:
            from omni.kit.livestream.bind import acquire_livestream_interface

            self.interface = acquire_livestream_interface()
            self.handle = self.interface.register_qos_status_callback(self._update)
        except (ImportError, AttributeError, RuntimeError) as exc:
            print(f"WebRTC QoS unavailable on this Kit version: {exc}")

    def _update(self, status):
        # Copy primitive values; the native status object is callback-scoped.
        self.latest = (time.perf_counter(), {
            "rtt_ms": status.average_rtd_ms,
            "bitrate_mbps": status.qos_bitrate / 1e6,
            "width": status.encode_width,
            "height": status.encode_height,
        })

    def summary(self):
        latest = self.latest
        if latest is None or time.perf_counter() - latest[0] > 10:
            return "WebRTC: awaiting client QoS"
        q = latest[1]
        rtt = f"{q['rtt_ms']} ms" if q['rtt_ms'] > 0 else "unavailable"
        # These dimensions are recommendations for a future frame, not a
        # measurement of the current encoded image. Zero means no size supplied.
        size = f", requested size {q['width']}x{q['height']}" if q['width'] > 0 and q['height'] > 0 else ""
        return f"WebRTC RTT {rtt}, bitrate target {q['bitrate_mbps']:.1f} Mbps{size}"

    def close(self):
        if self.interface is not None and self.handle is not None:
            self.interface.deregister_qos_status_callback(self.handle)
            self.handle = None
