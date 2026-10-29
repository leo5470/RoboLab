"""Simulator-independent settings and stability checks for new collections."""

from dataclasses import asdict, dataclass
from collections import Counter
import math

import numpy as np
from robolab.core.scenes.placement import supported_root_height


@dataclass(frozen=True)
class PreparationSettings:
    min_s: float = 0.5
    stable_s: float = 0.3
    timeout_s: float = 5.0
    linear_m_s: float = 0.01
    angular_rad_s: float = 0.5
    position_m: float = 0.001
    rotation_deg: float = 0.5
    clearance_m: float = 0.002
    support_force_n: float = 0.001
    pair_clearance_m: float = 0.02
    edge_clearance_m: float = 0.01
    max_tilt_change_deg: float = 15.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Scene preparation settings must be finite and positive")
        if self.timeout_s < max(self.min_s, self.stable_s):
            raise ValueError("Scene settling timeout must cover minimum and stable durations")
        if self.max_tilt_change_deg > 180:
            raise ValueError("Tilt change must be at most 180 degrees")

    @classmethod
    def from_args(cls, args):
        return cls(**{field: getattr(args, "scene_settle_" + field, default)
                      for field, default in asdict(cls()).items()})

    def metadata(self):
        return {"schema_version": 2, "placement": "transformed_bounds",
                "flat_layout_validation": "footprint_clearance_and_tilt", **asdict(self)}


def add_arguments(parser):
    defaults = PreparationSettings()
    for field, description in (
        ("min_s", "Minimum simulated settling duration"),
        ("stable_s", "Continuous supported and stationary interval"),
        ("timeout_s", "Maximum simulated settling duration before requiring retry/skip"),
        ("linear_m_s", "Maximum initial linear speed in m/s"),
        ("angular_rad_s", "Maximum initial angular speed in rad/s"),
    ):
        parser.add_argument("--scene-settle-" + field.replace("_", "-"), type=float,
                            default=getattr(defaults, field), help=description)


class StabilityGate:
    def __init__(self, settings):
        self.settings = settings
        self.since = None
        self.anchors = {}
        self.last_time = -1.0
        self.reason_counts = Counter()

    def observe(self, elapsed_s, bodies):
        if elapsed_s <= self.last_time or not math.isfinite(elapsed_s):
            raise ValueError("Settling observations must advance simulated time")
        self.last_time = elapsed_s
        reasons, diagnostics = [], {}
        for name, body in bodies.items():
            pose = np.asarray(body["pose"], dtype=float)
            velocity = np.asarray(body["velocity"], dtype=float)
            if (pose.shape != (7,) or velocity.shape != (6,) or not np.isfinite(pose).all()
                    or not np.isfinite(velocity).all() or np.linalg.norm(pose[3:]) < 1e-8):
                reasons.append(f"{name}: invalid pose/velocity")
                continue
            linear, angular = float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:]))
            supported = bool(body["supported"])
            diagnostics[name] = {"linear_m_s": linear, "angular_rad_s": angular,
                                 "supported": supported, "pose": pose.tolist()}
            if not supported:
                reasons.append(f"{name}: unsupported")
            if linear > self.settings.linear_m_s or angular > self.settings.angular_rad_s:
                reasons.append(f"{name}: moving ({linear:.4g} m/s, {angular:.4g} rad/s)")
            if name in self.anchors:
                anchor = self.anchors[name]
                dot = abs(float(np.dot(anchor[3:], pose[3:]) /
                                (np.linalg.norm(anchor[3:]) * np.linalg.norm(pose[3:]))))
                angle = math.degrees(2 * math.acos(min(1.0, dot)))
                if (np.linalg.norm(pose[:3] - anchor[:3]) > self.settings.position_m
                        or angle > self.settings.rotation_deg):
                    reasons.append(f"{name}: pose drift")
        if reasons:
            self.reason_counts.update(reason.split(" (")[0] for reason in reasons)
            self.since = None
            self.anchors = {}
        elif self.since is None:
            self.since = elapsed_s
            self.anchors = {name: np.asarray(body["pose"], dtype=float).copy() for name, body in bodies.items()}
        stable_for = elapsed_s - self.since if self.since is not None else 0.0
        ready = elapsed_s >= self.settings.min_s and stable_for + 1e-9 >= self.settings.stable_s
        status = "ready" if ready else "failed" if elapsed_s + 1e-9 >= self.settings.timeout_s else "settling"
        if status == "failed" and not reasons:
            reasons = [f"Stable interval too short ({stable_for:.3f}/{self.settings.stable_s:.3f} s)"]
        return {"status": status, "elapsed_s": elapsed_s, "stable_s": stable_for,
                "reasons": reasons, "bodies": diagnostics, "reason_counts": dict(self.reason_counts)}
