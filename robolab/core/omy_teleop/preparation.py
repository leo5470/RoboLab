"""Simulator-independent settings and stability checks for new collections."""

from dataclasses import asdict, dataclass
from collections import Counter
import math
from pathlib import Path

import numpy as np
from robolab.core.scenes.placement import supported_root_height

# Scene directories whose layouts place every target and the bin directly on
# the table, mapped to the re-seat clearance they require (None: any). Simple-PnP
# v2 is qualified from contact starts, so preparing it with another clearance
# would record episode starts that differ from the qualified ones.
FLAT_LAYOUT_SCENE_DIRS = {"simple_pnp": None, "simple_pnp_v2": 0.0}
# Libraries whose preparation measures exact local bounds from geometry points.
# BBoxCache's untransformed bound inflates rotated prims in composed scenes
# (up to ~2x; see robolab.core.scenes.placement.prim_local_bounds). v1 keeps the
# legacy, conservative bound so its collection behavior is unchanged.
EXACT_BOUNDS_SCENE_DIRS = {"simple_pnp_v2"}
# Libraries whose scene-preparation stability gate uses speeds derived from
# successive poses. PhysX can keep reporting 0.01-0.03 m/s / 0.2-0.7 rad/s for
# resting bodies whose poses change < 0.1 mm (Simple-PnP v2 pilot, 2026-11-02).
# v1 preparation keeps reported velocities so its start gate is unchanged. This
# gate is separate from the placement success check: both the v1 and v2 tasks
# pass speed_source="pose" to object_resting_in_container, so v1 starts are
# gated on reported speeds while v1 success is judged on pose-derived speeds.
POSE_SPEED_SCENE_DIRS = {"simple_pnp_v2"}


def rotation_angle(q1, q2):
    """Angle in radians between two orientations; stable near zero (no acos)."""
    a = np.asarray(q1, dtype=float); b = np.asarray(q2, dtype=float)
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    w = abs(float(np.dot(a, b)))
    # |vector part| of a^-1 b without forming the product explicitly.
    v = float(np.linalg.norm(a[0] * b[1:] - b[0] * a[1:] - np.cross(a[1:], b[1:])))
    return 2.0 * math.atan2(v, w)


def flat_layout_requirement(scene_path, body_names, clearance_m):
    """Return (run_flat_table_checks, setup_error) for a scene.

    v1 (``simple_pnp``) behavior is unchanged: a scene without a rigid ``table``
    skips the flat-table checks. Other flat-layout libraries fail closed.
    """
    folder = Path(str(scene_path)).parent.name
    if folder not in FLAT_LAYOUT_SCENE_DIRS:
        return False, None
    if "table" not in body_names:
        return False, None if folder == "simple_pnp" else f"{folder} scene has no rigid 'table': {scene_path}"
    required = FLAT_LAYOUT_SCENE_DIRS[folder]
    if required is not None and clearance_m != required:
        return True, (f"{folder} scenes require --scene-settle-clearance-m {required:g} "
                      f"(got {clearance_m:g})")
    return True, None


@dataclass(frozen=True)
class PreparationSettings:
    skip_checks: bool = False
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
        values = asdict(self)
        if type(values.pop("skip_checks")) is not bool:
            raise ValueError("skip_checks must be a boolean")
        clearance = values.pop("clearance_m")
        if any(not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("Scene preparation settings must be finite and positive")
        # 0 re-seats objects in contact with the tabletop (Simple-PnP v2).
        if not math.isfinite(clearance) or clearance < 0:
            raise ValueError("Scene preparation clearance must be finite and non-negative")
        if self.timeout_s < max(self.min_s, self.stable_s):
            raise ValueError("Scene settling timeout must cover minimum and stable durations")
        if self.max_tilt_change_deg > 180:
            raise ValueError("Tilt change must be at most 180 degrees")

    @classmethod
    def from_args(cls, args):
        return cls(**{field: getattr(args, "skip_settle_checks" if field == "skip_checks"
                                    else "scene_settle_" + field, default)
                      for field, default in asdict(cls()).items()})

    def metadata(self):
        values = asdict(self)
        # Preserve the exact metadata of existing sessions when checks are enabled.
        if not self.skip_checks:
            values.pop("skip_checks")
        return {"schema_version": 2, "placement": "transformed_bounds",
                "flat_layout_validation": "footprint_clearance_and_tilt", **values}


def add_arguments(parser):
    parser.add_argument("--skip-settle-checks", action="store_true",
                        help="Skip settle validation after the minimum settling time; retain placement preflight.")
    defaults = PreparationSettings()
    for field, description in (
        ("min_s", "Minimum simulated settling duration"),
        ("stable_s", "Continuous supported and stationary interval"),
        ("timeout_s", "Maximum simulated settling duration before requiring retry/skip"),
        ("linear_m_s", "Maximum initial linear speed in m/s"),
        ("angular_rad_s", "Maximum initial angular speed in rad/s"),
        ("clearance_m", "Gap in m between each object's transformed bottom and the tabletop "
                        "when preparation re-seats it (Simple-PnP v2 requires 0)"),
    ):
        parser.add_argument("--scene-settle-" + field.replace("_", "-"), type=float,
                            default=getattr(defaults, field), help=description)


class StabilityGate:
    """Supported-and-stationary gate.

    ``speed_source="reported"`` uses simulator velocities; ``"pose"`` uses
    speeds between successive observed poses (the first observation has no
    estimate and is never counted as stationary).
    """

    def __init__(self, settings, speed_source="reported"):
        if speed_source not in ("reported", "pose"):
            raise ValueError("speed_source must be 'reported' or 'pose'")
        self.settings = settings
        self.speed_source = speed_source
        self.previous = {}
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
            reported = (float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:])))
            if self.speed_source == "pose":
                prior = self.previous.get(name)
                if prior is None:
                    linear = angular = math.inf
                else:
                    dt = elapsed_s - prior[0]
                    linear = float(np.linalg.norm(pose[:3] - prior[1][:3])) / dt
                    angular = rotation_angle(prior[1][3:], pose[3:]) / dt
                self.previous[name] = (elapsed_s, pose.copy())
            else:
                linear, angular = reported
            supported = bool(body["supported"])
            diagnostics[name] = {"linear_m_s": linear, "angular_rad_s": angular,
                                 "reported_linear_m_s": reported[0], "reported_angular_rad_s": reported[1],
                                 "speed_source": self.speed_source, "supported": supported, "pose": pose.tolist()}
            if not supported:
                reasons.append(f"{name}: unsupported")
            if not math.isfinite(linear):
                reasons.append(f"{name}: no pose history yet")
            elif linear > self.settings.linear_m_s or angular > self.settings.angular_rad_s:
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
