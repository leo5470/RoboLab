"""Opt-in physics-substep diagnostics; simulator imports are deferred to metadata.

State reads and file I/O are measured separately from stepping. These reads may
force lazy sensors/GPU transfers and therefore this is a diagnostic run, not a
clean throughput benchmark. JSONL is bounded in memory and flushed periodically.
"""

import json
import math
import os
import threading
import time
from functools import wraps
from pathlib import Path

import numpy as np


class DiagnosticFailure(RuntimeError):
    """A simulated state crossed the diagnostic stop threshold."""


def enable_input_diagnostics(device):
    """Wrap existing reads without polling twice or changing retry behavior."""
    if hasattr(device, "diagnostic_input"):
        return
    device.diagnostic_input = None
    reader = device._latest
    lock = threading.Lock()
    serial_stats = {"read_calls": 0, "read_errors": 0, "last_read_duration_ms": None,
                    "max_read_duration_ms": 0.0, "last_error": None}
    if device.cfg.source == "serial":
        serial_read = device._serial.read

        def read():
            started = time.perf_counter()
            error = None
            try:
                return serial_read()
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                duration = (time.perf_counter() - started) * 1000
                with lock:
                    serial_stats["read_calls"] += 1
                    serial_stats["read_errors"] += int(error is not None)
                    serial_stats["last_read_duration_ms"] = duration
                    serial_stats["max_read_duration_ms"] = max(serial_stats["max_read_duration_ms"], duration)
                    if error is not None:
                        serial_stats["last_error"] = error

        device._serial.read = read

    def latest():
        reading, age = reader()
        with lock:
            counters = dict(serial_stats) if device.cfg.source == "serial" else None
        device.diagnostic_input = {
            "raw": dict(reading) if reading is not None else None,
            "age_s": age if reading is not None else None,
            "stale": reading is None or age > device.cfg.stale_s,
            "serial": counters,
        }
        return reading, age

    device._latest = latest


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def json_value(value):
    """Produce strict JSON, retaining nonfinite values as explicit strings."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray) or hasattr(value, "detach"):
        return json_value(array(value).tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def joint_violations(names, position, velocity, limits, tolerance):
    """Check hard limits, not tracking error (large target errors can be legitimate)."""
    errors = []
    for name, q, qd, (lo, hi) in zip(names, position, velocity, limits, strict=True):
        if not math.isfinite(float(q)) or not math.isfinite(float(qd)):
            errors.append(f"{name}: nonfinite joint state q={q}, qd={qd}")
        elif q < lo - tolerance or q > hi + tolerance:
            errors.append(f"{name}: q={q:.6f} outside hard limits [{lo:.6f}, {hi:.6f}] "
                          f"with tolerance {tolerance:.3f} rad")
    return errors


def world_anchor(pose, local_position):
    """Rotate a local joint anchor using Isaac's wxyz body-link quaternion."""
    position = np.asarray(pose[:3])
    w, x, y, z = pose[3:7]
    axis = np.array([x, y, z])
    local = np.asarray(local_position)
    return position + local + 2 * np.cross(axis, np.cross(axis, local) + w * local)


def inspect_stage(stage, robot_path):
    """Read composed joint constraints and colliders without authoring USD."""
    from pxr import Usd, UsdPhysics

    joints, colliders = [], []
    root = stage.GetPrimAtPath(robot_path)
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            attributes = {attr.GetName(): attr.Get() for attr in prim.GetAttributes()
                          if attr.GetName().startswith(("physics:", "physx", "drive:"))}
            relationships = {rel.GetName(): [str(path) for path in rel.GetTargets()]
                             for rel in prim.GetRelationships()
                             if rel.GetName().startswith(("physics:", "physx"))}
            # Convert Gf vectors/quaternions only for the anchor positions used below;
            # the full attribute audit uses strings for USD compound values.
            attributes = {key: (value if isinstance(value, (str, bool, int, float, type(None)))
                                else str(value)) for key, value in attributes.items()}
            joints.append({
                "path": str(prim.GetPath()), "type": prim.GetTypeName(),
                "schemas": list(prim.GetAppliedSchemas()),
                "attributes": attributes, "relationships": relationships,
                "body0": relationships.get("physics:body0", []),
                "body1": relationships.get("physics:body1", []),
                "local_pos0": list(joint.GetLocalPos0Attr().Get()),
                "local_pos1": list(joint.GetLocalPos1Attr().Get()),
            })
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(),
                              "attributes": {
                                  attr.GetName(): str(attr.Get()) for attr in prim.GetAttributes()
                                  if attr.GetName().startswith(("physics:", "physxCollision:",
                                                               "physxConvexDecomposition:"))
                              }})
    return {"joints": joints, "colliders": colliders}


class RuntimeDiagnostics:
    def __init__(self, env, output, timing, *, joint_tolerance=0.2, stage_audit=None):
        self.env, self.timing = env, timing
        self.robot = env.scene["robot"]
        self.names = list(self.robot.joint_names)
        self.body_names = list(self.robot.body_names)
        self.limits = array(self.robot.data.joint_pos_limits[0]).copy()
        self.arm_ids = [self.names.index(f"panda_joint{i}") for i in range(1, 8)]
        self.tolerance = joint_tolerance
        self.output = Path(output)
        self.stream = (self.output / "omy_diagnostics.jsonl").open("x", buffering=65536)
        self.pending = []
        self.control_step = None
        self.device = None
        self.action = None
        self.episode = 0
        self.count = 0
        self.fault = None
        self.original_update = None
        self.wrapper = None
        self.started = time.monotonic()
        self.last_flush = self.started
        self.control_before = None
        self.max_anchor_gap = 0.0
        self.max_arm_error = 0.0
        self.max_object_speed = 0.0
        self.max_substep_ms = 0.0
        self.previous_regions = {}
        self.audit = stage_audit or {"joints": [], "colliders": []}
        self.link_paths = []

    def install(self):
        """Install after the timing wrappers; sample only during commanded steps."""
        self.link_paths = list(self.robot.root_physx_view.link_paths[0])
        self.audit = inspect_stage(self.env.sim.get_initial_stage(), self.robot.root_physx_view.prim_paths[0])
        metadata = {
            "schema_version": 1, "joint_names": self.names, "body_names": self.body_names,
            "body_paths": self.link_paths, "joint_pos_limits_rad": self.limits,
            "joint_vel_limits_rad_s": self.robot.data.joint_vel_limits[0],
            "joint_stiffness": self.robot.data.default_joint_stiffness[0],
            "joint_damping": self.robot.data.default_joint_damping[0],
            "joint_effort_limits": self.robot.data.joint_effort_limits[0],
            "physics_dt_s": self.env.physics_dt, "control_dt_s": self.env.step_dt,
            "joint_limit_tolerance_rad": self.tolerance,
            "stop_conditions": ["nonfinite joint/body/object state", "hard joint limit excursion"],
            "scene_audit": self.audit,
            "object_collision_audits": {
                name: inspect_stage(self.env.sim.get_initial_stage(), asset.root_physx_view.prim_paths[0])
                for name, asset in self.env.scene.rigid_objects.items()
                if hasattr(asset, "root_physx_view")
            },
            "host_cpu": {"affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                         **{name: (path.read_text().strip() if path.is_file() else None) for name, path in (
                             ("driver", Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver")),
                             ("governor", Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")),
                             ("energy_preference", Path("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference")),
                         )}},
            "notes": ["Forces are normal contact forces only; no friction force is inferred.",
                      "Substep region times are host wall time including synchronization.",
                      "State/sensor reads and JSON serialization add measured diagnostic overhead.",
                      "Anchor gaps are available only when both bodies have tracked physics links.",
                      "Source age is the age of the exact sample consumed by upstream advance()."],
        }
        (self.output / "omy_diagnostic_metadata.json").write_text(
            json.dumps(json_value(metadata), indent=2, allow_nan=False) + "\n")
        self.original_update = self.env.scene.update

        @wraps(self.original_update)
        def update(*args, **kwargs):
            result = self.original_update(*args, **kwargs)
            if self.control_step is not None:
                self.capture("physics_state")
            return result

        self.wrapper = update
        self.env.scene.update = update

    def start_episode(self, episode, device):
        self.episode, self.device = episode, device
        device.enable_input_diagnostics()
        self.control_step = None
        self.action = None
        self.capture("reset_state")
        self.flush()

    def begin_control(self, action, step):
        self.control_step = step
        self.action = array(action).copy()
        self.control_before = dict(self.timing.seconds)
        self.previous_regions = dict(self.timing.seconds)

    def anchor_gaps(self, poses):
        by_path = dict(zip(self.link_paths, poses, strict=True))
        gaps = {}
        for joint in self.audit["joints"]:
            left, right = joint["body0"], joint["body1"]
            if len(left) != 1 or len(right) != 1 or left[0] not in by_path or right[0] not in by_path:
                continue
            gap = float(np.linalg.norm(world_anchor(by_path[left[0]], joint["local_pos0"]) -
                                       world_anchor(by_path[right[0]], joint["local_pos1"])))
            gaps[joint["path"]] = gap
        return gaps

    def capture(self, kind):
        capture_started = time.perf_counter()
        with self.timing.measure("diagnostic_state"):
            data = self.robot.data
            q, qd = array(data.joint_pos[0]).copy(), array(data.joint_vel[0]).copy()
            poses = array(data.body_link_pose_w[0]).copy()
            velocities = array(data.body_link_state_w[0, :, 7:]).copy()
            violations = joint_violations(self.names, q, qd, self.limits, self.tolerance)
            if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
                violations.append("robot: nonfinite body pose or velocity")
            gaps = self.anchor_gaps(poses)
            objects = {}
            for name, asset in self.env.scene.rigid_objects.items():
                state = array(asset.data.root_state_w[0]).copy()
                objects[name] = {"root_state_w": state, "linear_speed_m_s": float(np.linalg.norm(state[7:10]))}
                if not np.isfinite(state).all():
                    violations.append(f"{name}: nonfinite object state")
            contacts = {}
            for name, sensor in self.env.scene.sensors.items():
                if not hasattr(sensor.cfg, "filter_prim_paths_expr"):
                    continue
                contacts[name] = {"net_normal_forces_w_n": sensor.data.net_forces_w[0],
                                  "filtered_normal_forces_w_n": (sensor.data.force_matrix_w[0]
                                      if sensor.data.force_matrix_w is not None else None)}
            arm_error = (float(np.max(np.abs(q[self.arm_ids] - self.action[:7])))
                         if self.action is not None else None)
            record = {
                "kind": kind, "episode": self.episode, "control_step": self.control_step,
                "physics_step": self.env._sim_step_counter,
                "sim_time_s": self.env._sim_step_counter * self.env.physics_dt,
                "wall_time_s": time.time(), "elapsed_s": time.monotonic() - self.started,
                "action": self.action, "leader": getattr(self.device, "diagnostic_input", None),
                "joint_pos_rad": q, "joint_vel_rad_s": qd,
                "body_link_pose_w": poses, "body_link_velocity_w": velocities,
                "max_arm_tracking_error_rad": arm_error,
                "anchor_separation_m": gaps, "objects": objects, "contacts": contacts,
                "violations": violations,
            }
            # Detach all live tensor references before the next physics update.
            record = json_value(record)
        record["capture_ms"] = (time.perf_counter() - capture_started) * 1000
        current = dict(self.timing.seconds)
        record["region_ms_since_previous_substep"] = {
            key: (value - self.previous_regions.get(key, 0.0)) * 1000 for key, value in current.items()
        } if kind == "physics_state" else {}
        self.previous_regions = current
        self.max_substep_ms = max(self.max_substep_ms, record["region_ms_since_previous_substep"].get("physics", 0.0))
        self.max_arm_error = max(self.max_arm_error, arm_error or 0.0)
        self.max_anchor_gap = max(self.max_anchor_gap, max(gaps.values(), default=0.0))
        self.max_object_speed = max(self.max_object_speed, max((obj["linear_speed_m_s"] for obj in objects.values()), default=0.0))
        self.add(record)
        if violations:
            self.fault = {"episode": self.episode, "control_step": self.control_step,
                          "physics_step": self.env._sim_step_counter, "violations": violations}
            self.flush()
            raise DiagnosticFailure("; ".join(violations))

    def end_control(self, *, incomplete=False):
        if self.control_step is None:
            return
        self.add({"kind": "control_timing", "control_step": self.control_step, "episode": self.episode,
                  "physics_step": self.env._sim_step_counter, "incomplete": incomplete,
                  "region_ms": {key: (value - self.control_before.get(key, 0.0)) * 1000
                                for key, value in self.timing.seconds.items()},
                  "host_work_before_timing_write_ms": (self.timing.clock() - self.timing._start) * 1000})
        self.control_step = None

    def add(self, record):
        with self.timing.measure("diagnostic_io"):
            self.pending.append(json.dumps(json_value(record), separators=(",", ":"), allow_nan=False) + "\n")
            self.count += 1
            if len(self.pending) >= 32 or time.monotonic() - self.last_flush >= 1.0:
                self.flush()

    def flush(self):
        if self.pending:
            self.stream.writelines(self.pending)
            self.pending.clear()
        self.stream.flush()
        self.last_flush = time.monotonic()

    def close(self):
        if self.wrapper is not None and self.env.scene.update is self.wrapper:
            self.env.scene.update = self.original_update
        try:
            self.flush()
        finally:
            self.stream.close()
        summary = {"schema_version": 1, "records": self.count, "first_failure": self.fault,
                   "max_arm_tracking_error_rad": self.max_arm_error,
                   "max_tracked_anchor_separation_m": self.max_anchor_gap,
                   "max_object_linear_speed_m_s": self.max_object_speed,
                   "max_physics_substep_host_ms": self.max_substep_ms}
        (self.output / "omy_diagnostic_summary.json").write_text(
            json.dumps(json_value(summary), indent=2, allow_nan=False) + "\n")
