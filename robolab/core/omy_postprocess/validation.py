"""Read-only validation of OMY state recordings; imports no simulator modules."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .timing import validate_native15_timeline


ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
ROBOT_JOINTS = (*ARM_JOINTS, "finger_joint", "right_outer_knuckle_joint", "left_inner_finger_joint",
                "right_inner_finger_joint", "left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint")
JOINT_ALIGNMENT_ATOL_RAD = 2e-6
CADENCE_ERROR_CODES = {"non_native_configuration", "non_native_interval", "non_native_spacing", "non_native_grid"}


def _plain(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _blocks(dataset):
    if not dataset.shape:
        yield dataset[()]
        return
    if dataset.shape[0] == 0:
        return
    # Keep full-state scans bounded even for older long episodes and images.
    row_bytes = max(1, int(np.prod(dataset.shape[1:], dtype=np.int64)) * dataset.dtype.itemsize)
    block_rows = max(1, (1024 * 1024) // row_bytes)
    for first in range(0, dataset.shape[0], block_rows):
        yield dataset[first:first + block_rows]


def _fingerprint(datasets):
    digest = hashlib.sha256()
    for name, dataset in sorted(datasets.items()):
        # Content identity is trajectory identity, not capture-wall-clock or
        # subsequently materialized camera identity. Repackaging/relocation
        # and additional policy images must not manufacture another attempt.
        if not (name == "actions" or name.startswith(("initial_state/articulation/", "initial_state/rigid_object/",
                                                       "states/", "obs/proprio_obs/"))
                or name in ("timing/sim_before_s", "timing/sim_after_s")):
            continue
        if dataset.dtype.kind not in "biufc":
            continue
        digest.update(json.dumps([name, dataset.dtype.str, list(dataset.shape)], separators=(",", ":")).encode())
        for block in _blocks(dataset):
            values = np.asarray(block)
            if values.dtype.kind in "OUS":
                digest.update(json.dumps(_plain(values), ensure_ascii=False, sort_keys=True).encode())
            else:
                digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def validate_episode(group, metadata):
    """Inspect an HDF5 episode without altering it or trimming quality defects.

    ``metadata`` is the decoded ``robolab_omy_teleop`` attribute. Structural
    validity and native-15 acceptance are distinct: intact 30/60 Hz episodes
    remain archivable but never gain a 15 FPS label through this validator.
    """
    errors, warnings = [], []

    def issue(target, code, message, **details):
        target.append({"code": code, "message": message, **details})

    def error(code, message, **details):
        issue(errors, code, message, **details)

    def warn(code, message, **details):
        issue(warnings, code, message, **details)

    datasets = {}
    group.visititems(lambda name, item: datasets.update({name: item}) if isinstance(item, h5py.Dataset) else None)
    actions = datasets.get("actions")
    count = int(actions.shape[0]) if actions is not None and actions.ndim else 0
    report = {
        "group": group.name.lstrip("/"),
        "name": group.name.rsplit("/", 1)[-1],
        "sample_count": count,
        "outcome": "unknown",
        "outcome_evidence": {},
        "structural_valid": False,
        "native_15_hz": False,
        "training_eligible": False,
        "errors": errors,
        "warnings": warnings,
        "fingerprint": None,
        "joint_names": [],
        "timing": {},
        "alignment_tolerance_rad": JOINT_ALIGNMENT_ATOL_RAD,
    }
    if "success" in group.attrs:
        success = _plain(group.attrs["success"])
        if isinstance(success, (bool, int)) and success in (0, 1):
            report["outcome"] = "success" if bool(success) else "failure"
            report["outcome_evidence"] = {"attribute": "success", "value": success}
        else:
            warn("unknown_outcome", "Episode success attribute is not a valid boolean")
    else:
        warn("unknown_outcome", "No finalized episode success attribute; outcome remains unknown")
    if count == 0:
        error("empty_episode", "Episode contains no actions")
    if actions is None or actions.ndim != 2 or actions.shape[1] != 8:
        error("action_shape", "actions must have shape (N, 8)")
    elif actions.dtype.kind != "f":
        error("action_dtype", "Recorded commands must contain real floating-point values")
    if "num_samples" in group.attrs:
        try:
            declared_count = int(group.attrs["num_samples"])
        except (ValueError, TypeError):
            declared_count = -1
        if declared_count != count:
            error("declared_sample_count", "num_samples does not equal the action count")

    for name, dataset in datasets.items():
        if name.startswith(("states/", "obs/", "timing/")) and (not dataset.shape or dataset.shape[0] != count):
            error("sample_count", "Per-step dataset count differs from actions", field=name)
        if name.startswith("initial_state/") and (not dataset.shape or dataset.shape[0] != 1):
            error("initial_state_count", "Initial-state dataset must contain exactly one state", field=name)
        if dataset.dtype.kind in "fc":
            for values in _blocks(dataset):
                if not np.all(np.isfinite(values)):
                    error("nonfinite_values", "Recorded numerical values must be finite", field=name)
                    break

    performance = metadata.get("performance", {})
    if not isinstance(performance, dict):
        performance = {}
        error("metadata_type", "performance metadata must be an object")
    joint_names = performance.get("joint_order", metadata.get("joint_order", []))
    if (not isinstance(joint_names, (list, tuple)) or len(joint_names) != 13
            or any(not isinstance(name, str) for name in joint_names)
            or set(joint_names) != set(ROBOT_JOINTS)):
        error("joint_names", "Recorded joint_order must name all thirteen known DROID linkage joints exactly once")
        joint_names = []
    report["joint_names"] = list(joint_names)

    required = {
        "initial_state/articulation/robot/joint_position": (1, 13),
        "initial_state/articulation/robot/joint_velocity": (1, 13),
        "initial_state/articulation/robot/root_pose": (1, 7),
        "initial_state/articulation/robot/root_velocity": (1, 6),
        "states/articulation/robot/joint_position": (count, 13),
        "states/articulation/robot/joint_velocity": (count, 13),
        "states/articulation/robot/root_pose": (count, 7),
        "states/articulation/robot/root_velocity": (count, 6),
        "obs/proprio_obs/arm_joint_pos": (count, 7),
        "obs/proprio_obs/gripper_pos": (count, 1),
    }
    for name, shape in required.items():
        if name not in datasets:
            error("missing_required_dataset", "Required recorded channel is missing", field=name)
        elif datasets[name].shape != shape:
            error("dataset_shape", f"Recorded channel must have shape {shape}", field=name)
        elif datasets[name].dtype.kind != "f":
            error("dataset_dtype", "Recorded state/proprioception must contain real floating-point values", field=name)

    for name, dataset in datasets.items():
        if name.startswith("states/"):
            initial_name = "initial_state/" + name[len("states/"):]
            initial = datasets.get(initial_name)
            if initial is None:
                error("missing_initial_state", "Post-step state has no matching initial-state channel", field=name)
            elif initial.shape[1:] != dataset.shape[1:]:
                error("initial_state_shape", "Initial/post-step state channel shapes differ", field=name)
        elif name.startswith("initial_state/") and not name.startswith("initial_state/cameras/"):
            post_name = "states/" + name[len("initial_state/"):]
            if post_name not in datasets:
                error("missing_post_state", "Initial state has no matching per-action post-state channel", field=name)
    # A rigid object's pose and velocity are both needed for faithful restore.
    rigid_objects = {name.split("/")[2] for name in datasets if name.startswith("states/rigid_object/")}
    for object_name in sorted(rigid_objects):
        for leaf, width in (("root_pose", 7), ("root_velocity", 6)):
            for root, rows in (("states", count), ("initial_state", 1)):
                name = f"{root}/rigid_object/{object_name}/{leaf}"
                if name not in datasets or datasets[name].shape != (rows, width):
                    error("rigid_object_shape", f"Rigid-object restore channel must have shape {(rows, width)}", field=name)
                elif datasets[name].dtype.kind != "f":
                    error("dataset_dtype", "Rigid-object restore channels must contain real floating-point values", field=name)

    initial = datasets.get("initial_state/articulation/robot/joint_position")
    states = datasets.get("states/articulation/robot/joint_position")
    measured = datasets.get("obs/proprio_obs/arm_joint_pos")
    if (count and joint_names and initial is not None and states is not None and measured is not None
            and initial.shape == (1, 13) and states.shape == (count, 13) and measured.shape == (count, 7)
            and initial.dtype.kind == states.dtype.kind == measured.dtype.kind == "f"):
        indices = [joint_names.index(name) for name in ARM_JOINTS]
        expected = np.concatenate((initial[:], states[:-1]), axis=0)[:, indices]
        difference = np.abs(measured[:] - expected)
        bad = np.flatnonzero(np.any(difference > JOINT_ALIGNMENT_ATOL_RAD, axis=1))
        if bad.size:
            error("preaction_alignment", "Measured arm observations differ from initial/shifted recorded state", indices=bad.tolist())
        report["max_preaction_alignment_error_rad"] = float(np.max(difference)) if np.all(np.isfinite(difference)) else None

    for field, expected in (("action_type", "absolute_joint_position"), ("joint_units", "rad"),
                            ("gripper_encoding", "+1=open,-1=closed")):
        if field not in metadata:
            warn("missing_provenance", f"Missing declared {field}", field=field)
        elif metadata[field] != expected:
            error("unsupported_units", f"Declared {field} must be {expected!r}", field=field)
    action_names = metadata.get("action_names")
    if action_names is None:
        warn("missing_provenance", "Missing named action order", field="action_names")
    elif action_names != [*ARM_JOINTS, "gripper"]:
        error("action_order", "Actions must name seven Franka joints in order, followed by gripper")
    if not metadata.get("task"):
        warn("missing_provenance", "Missing task identity", field="task")
    if not metadata.get("calibration"):
        warn("missing_provenance", "Missing calibration provenance", field="calibration")
    if performance.get("stepping_fallback_reasons"):
        warn("stepping_fallback", "Recorded stepping fell back from its requested profile",
             reasons=performance["stepping_fallback_reasons"])

    before = datasets.get("timing/sim_before_s")
    after = datasets.get("timing/sim_after_s")
    timeline = validate_native15_timeline(
        before[()] if before is not None else None,
        after[()] if after is not None else None,
        declared_control_hz=metadata.get("control_hz", performance.get("control_hz")),
        physics_dt=performance.get("physics_dt", metadata.get("physics_dt")),
        decimation=performance.get("decimation", metadata.get("decimation")),
    )
    if "control_hz" in metadata and "control_hz" in performance:
        try:
            consistent = abs(float(metadata["control_hz"]) - float(performance["control_hz"])) <= 1e-9
        except (ValueError, TypeError):
            consistent = False
        if not consistent:
            timeline["errors"].append({"code": "inconsistent_control_metadata", "message": "Top-level and effective control_hz metadata disagree"})
            timeline["valid"] = timeline["native_15_hz"] = False
            timeline["structural_valid"] = False
            timeline["timestamps_s"] = []
    for field in ("physics_dt", "decimation"):
        if field in metadata and field in performance:
            try:
                consistent = abs(float(metadata[field]) - float(performance[field])) <= 1e-12
            except (ValueError, TypeError):
                consistent = False
            if not consistent:
                timeline["errors"].append({"code": "inconsistent_physics_metadata",
                                           "message": f"Top-level and effective {field} metadata disagree"})
                timeline["valid"] = timeline["native_15_hz"] = timeline["structural_valid"] = False
                timeline["timestamps_s"] = []
    report["timing"] = timeline
    errors.extend(timeline["errors"])
    report["native_15_hz"] = timeline["native_15_hz"]
    report["structural_valid"] = not any(item["code"] not in CADENCE_ERROR_CODES for item in errors)
    valid = datasets.get("timing/source_valid")
    if valid is None:
        warn("missing_source_validity", "No recorded leader validity flags")
    elif valid.shape != (count,) or valid.dtype.kind not in "biu":
        error("source_validity_shape", "Leader validity flags must be a boolean sample vector")
        report["structural_valid"] = False
    else:
        flags = valid[:]
        if np.any((flags != 0) & (flags != 1)):
            error("source_validity_values", "Leader validity flags must contain only false/true values")
            report["structural_valid"] = False
        bad = np.flatnonzero(~flags.astype(bool))
        if bad.size:
            warn("invalid_leader_samples", "Commands held without valid fresh leader input; samples are retained", indices=bad.tolist())
    age = datasets.get("timing/source_age_s")
    stale_limit = metadata.get("device", {}).get("stale_s") if isinstance(metadata.get("device", {}), dict) else None
    if age is not None and age.shape == (count,) and stale_limit is not None:
        try:
            bad = np.flatnonzero(age[:] > float(stale_limit))
            if bad.size:
                warn("stale_leader_samples", "Leader age exceeds the recorded stale-input threshold", indices=bad.tolist())
        except (ValueError, TypeError):
            warn("missing_provenance", "Invalid stale-input threshold", field="device.stale_s")
    pre = datasets.get("timing/pre_monotonic_ns")
    post = datasets.get("timing/post_monotonic_ns")
    if pre is not None and post is not None and pre.shape == post.shape == (count,) and count:
        durations_s = (post[:] - pre[:]).astype(np.float64) / 1e9
        if np.any(durations_s < 0) or np.any(np.diff(pre[:]) < 0):
            warn("nonmonotonic_wall_timing", "Wall-clock diagnostics contain negative/nonmonotonic intervals")
        report["timing"]["wall_step_duration_max_s"] = float(durations_s.max())
        bad = np.flatnonzero(durations_s > 0.5)
        if bad.size:
            warn("large_wall_gap", "Recorded wall-time action interval exceeds 0.5 second", indices=bad.tolist())
    try:
        report["fingerprint"] = _fingerprint(datasets)
    except (OSError, TypeError, ValueError) as exc:
        error("unreadable_content", str(exc))
        report["structural_valid"] = False
    report["training_eligible"] = bool(report["structural_valid"] and report["native_15_hz"]
                                      and report["outcome"] == "success" and not warnings)
    return report


def validate_recording(path):
    """Return a source-bound JSON-safe index for a finalized HDF5 recording.

    Hashes are checked before/after inspection to reject concurrent mutation.
    This is not a replacement for the caller keeping live collector outputs
    out of the processing queue.
    """
    path = Path(path).resolve()
    checksum = _file_sha256(path)
    report = {"schema_version": 1, "source": {"path": str(path), "sha256": checksum},
              "metadata": {}, "episodes": [], "errors": [], "warnings": []}
    with h5py.File(path, "r") as stream:
        if "data" not in stream or not isinstance(stream["data"], h5py.Group):
            report["errors"].append({"code": "missing_data_group", "message": "HDF5 source lacks data group"})
        else:
            data = stream["data"]
            metadata = {key: _plain(value) for key, value in data.attrs.items()}
            for key in ("robolab_omy_teleop", "robolab_omy_collection"):
                if key in metadata:
                    try:
                        def reject_nonfinite_constant(value):
                            raise ValueError(f"Nonfinite JSON metadata constant: {value}")
                        value = (json.loads(metadata[key], parse_constant=reject_nonfinite_constant)
                                 if isinstance(metadata[key], str) else metadata[key])
                        if not isinstance(value, dict):
                            raise ValueError("Metadata must decode to an object")
                        metadata[key] = value
                    except (TypeError, ValueError) as exc:
                        report["errors"].append({"code": "malformed_metadata", "message": str(exc), "field": key})
                        metadata[key] = {}
            report["metadata"] = metadata
            teleop = metadata.get("robolab_omy_teleop", {})
            if not teleop:
                report["errors"].append({"code": "missing_omy_metadata", "message": "Source lacks valid robolab_omy_teleop metadata"})
            groups = [(name, item) for name, item in data.items() if isinstance(item, h5py.Group) and name.startswith("demo_")]
            if not groups:
                report["errors"].append({"code": "no_episodes", "message": "Source contains no demo_N episode groups"})
            for _, group in sorted(groups, key=lambda pair: (0, int(pair[0][5:])) if pair[0][5:].isdigit() else (1, pair[0])):
                report["episodes"].append(validate_episode(group, teleop))
    if _file_sha256(path) != checksum:
        report["errors"].append({"code": "source_changed", "message": "Source content changed during validation"})
        for episode in report["episodes"]:
            episode["structural_valid"] = episode["training_eligible"] = False
    if report["errors"]:
        for episode in report["episodes"]:
            episode["training_eligible"] = False
    return report
