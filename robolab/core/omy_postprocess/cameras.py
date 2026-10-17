"""Lightweight OMY metadata and camera-alignment provenance.

No simulator is imported here. A recorded teleop session and a Space/READY
collection use the same absolute-joint action contract; the collection marker
describes its workflow, not a different environment or observation alignment.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np

from robolab.core.task.task_index import normalize_task_dirs, validate_task_source


OMY_TELEOP_ATTR = "robolab_omy_teleop"
OMY_COLLECTION_ATTR = "robolab_omy_collection"
FRAME_MAPPING_GROUP = "omy_materialized_alignment"
ACTION_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8)) + ("gripper",)


def _json_object(raw, label: str) -> dict:
    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        value = json.loads(raw)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ValueError(f"{label}: invalid JSON metadata") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label}: metadata must be a JSON object")
    return value


def _positive_number(value, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite positive number")
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return result


def read_omy_metadata(path: str | Path, *, task_override: str | None = None) -> dict:
    """Resolve both OMY recording types without a relative-IK fallback.

    Non-OMY files return ``{}``. Any OMY marker commits a file to the OMY
    metadata contract: malformed/missing companion metadata, unsupported
    actions, or conflicting task/timing information are errors, never an excuse
    to silently register a keyboard environment.
    """
    with h5py.File(path, "r") as file:
        data = file.get("data")
        if data is None:
            raise ValueError(f"{path}: recording has no 'data' group")
        has_teleop = OMY_TELEOP_ATTR in data.attrs
        has_collection = OMY_COLLECTION_ATTR in data.attrs
        if not has_teleop and not has_collection:
            return {}
        if not has_teleop:
            raise ValueError(f"{OMY_COLLECTION_ATTR} requires {OMY_TELEOP_ATTR}")
        source = _json_object(data.attrs[OMY_TELEOP_ATTR], OMY_TELEOP_ATTR)
        workflow = (_json_object(data.attrs[OMY_COLLECTION_ATTR], OMY_COLLECTION_ATTR)
                    if has_collection else {})

    if type(source.get("schema_version")) is not int or source["schema_version"] != 1:
        raise ValueError("Unsupported OMY recording schema_version; expected 1")
    task = source.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("OMY recording must name a nonempty task")
    if task_override is not None and task_override != task:
        raise ValueError(f"--task {task_override!r} conflicts with recorded OMY task {task!r}")
    if source.get("action_type") != "absolute_joint_position":
        raise ValueError("OMY materialization requires action_type='absolute_joint_position'")
    names = source.get("action_names")
    if not isinstance(names, list) or tuple(names) != ACTION_NAMES:
        raise ValueError("OMY action_names must be panda_joint1..7 followed by gripper")
    if source.get("joint_units") != "rad":
        raise ValueError("OMY materialization requires joint_units='rad'")
    if source.get("gripper_encoding") != "+1=open,-1=closed":
        raise ValueError("OMY materialization requires gripper_encoding='+1=open,-1=closed'")

    hz = _positive_number(source.get("control_hz"), "OMY control_hz")
    performance = source.get("performance")
    if not isinstance(performance, dict):
        raise ValueError("OMY recording requires effective performance timing metadata")
    dt = _positive_number(performance.get("physics_dt"), "OMY physics_dt")
    decimation_value = performance.get("decimation")
    decimation = _positive_number(decimation_value, "OMY decimation")
    if not decimation.is_integer():
        raise ValueError("OMY decimation must be a positive integer")
    if not math.isclose(dt * decimation, 1 / hz, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("OMY control_hz disagrees with physics_dt * decimation")
    if "control_hz" in performance and not math.isclose(
            _positive_number(performance["control_hz"], "OMY performance control_hz"), hz,
            rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("OMY control_hz disagrees with effective performance control_hz")
    device = source.get("device", {})
    if not isinstance(device, dict):
        raise ValueError("OMY device metadata must be an object")
    if "dt" in device and not math.isclose(
            _positive_number(device["dt"], "OMY device dt"), 1 / hz,
            rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("OMY device dt disagrees with control_hz")
    for key, expected in (("task", task), ("action_type", "absolute_joint_position")):
        if key in workflow and workflow[key] != expected:
            raise ValueError(f"OMY collection {key} conflicts with teleop metadata")
    if "control_hz" in workflow and not math.isclose(
            _positive_number(workflow["control_hz"], "OMY collection control_hz"), hz,
            rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("OMY collection control_hz conflicts with teleop metadata")

    result = {
        "omy_recording": True,
        "omy_collection": has_collection,
        "recording_kind": "collection" if has_collection else "teleop",
        "task": task,
        "control_hz": hz,
        "sim_dt": dt,
        "decimation": int(decimation),
        # Episodes carry their own recorded seeds. This is just environment
        # construction; the materializer resets each episode with its own seed.
        "base_seed": 0,
        "action_type": "absolute_joint_position",
        "action_names": list(ACTION_NAMES),
        "metadata_sources": [OMY_TELEOP_ATTR] + ([OMY_COLLECTION_ATTR] if has_collection else []),
    }
    if "task_dirs" in source:
        result["task_dirs"] = list(normalize_task_dirs(source["task_dirs"]))
    if "task_source" in source:
        result["task_source"] = validate_task_source(source["task_source"], task=task)
    if "task_dirs" in workflow and normalize_task_dirs(workflow["task_dirs"]) != normalize_task_dirs(source.get("task_dirs")):
        raise ValueError("OMY collection task_dirs conflicts with teleop metadata")
    return result


def select_materialization_metadata(path: str | Path, *, task_override: str | None = None) -> dict:
    """Prefer explicit OMY metadata, retaining legacy deferred-image behavior."""
    omy = read_omy_metadata(path, task_override=task_override)
    if omy:
        return omy
    # Importing replay's package initializer pulls in Isaac Lab. Retain its
    # legacy permissive deferred-metadata behavior directly, so this selection
    # remains a pure, pre-SimulationApp check.
    with h5py.File(path, "r") as file:
        raw = file["data"].attrs.get("robolab_deferred_images")
    if raw is None:
        return {}
    try:
        return _json_object(raw, "robolab_deferred_images")
    except ValueError:
        return {}


def preflight_omy_materialization(path: str | Path, *, episodes: list[int] | None = None,
                                 task_override: str | None = None,
                                 require_native15: bool = False) -> tuple[dict, dict]:
    """Reject malformed OMY inputs before a costly simulator launch.

    Timing/structure failures stop rendering. Quality warnings (for example,
    stale leader input) remain available for review previews and are not a
    license to erase samples. Native-15 processing can explicitly require that
    profile; legacy rates never get relabeled as 15 FPS.
    """
    metadata = select_materialization_metadata(path, task_override=task_override)
    if not metadata.get("omy_recording"):
        return metadata, {}
    from .validation import validate_recording

    report = validate_recording(path)
    if report["errors"]:
        raise ValueError(f"OMY recording validation failed: {report['errors']}")
    episode_reports = {item["group"]: item for item in report["episodes"]}
    requested = ([f"data/demo_{index}" for index in episodes] if episodes else list(episode_reports))
    if not requested:
        raise ValueError("OMY recording contains no episodes to materialize")
    for name in requested:
        if name not in episode_reports:
            raise ValueError(f"OMY materialization episode {name} is not in the recording")
        episode = episode_reports[name]
        if not episode["structural_valid"]:
            raise ValueError(f"{name}: OMY materialization validation failed: {episode['errors']}")
        if require_native15 and not episode["native_15_hz"]:
            raise ValueError(f"{name}: native 15 Hz materialization profile requires verified 15 Hz timing")
        # Legacy rates can still be materialized, but no rate is an excuse to
        # render discontinuous, missing, or mismatched simulated-time rows.
        validated_frame_timestamps(metadata, episode)
    return metadata, report


def validated_frame_timestamps(metadata: dict, episode_report: dict) -> np.ndarray:
    """Resolve only verified simulated-time frame stamps at the recorded rate."""
    hz = _positive_number(metadata["control_hz"], "camera control_hz")
    timing = episode_report["timing"]
    count = int(episode_report["sample_count"])
    if count < 1:
        raise ValueError("No recorded actions to align with camera frames")
    if math.isclose(hz, 15, rel_tol=0, abs_tol=1e-9):
        if not episode_report["native_15_hz"] or not timing["valid"]:
            raise ValueError("OMY native 15 Hz camera alignment requires verified native-15 timing")
        stamps = np.asarray(timing["timestamps_s"], dtype=np.float64)
    else:
        # The first exporter rejects these rates. This path only preserves
        # legacy materialization, using recorded elapsed time at its actual
        # cadence, without reinterpreting it as a native-15 dataset.
        stamps = np.asarray(timing.get("recorded_elapsed_s", []), dtype=np.float64)
        for field in ("interval_min_s", "interval_max_s"):
            if field not in timing or not math.isclose(timing[field], 1 / hz, rel_tol=0, abs_tol=1e-8):
                raise ValueError(f"Recorded camera timing does not match its declared {hz:g} Hz rate")
        if not math.isclose(timing.get("measured_action_duration_s", -1), count / hz,
                            rel_tol=0, abs_tol=1e-8):
            raise ValueError("Recorded camera trajectory has discontinuous action intervals")
    if stamps.shape != (count,) or not np.all(np.isfinite(stamps)):
        raise ValueError("Validated camera timestamps must be one finite value per action")
    if not np.allclose(stamps, np.arange(count, dtype=np.float64) / hz, rtol=0, atol=1e-8):
        raise ValueError(f"Recorded camera timestamps do not match its declared {hz:g} Hz rate")
    return stamps


def resolved_asset_snapshot(env_cfg) -> dict:
    """Capture currently resolvable scene-asset dependencies after Kit starts.

    This establishes a processing-time dependency snapshot, not proof that
    today's asset bytes match an unrecorded collection-time checksum. Remote,
    procedural, missing, or uninspectable dependencies explicitly keep the
    profile provisional. USD imports intentionally occur only inside this
    simulator-side function, never while indexing/validating a recording.
    """
    from .io import sha256_file

    roots = {}
    for name, entity in vars(env_cfg.scene).items():
        spawn = getattr(entity, "spawn", None)
        if spawn is None:
            continue
        for field in ("usd_path", "texture_file"):
            value = getattr(spawn, field, None)
            if value:
                roots[f"scene/{name}/spawn/{field}"] = str(value)
    result = {
        "schema_version": 1,
        "scope": "processing_time_snapshot_not_collection_time_verification",
        "roots": roots,
        "files": {},
        "unresolved": [],
        "unhashed": [],
        "inspection_errors": [],
        "historical_asset_identity_verified": False,
    }
    dependencies = set(roots.values())
    try:
        from pxr import Sdf, UsdUtils

        for root in roots.values():
            if Path(root).suffix.lower() not in (".usd", ".usda", ".usdc"):
                continue
            try:
                layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(root))
                for layer in layers:
                    identifier = str(layer.realPath or layer.identifier)
                    if identifier:
                        dependencies.add(identifier)
                dependencies.update(str(asset) for asset in assets)
                result["unresolved"].extend(str(asset) for asset in unresolved)
            except Exception as exc:
                result["inspection_errors"].append({"root": root, "error": str(exc)})
    except ImportError as exc:
        result["inspection_errors"].append({"error": str(exc)})
    for asset in sorted(dependencies):
        path = Path(asset)
        if "://" in asset or not path.is_file():
            result["unhashed"].append(asset)
            continue
        try:
            result["files"][str(path.resolve())] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        except OSError as exc:
            result["inspection_errors"].append({"root": asset, "error": str(exc)})
    result["unresolved"] = sorted(set(result["unresolved"]))
    result["resolved_local_closure"] = bool(roots) and not (
        result["unresolved"] or result["unhashed"] or result["inspection_errors"])
    return result


def write_frame_mapping(demo: h5py.Group, *, source_episode: str,
                        source_sha256: str, validated_timestamps: np.ndarray,
                        control_hz: float = 15.0) -> None:
    """Stamp a one-to-one frame/source-step map after strict timing validation.

    Callers must obtain ``validated_timestamps`` from the shared validator,
    not from an FPS assumption. Existing raw channels remain unchanged. The
    ``-1`` state-row sentinel refers to ``initial_state[0]``; later image rows
    refer to post-action ``states[t - 1]``. The final post-action state has no
    invented policy frame.
    """
    count = int(demo["actions"].shape[0])
    if count < 1:
        raise ValueError("No recorded actions to align with camera frames")
    hz = _positive_number(control_hz, "camera control_hz")
    timestamps = np.asarray(validated_timestamps, dtype=np.float64)
    expected = np.arange(count, dtype=np.float64) / hz
    if timestamps.shape != (count,) or not np.all(np.isfinite(timestamps)):
        raise ValueError("Validated camera timestamps must be one finite value per action")
    if not np.allclose(timestamps, expected, rtol=0, atol=1e-8):
        raise ValueError(f"Camera timestamps do not match the recorded {hz:g} Hz row contract")
    if FRAME_MAPPING_GROUP in demo:
        del demo[FRAME_MAPPING_GROUP]
    mapping = demo.create_group(FRAME_MAPPING_GROUP)
    mapping.attrs["schema_version"] = 1
    mapping.attrs["source_episode"] = source_episode
    mapping.attrs["source_sha256"] = source_sha256
    mapping.attrs["fps"] = hz
    mapping.attrs["native_15_hz"] = math.isclose(hz, 15, rel_tol=0, abs_tol=1e-9)
    mapping.attrs["alignment"] = "pre_action: frame[0]=initial_state[0]; frame[t]=states[t-1]"
    mapping.attrs["physics_replay"] = False
    mapping.create_dataset("source_step", data=np.arange(count, dtype=np.int64))
    mapping.create_dataset("post_state_row", data=np.arange(count, dtype=np.int64) - 1)
    mapping.create_dataset("timestamp_s", data=timestamps)
    mapping.create_dataset("recorded_sim_before_s", data=demo["timing/sim_before_s"][()])
    mapping.create_dataset("recorded_sim_after_s", data=demo["timing/sim_after_s"][()])
