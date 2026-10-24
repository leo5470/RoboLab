"""Offline, native-15 OMY row mapping and an explicitly recipe-gated PPS writer.

The numerical diagnostic artifact is useful before checkpoint choices are frozen.
It is deliberately not a LeRobot dataset and is never advertised as training-ready.
LeRobot is imported lazily: collection and validation do not depend on it.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import h5py
import numpy as np


ARM_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
CAMERA_FEATURES = ("exterior_image_1_left", "wrist_image_left")
FPS = 15


class ExportError(ValueError):
    """Unsafe or underspecified export; the immutable source is not changed."""


@dataclass(frozen=True)
class PPSRecipe:
    """Explicit mapping choices; unresolved PPS dependencies may remain null.

    ``raw_finger_joint`` and ``normalized_closure`` are different measured-state
    conventions. Neither is inferred from the issued gripper command.
    """

    contents: Mapping[str, Any]

    @classmethod
    def from_dict(cls, contents: Mapping[str, Any]) -> "PPSRecipe":
        # Own a JSON snapshot rather than a caller's mutable dictionary.
        copied = json.loads(json.dumps(contents, allow_nan=False))
        if copied.get("schema_version") != 1 or copied.get("fps") != FPS:
            raise ExportError("Recipe must declare schema_version=1 and native fps=15; no resampling is supported")
        if tuple(copied.get("arm_joint_names", ())) != ARM_JOINT_NAMES:
            raise ExportError("Recipe arm_joint_names must be the seven ordered Franka joints")
        measured = copied.get("measured_gripper", {})
        conventions = {"raw_finger_joint": "rad", "normalized_closure": "normalized_closure"}
        if measured.get("source") not in conventions or measured.get("units") != conventions.get(measured.get("source")):
            raise ExportError("Choose measured_gripper.source/units explicitly: raw_finger_joint/rad or normalized_closure/normalized_closure")
        if copied.get("gripper_command", {}).get("encoding") not in ("zero_one_closed", "omy_plusminus_open"):
            raise ExportError("Choose gripper_command.encoding explicitly: zero_one_closed or omy_plusminus_open")
        return cls(copied)

    @property
    def measured_gripper(self) -> str:
        return self.contents["measured_gripper"]["source"]

    @property
    def command_encoding(self) -> str:
        return self.contents["gripper_command"]["encoding"]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_json_bytes(self.contents)).hexdigest()

    def require_training_contract(self) -> None:
        """Check recipe completeness, not compatibility certification.

        The recipe records the selected source/checkpoint identities. Actual
        acceptance by both pinned PPS loaders is a separate acceptance test.
        """
        pps = self.contents.get("pps", {})
        if not re.fullmatch(r"[0-9a-f]{40}", str(pps.get("revision", ""))):
            raise ExportError("Training export requires a full pinned PPS revision")
        if not pps.get("base_checkpoint"):
            raise ExportError("Training export requires the base/teacher checkpoint identity")
        norm = pps.get("normalization", {})
        if not norm.get("artifact") or not re.fullmatch(r"[0-9a-f]{64}", str(norm.get("sha256", ""))):
            raise ExportError("Training export requires a recoverable normalization artifact and SHA256")
        for role in ("reference", "task"):
            selected = pps.get("proxies", {}).get(role, {})
            if not selected.get("config") or not isinstance(selected.get("action_horizon"), int) or isinstance(selected["action_horizon"], bool) or selected["action_horizon"] <= 0:
                raise ExportError(f"Training export requires the {role} proxy configuration and positive action_horizon")
        backend = self.contents.get("lerobot", {})
        version = str(backend.get("version") or "")
        if not re.fullmatch(r"\d+\.\d+\.\d+(?:[a-zA-Z0-9.+-]*)", version):
            raise ExportError("Choose an exact LeRobot version in an isolated export environment; no dependency will be installed automatically")
        if backend.get("api") != "common_v2" or backend.get("image_storage") != "image":
            raise ExportError("This provisional adapter supports only explicit common_v2 API and image storage; video/v3 need separate acceptance")
        if backend.get("finalize") not in ("save_episode", "consolidate"):
            raise ExportError("Choose LeRobot finalize explicitly: save_episode or consolidate")
        self.camera_paths()

    def camera_paths(self) -> dict[str, str]:
        cameras = self.contents.get("cameras", {})
        paths = {}
        for name in CAMERA_FEATURES:
            selected = cameras.get(name, {})
            if not isinstance(selected.get("source"), str) or not selected["source"]:
                raise ExportError(f"Recipe must specify the materialized {name} HDF5 path")
            if tuple(selected.get("shape", ())) != (180, 320, 3):
                raise ExportError(f"Recipe {name} must be RGB [180,320,3]; implicit resizing is forbidden")
            if selected.get("color") != "RGB" or selected.get("dtype") != "uint8":
                raise ExportError(f"Recipe {name} must explicitly declare RGB uint8")
            paths[name] = selected["source"]
        return paths


def load_recipe(path: str | Path) -> PPSRecipe:
    return PPSRecipe.from_dict(json.loads(Path(path).read_text()))


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _metadata(file: h5py.File) -> dict[str, Any]:
    raw = file["data"].attrs.get("robolab_omy_teleop")
    if raw is None:
        raise ExportError("Recording lacks robolab_omy_teleop mapping metadata")
    if isinstance(raw, bytes):
        raw = raw.decode()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ExportError("robolab_omy_teleop metadata must be an object")
    return value


def _validate_source(path: Path, expected_sha256: str) -> None:
    from .validation import validate_recording

    report = validate_recording(path)
    if report["errors"]:
        raise ExportError(f"Recording-level validation failed: {report['errors']}")
    if report["source"]["sha256"] != expected_sha256:
        raise ExportError("Source recording changed before row mapping")


def _validate_group(group: h5py.Group, metadata: Mapping[str, Any], *, source_sha256: str, diagnostic: bool, review: Mapping[str, Any] | None) -> dict[str, Any]:
    from .validation import validate_episode

    result = validate_episode(group, metadata)
    if not result["structural_valid"] or not result["native_15_hz"]:
        codes = ", ".join(item["code"] for item in result["errors"])
        raise ExportError(f"Export refuses structurally invalid or non-native-15 episode {group.name}: {codes}")
    if review is not None:
        expected = {"source_sha256": source_sha256, "source_group": group.name.lstrip("/"), "fingerprint": result["fingerprint"]}
        if any(review.get(name) != value for name, value in expected.items()):
            raise ExportError("Review source_sha256, source_group and fingerprint must match this exact recording")
        if not review.get("reviewer") or not review.get("reason"):
            raise ExportError("Recorded review requires a reviewer and reason")
    if not diagnostic:
        if result.get("training_excluded"):
            raise ExportError("Recording is explicitly excluded from training; use diagnostic mapping to inspect it")
        if result["outcome"] != "success":
            raise ExportError("Training selection requires a successful episode; use diagnostic mapping to inspect failures")
        warning_codes = {item["code"] for item in result["warnings"]}
        reviewed = set((review or {}).get("accepted_warning_codes", ()))
        if warning_codes and (not review or not review.get("reviewer") or not review.get("reason") or warning_codes - reviewed):
            raise ExportError(f"Episode has unresolved quality warnings {sorted(warning_codes)}; supply a recorded reviewer/reason and accepted_warning_codes or use diagnostic mapping")
        if reviewed - warning_codes:
            raise ExportError("Review must match this episode's warning codes, not another recording")
    return result


def _attempt_identity(file: h5py.File, report: Mapping[str, Any]) -> dict[str, Any]:
    raw = file["data"].attrs.get("robolab_omy_attempt")
    if raw is None:
        return {}
    try:
        identity = json.loads(raw)
        if not isinstance(identity, dict) or not isinstance(identity.get("attempt_id"), str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", identity["attempt_id"]):
            raise ValueError("Missing attempt ID")
        uuid.UUID(identity["attempt_uuid"])
        if identity.get("fingerprint") != report["fingerprint"]:
            raise ValueError("Fingerprint mismatch")
        source = identity.get("source", {})
        if not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", ""))) or not source.get("group"):
            raise ValueError("Missing original source identity")
    except (ValueError, TypeError, KeyError) as error:
        raise ExportError("Packaged robolab_omy_attempt identity is malformed or differs from the recorded trajectory") from error
    return identity


def _check_attempt_outcome(identity: Mapping[str, Any], *, diagnostic: bool) -> None:
    if not identity or diagnostic:
        return
    if identity.get("outcome") != "success":
        raise ExportError("Packaged attempt outcome is not success; raw success attributes cannot override interrupted/failure evidence")
    event = identity.get("end_event")
    if event is not None and (not isinstance(event, dict) or isinstance(event.get("result"), str)):
        raise ExportError("Packaged attempt has interrupted/manual-reset end evidence; use diagnostic mapping only")


def _check_mapping_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("action_type") != "absolute_joint_position" or metadata.get("joint_units") != "rad":
        raise ExportError("OMY action metadata must declare absolute_joint_position in radians")
    if tuple(metadata.get("action_names", ())) != (*ARM_JOINT_NAMES, "gripper"):
        raise ExportError("Issued actions must use the declared ordered seven Franka joints plus gripper")
    if metadata.get("gripper_encoding") != "+1=open,-1=closed":
        raise ExportError("OMY source gripper command convention is missing or unsupported")


def _image_sources(group: h5py.Group, recipe: PPSRecipe, count: int) -> dict[str, h5py.Dataset]:
    selected = {}
    for name, path in recipe.camera_paths().items():
        if path not in group:
            raise ExportError(f"Policy camera not materialized: {group.name}/{path}")
        dataset = group[path]
        if not isinstance(dataset, h5py.Dataset) or dataset.shape != (count, 180, 320, 3) or dataset.dtype != np.dtype("uint8"):
            raise ExportError(f"Camera {path} must contain exactly {count} uint8 RGB frames [180,320,3]")
        selected[name] = dataset
    # Image arrays alone cannot establish whether frame t is pre- or post-action.
    alignment = group.file["data"].attrs.get("robolab_materialized")
    if isinstance(alignment, bytes):
        alignment = alignment.decode()
    if alignment is None or json.loads(alignment).get("alignment") != "pre_action":
        raise ExportError("Materialized camera metadata must certify pre_action alignment")
    provenance = json.loads(alignment)
    mapping_name = "omy_materialized_alignment"
    if mapping_name not in group or not isinstance(group[mapping_name], h5py.Group):
        raise ExportError("Policy cameras lack their verified native-15 source-step mapping")
    mapping = group[mapping_name]
    expected = {"source_step": np.arange(count, dtype=np.int64), "post_state_row": np.arange(count, dtype=np.int64) - 1, "timestamp_s": np.arange(count, dtype=np.float64) / FPS, "recorded_sim_before_s": group["timing/sim_before_s"][:], "recorded_sim_after_s": group["timing/sim_after_s"][:]}
    for name, values in expected.items():
        if name not in mapping or not isinstance(mapping[name], h5py.Dataset) or not np.array_equal(mapping[name][:], values):
            raise ExportError(f"Policy-camera mapping {name} differs from the recorded native-15 pre-action source")
    for name in ("timestamp_s", "recorded_sim_before_s", "recorded_sim_after_s"):
        if mapping[name].dtype != np.dtype("float64"):
            raise ExportError(f"Policy-camera mapping {name} must retain float64 simulation timing")
    actual_group = group.name.lstrip("/")
    source_episode = mapping.attrs.get("source_episode", "")
    if isinstance(source_episode, bytes):
        source_episode = source_episode.decode()
    if f"data/{source_episode}" != actual_group or ("source_group" in mapping.attrs and mapping.attrs["source_group"] != actual_group):
        raise ExportError("Policy-camera source episode/group metadata differs from the selected attempt")
    source_sha = mapping.attrs.get("source_sha256", "")
    if isinstance(source_sha, bytes):
        source_sha = source_sha.decode()
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha) or source_sha != provenance.get("source_sha256"):
        raise ExportError("Policy-camera source checksum differs from materialization provenance")
    if source_episode not in provenance.get("episodes", ()):
        raise ExportError("Selected camera episode is absent from materialization provenance")
    if mapping.attrs.get("fps") != FPS or mapping.attrs.get("native_15_hz") != True or mapping.attrs.get("physics_replay") != False:
        raise ExportError("Policy-camera mapping does not declare native 15 Hz state restoration without physics replay")
    return selected


def _rows(group: h5py.Group, metadata: Mapping[str, Any], report: Mapping[str, Any], recipe: PPSRecipe, *, require_images: bool) -> Iterator[dict[str, Any]]:
    count = report["sample_count"]
    joint_names = report["joint_names"]
    finger = joint_names.index("finger_joint")
    initial = group["initial_state/articulation/robot/joint_position"][0]
    states = group["states/articulation/robot/joint_position"]
    images = _image_sources(group, recipe, count) if require_images else {}
    if recipe.measured_gripper == "normalized_closure":
        declared = metadata.get("gripper_observation", {})
        if declared.get("units") != "normalized_closure" or declared.get("joint") != "finger_joint":
            raise ExportError("Normalized measured gripper export requires explicit finger_joint normalized_closure source metadata")
    origin = float(np.asarray(group["timing/sim_before_s"][0]).reshape(-1)[0])
    for step in range(count):
        # All sources are direct, bounded per-row reads. No terminal-state row,
        # action lookahead, arm-delta conversion or command interpolation.
        action = np.asarray(group["actions"][step], dtype=np.float32).copy()
        if action[-1] not in (-1.0, 1.0):
            raise ExportError(f"Issued binary OMY gripper command at step {step} is not +1 or -1")
        original = action.copy()
        if recipe.command_encoding == "zero_one_closed":
            action[-1] = (1.0 - action[-1]) / 2.0
        joints = np.asarray(group["obs/proprio_obs/arm_joint_pos"][step], dtype=np.float32).copy()
        normalized = np.asarray(group["obs/proprio_obs/gripper_pos"][step], dtype=np.float32).reshape(1).copy()
        if normalized[0] < -1e-5 or normalized[0] > 1.0 + 1e-5:
            raise ExportError(f"Normalized measured gripper closure is outside [0,1] at step {step}; no clipping is performed")
        raw_finger = np.array([(initial if step == 0 else states[step - 1])[finger]], dtype=np.float32)
        row = {
            "source_step": step,
            # Raw times were already validated against this float64 grid. This
            # avoids accumulated timestamp rounding and ignores wall stalls.
            "timestamp": np.float64(step / FPS),
            "source_sim_before_s": np.float64(np.asarray(group["timing/sim_before_s"][step]).reshape(-1)[0]),
            "source_sim_origin_s": np.float64(origin),
            "joint_position": joints,
            "gripper_position": raw_finger.copy() if recipe.measured_gripper == "raw_finger_joint" else normalized.copy(),
            "actions": action,
            "original_actions": original,
            "raw_finger_position": raw_finger,
            "normalized_gripper_position": normalized,
        }
        for name, dataset in images.items():
            row[name] = dataset[step]
        yield row


def iter_pps_rows(source: str | Path, group: str = "data/demo_0", *, recipe: PPSRecipe, require_images: bool = False, diagnostic: bool = False, review: Mapping[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Yield exactly N pre-action rows for N verified native-15 issued actions.

    Diagnostic mode admits quality flags/outcomes for inspection, never bad
    timing or structure. No image arrays are loaded until their row is yielded.
    The caller must exhaust/close the iterator before changing its source file.
    """
    source = Path(source)
    before = _sha256(source)
    _validate_source(source, before)
    with h5py.File(source, "r") as file:
        metadata = _metadata(file)
        _check_mapping_metadata(metadata)
        selected = file[group]
        report = _validate_group(selected, metadata, source_sha256=before, diagnostic=diagnostic, review=review)
        identity = _attempt_identity(file, report)
        _check_attempt_outcome(identity, diagnostic=diagnostic)
        yield from _rows(selected, metadata, report, recipe, require_images=require_images)
    if _sha256(source) != before:
        raise ExportError("Source recording changed while rows were read; discard the derivative")


def _write_json(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _publish(partial: Path, target: Path) -> None:
    from .io import fsync_directory, publish_directory

    for path in partial.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    for directory in sorted((p for p in partial.rglob("*") if p.is_dir()), reverse=True):
        fsync_directory(directory)
    fsync_directory(partial)
    publish_directory(partial, target)


def export_canonical_rows(source: str | Path, output_directory: str | Path, *, group: str = "data/demo_0", recipe: PPSRecipe, diagnostic: bool = False, review: Mapping[str, Any] | None = None) -> Path:
    """Stream a numerical mapping artifact for round-trip/alignment review.

    This does not require camera materialization, PPS or LeRobot, and cannot
    become training-ready just by changing a JSON flag.
    """
    source, target = Path(source).resolve(), Path(output_directory).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing existing output: {target}")
    source_hash = _sha256(source)
    _validate_source(source, source_hash)
    with h5py.File(source, "r") as file:
        metadata = _metadata(file)
        _check_mapping_metadata(metadata)
        selected = file[group]
        report = _validate_group(selected, metadata, source_sha256=source_hash, diagnostic=diagnostic, review=review)
        count = report["sample_count"]
        identity = _attempt_identity(file, report)
        _check_attempt_outcome(identity, diagnostic=diagnostic)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=target.parent))
        with h5py.File(partial / "rows.hdf5", "w") as output:
            output.attrs.update(schema_version=1, fps=FPS, training_ready=False, alignment="pre_action", source_sha256=source_hash, source_group=group)
            shapes = {"source_step": (), "timestamp": (), "source_sim_before_s": (), "source_sim_origin_s": (), "joint_position": (7,), "gripper_position": (1,), "actions": (8,), "original_actions": (8,), "raw_finger_position": (1,), "normalized_gripper_position": (1,)}
            for name, shape in shapes.items():
                dtype = np.int64 if name == "source_step" else np.float64 if name in ("timestamp", "source_sim_before_s", "source_sim_origin_s") else np.float32
                output.create_dataset(name, shape=(count, *shape), dtype=dtype, chunks=(min(count, 256), *shape))
            for row in _rows(selected, metadata, report, recipe, require_images=False):
                for name in shapes:
                    output[name][row["source_step"]] = row[name]
        if _sha256(source) != source_hash:
            raise ExportError(f"Source changed during mapping; unpublished diagnostics retained at {partial}")
        _write_json(partial / "recipe.json", recipe.contents)
        _write_json(partial / "validation.json", report)
        manifest = {
            "schema_version": 1, "artifact_kind": "omy_native15_numerical_mapping", "training_ready": False,
            "source": {"path": str(source), "sha256": source_hash, "group": group, "fingerprint": report.get("fingerprint")},
            "attempt": identity, "fps": FPS, "sample_count": count,
            "row_mapping": {"source_step_first": 0, "source_step_last": count - 1, "stride": 1, "alignment": "pre_action", "action_shift": 0},
            "duration_s": count / FPS, "last_row_timestamp_s": (count - 1) / FPS,
            "diagnostic": diagnostic, "review": dict(review) if review is not None else None,
            "recipe_sha256": recipe.sha256, "files": {"rows.hdf5": _sha256(partial / "rows.hdf5"), "recipe.json": _sha256(partial / "recipe.json"), "validation.json": _sha256(partial / "validation.json")},
        }
        _write_json(partial / "manifest.json", manifest)
    _publish(partial, target)
    return target


@dataclass(frozen=True)
class EpisodeSelection:
    source: Path
    group: str
    attempt_id: str
    attempt_uuid: str
    split: str
    roles: tuple[str, ...]
    task_instruction: str
    review: Mapping[str, Any] | None = None


def _lerobot_class(recipe: PPSRecipe):
    requested = recipe.contents["lerobot"]["version"]
    try:
        installed = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError as error:
        raise ExportError(f"LeRobot {requested} is not installed. Use a separate pinned PPS export environment; do not sync or modify the live simulation .venv") from error
    if installed != requested:
        raise ExportError(f"LeRobot version mismatch: recipe requires {requested}, found {installed}")
    try:
        return importlib.import_module("lerobot.common.datasets.lerobot_dataset").LeRobotDataset
    except (ImportError, AttributeError) as error:
        raise ExportError("Selected version does not provide the explicit common_v2 API; choose and test a compatible export recipe") from error


def _verify_lerobot_rows(dataset_root: Path, selections: Sequence[EpisodeSelection], reports: Sequence[Mapping[str, Any]], recipe: PPSRecipe) -> dict[str, Any]:
    """Read actual written rows in bounded Parquet batches, not just API calls.

    LeRobot may encode timestamps as float32. They must be exactly the float32
    representation of source_step/15, while the authoritative sidecar keeps
    float64. No arbitrary FPS setting can substitute for these value checks.
    """
    try:
        parquet = importlib.import_module("pyarrow.parquet")
    except ImportError as error:
        raise ExportError("Actual LeRobot output validation requires pyarrow in the isolated export environment") from error
    paths = sorted((dataset_root / "data").rglob("*.parquet"))
    if not paths:
        raise ExportError("Selected LeRobot writer produced no inspectable Parquet rows")
    columns = ["episode_index", "frame_index", "timestamp", "joint_position", "gripper_position", "actions"]
    counts = [0] * len(selections)
    current_episode = -1
    expected_rows = None
    for path in paths:
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
            records = batch.to_pydict()
            for row in range(batch.num_rows):
                episode = int(records["episode_index"][row])
                frame = int(records["frame_index"][row])
                if episode < 0 or episode >= len(selections) or frame != counts[episode]:
                    raise ExportError("Written LeRobot episode/frame indices are missing, duplicated, or reordered")
                if episode != current_episode:
                    if episode != current_episode + 1:
                        raise ExportError("Written LeRobot episode order differs from the deterministic selection manifest")
                    if expected_rows is not None:
                        try:
                            next(expected_rows)
                        except StopIteration:
                            pass
                        else:
                            raise ExportError("Written LeRobot episode ended before all source rows")
                    selected = selections[episode]
                    expected_rows = iter_pps_rows(selected.source, selected.group, recipe=recipe, review=selected.review)
                    current_episode = episode
                try:
                    expected = next(expected_rows)
                except StopIteration as error:
                    raise ExportError("Written LeRobot episode contains extra source rows") from error
                actual_time = records["timestamp"][row]
                expected_time = expected["timestamp"]
                if not np.isfinite(actual_time) or np.float32(actual_time) != np.float32(expected_time):
                    raise ExportError("Written LeRobot timestamps do not match the verified native-15 source-step grid")
                for feature in ("joint_position", "gripper_position", "actions"):
                    actual = np.asarray(records[feature][row], dtype=np.float32)
                    if actual.shape != expected[feature].shape or not np.array_equal(actual, expected[feature]):
                        raise ExportError(f"Written LeRobot {feature} does not round-trip to the recorded pre-action row")
                counts[episode] += 1
    if counts != [report["sample_count"] for report in reports]:
        raise ExportError("Written LeRobot frame counts differ from the immutable source attempts")
    if expected_rows is not None:
        try:
            next(expected_rows)
        except StopIteration:
            pass
        else:
            raise ExportError("Written LeRobot episode is missing final source rows")
    return {"status": "passed", "frame_counts": counts, "fps": FPS, "numeric_round_trip": "exact_float32", "timestamp_contract": "float32(source_step/15); float64 in row_sources.jsonl"}


def export_lerobot(selections: Sequence[EpisodeSelection], output_directory: str | Path, *, recipe: PPSRecipe, repo_id: str, _dataset_class=None) -> Path:
    """Provisional, local-only adapter for the pinned recipe's common_v2 API.

    Requires resolved recipe fields, reviewed quality, already aligned cameras,
    and an explicitly selected split/role per attempt. No remote upload occurs.
    A successful conversion is NOT both-proxy loader compatibility certification.
    ``_dataset_class`` is an injection hook for pure writer-contract tests only.
    """
    recipe.require_training_contract()
    if not selections:
        raise ExportError("Select at least one nonempty episode")
    target = Path(output_directory).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing existing output: {target}")
    reports, fingerprints, ids, snapshots = [], set(), set(), []
    attempt_ids = set()
    for item in selections:
        if not item.attempt_id or not item.attempt_uuid or not item.task_instruction:
            raise ExportError("Every selection requires persistent attempt identity and a task instruction")
        if item.split not in ("train", "validation", "test") or not item.roles or set(item.roles) - {"reference", "task"}:
            raise ExportError("Every attempt needs an explicit episode-level split and reference/task role selection")
        if item.attempt_uuid in ids or item.attempt_id in attempt_ids:
            raise ExportError("Repeated attempt UUID in export selections")
        ids.add(item.attempt_uuid)
        attempt_ids.add(item.attempt_id)
        source = Path(item.source)
        source_hash = _sha256(source)
        _validate_source(source, source_hash)
        with h5py.File(source, "r") as file:
            metadata = _metadata(file)
            _check_mapping_metadata(metadata)
            report = _validate_group(file[item.group], metadata, source_sha256=source_hash, diagnostic=False, review=item.review)
            identity = _attempt_identity(file, report)
            if not identity:
                raise ExportError("Training conversion requires immutable packaged robolab_omy_attempt identity; package the raw recording first")
            _check_attempt_outcome(identity, diagnostic=False)
            try:
                uuid.UUID(item.attempt_uuid)
            except (ValueError, TypeError, AttributeError) as error:
                raise ExportError("EpisodeSelection attempt_uuid must be the valid UUID of its packaged source") from error
            if identity["attempt_id"] != item.attempt_id or identity["attempt_uuid"] != item.attempt_uuid:
                raise ExportError("EpisodeSelection identity does not match the packaged attempt ID/UUID")
            _image_sources(file[item.group], recipe, report["sample_count"])
        fingerprint = report["fingerprint"]
        if fingerprint in fingerprints:
            raise ExportError("Duplicate episode content is selected; refusing possible train/evaluation leakage")
        fingerprints.add(fingerprint)
        reports.append(report)
        snapshots.append(source_hash)
    dataset_class = _dataset_class or _lerobot_class(recipe)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=target.parent))
    dataset_root = partial / "dataset"
    features = {name: {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]} for name in CAMERA_FEATURES}
    features.update({name: {"dtype": "float32", "shape": shape, "names": [name]} for name, shape in (("joint_position", (7,)), ("gripper_position", (1,)), ("actions", (8,)))})
    # A single synchronous image writer bounds backlog; no worker fan-out into
    # the simulation environment. v3/video adapters are intentionally absent.
    dataset = dataset_class.create(repo_id=repo_id, root=dataset_root, fps=FPS, robot_type="panda", features=features, use_videos=False, image_writer_threads=0, image_writer_processes=0)
    manifests = {role: {"schema_version": 1, "role": role, "recipe_sha256": recipe.sha256, "pps": recipe.contents["pps"], "episodes": []} for role in ("reference", "task")}
    with (partial / "row_sources.jsonl").open("x") as mapping_file:
        for index, (selected, report, source_hash) in enumerate(zip(selections, reports, snapshots, strict=True)):
            rows_written = 0
            for row in iter_pps_rows(selected.source, selected.group, recipe=recipe, require_images=True, review=selected.review):
                frame = {name: row[name] for name in features}
                frame.update(task=selected.task_instruction, timestamp=float(row["timestamp"]))
                dataset.add_frame(frame)
                mapping_file.write(json.dumps({"episode_index": index, "frame_index": row["source_step"], "source_step": row["source_step"], "timestamp": float(row["timestamp"]), "attempt_id": selected.attempt_id, "attempt_uuid": selected.attempt_uuid, "source_sha256": source_hash, "source_group": selected.group}, sort_keys=True) + "\n")
                rows_written += 1
            if rows_written != report["sample_count"] or _sha256(Path(selected.source)) != source_hash:
                raise ExportError(f"Source changed or frame count mismatch; unpublished export retained at {partial}")
            dataset.save_episode()
            episode = {"episode_index": index, "attempt_id": selected.attempt_id, "attempt_uuid": selected.attempt_uuid, "split": selected.split, "sample_count": rows_written, "source": {"path": str(selected.source), "sha256": source_hash, "group": selected.group, "fingerprint": report["fingerprint"]}, "task": selected.task_instruction, "review": dict(selected.review) if selected.review else None}
            for role in selected.roles:
                manifests[role]["episodes"].append(episode)
        mapping_file.flush()
        os.fsync(mapping_file.fileno())
    if recipe.contents["lerobot"]["finalize"] == "consolidate":
        finalize = getattr(dataset, "consolidate", None)
        if not callable(finalize):
            raise ExportError(f"Selected LeRobot version lacks recipe's consolidate API; partial export retained at {partial}")
        finalize()
    row_verification = _verify_lerobot_rows(dataset_root, selections, reports, recipe) if _dataset_class is None else {"status": "not_run_test_backend"}
    _write_json(partial / "recipe.json", recipe.contents)
    for role, manifest in manifests.items():
        _write_json(partial / f"{role}_selection.json", manifest)
    _write_json(partial / "manifest.json", {"schema_version": 1, "artifact_kind": "provisional_pps_lerobot_export", "training_ready": False, "loader_acceptance": "pending_both_pinned_proxy_loaders", "row_verification": row_verification, "fps": FPS, "alignment": "pre_action", "action_shift": 0, "resampling": None, "recipe_sha256": recipe.sha256, "episode_count": len(selections), "frame_count": sum(report["sample_count"] for report in reports), "row_mapping_sha256": _sha256(partial / "row_sources.jsonl"), "dependency_version": recipe.contents["lerobot"]["version"]})
    _publish(partial, target)
    return target
