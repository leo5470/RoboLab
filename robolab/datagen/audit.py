"""Read-only source checks and a static task census; no simulator required.

The census reports conversion *candidates*, never generation readiness. Runtime
predicate extraction and comparison with scoring subtasks remain separate gates.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_versions() -> dict:
    result = {}
    for name in ("isaaclab", "isaacsim", "torch", "numpy", "h5py"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def validate_controller(config: dict) -> dict:
    """Fail closed on controller conventions not supported by the first adapter."""
    actions = config.get("actions", {})
    if list(actions) != ["arm_action", "finger_joint"]:
        raise ValueError("Expected arm_action followed by finger_joint")
    arm = actions["arm_action"]
    controller = arm.get("controller", {})
    if (arm.get("asset_name") != "robot" or arm.get("body_name") != "base_link"
            or controller.get("command_type") != "pose" or controller.get("use_relative_mode") is not True):
        raise ValueError("Expected DROID robot/base_link relative pose IK")
    if not str(arm.get("class_type", "")).endswith(":DifferentialInverseKinematicsAction"):
        raise ValueError("Expected DifferentialInverseKinematicsAction")
    if arm.get("clip") is not None:
        raise ValueError("Clipped IK commands are not supported by the initial adapter")
    offset = arm.get("body_offset") or {}
    position = np.asarray(offset.get("pos", [0, 0, 0]))
    rotation = np.asarray(offset.get("rot", [1, 0, 0, 0]))
    if (position.shape != (3,) or rotation.shape != (4,)
            or not np.allclose(position, [0, 0, 0]) or not np.allclose(rotation, [1, 0, 0, 0])):
        raise ValueError("Initial adapter requires the recorded identity base_link body offset")
    scale = np.asarray(arm.get("scale"), dtype=float)
    if scale.shape not in ((), (6,)) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("IK scale must be a positive finite scalar or six-vector")
    finger = actions["finger_joint"]
    if (not str(finger.get("class_type", "")).endswith(":BinaryJointPositionZeroToOneAction")
            or finger.get("asset_name") != "robot" or finger.get("joint_names") != ["finger_joint"]
            or finger.get("open_command_expr") != {"finger_joint": 0.0}
            or not np.isclose(finger.get("close_command_expr", {}).get("finger_joint", np.nan), np.pi / 4)
            or finger.get("clip") is not None):
        raise ValueError("Expected DROID Robotiq gripper: float 0=open, 1=closed")
    return {"eef": "droid", "body": "base_link", "frame": "robot_root", "scale": scale.tolist(),
            "gripper": {"open": 0.0, "closed": 1.0}}


def _datasets(group):
    result = {}
    group.visititems(lambda name, item: result.update({name: item}) if isinstance(item, h5py.Dataset) else None)
    return result


def audit_dataset(path: str | Path) -> dict:
    path = Path(path).resolve()
    report = {"path": str(path), "sha256": fingerprint(path), "errors": [], "warnings": [], "episodes": []}
    sidecar = path.with_name("env_cfg.json")
    if not sidecar.is_file():
        report["errors"].append("Missing env_cfg.json: controller and scene cannot be reconstructed faithfully")
    else:
        report["config_sha256"] = fingerprint(sidecar)
        try:
            config = json.loads(sidecar.read_text())
            report["controller"] = validate_controller(config)
            report["control_dt"] = float(config["sim"]["dt"]) * int(config["decimation"])
            if not np.isfinite(report["control_dt"]) or report["control_dt"] <= 0:
                raise ValueError("Invalid control timestep")
        except (ValueError, KeyError, TypeError) as error:
            report["errors"].append(str(error))
    with h5py.File(path, "r") as file:
        if "data" not in file:
            report["errors"].append("Missing data group")
            return report
        data = file["data"]
        report["recorded_versions"] = {key: str(data.attrs.get(key, "unknown"))
                                        for key in ("isaaclab_version", "isaacsim_version")}
        raw_metadata = data.attrs.get("robolab_deferred_images", "{}")
        try:
            report["collection"] = json.loads(raw_metadata)
            if not isinstance(report["collection"], dict) or not report["collection"].get("task"):
                report["errors"].append("Collection metadata must identify a task")
        except (ValueError, TypeError):
            report["errors"].append("Invalid robolab_deferred_images metadata")
        for name in sorted(data):
            episode = data[name]
            if not isinstance(episode, h5py.Group):
                continue
            errors = []
            leaves = _datasets(episode)
            action = leaves.get("actions")
            count = action.shape[0] if action is not None and action.ndim else 0
            success = bool(episode.attrs.get("success", False))
            if action is None or action.ndim != 2 or action.shape[1] != 7 or not count:
                errors.append("Expected nonempty actions with shape (T, 7)")
            if count != episode.attrs.get("num_samples"):
                errors.append("num_samples differs from action count")
            for required in ("initial_state/articulation/robot/joint_position",
                             "initial_state/articulation/robot/root_pose",
                             "states/articulation/robot/joint_position",
                             "states/articulation/robot/root_pose",
                             "ee_pose/position", "ee_pose/orientation",
                             "robot_root_pose/position", "robot_root_pose/orientation"):
                if required not in leaves:
                    errors.append(f"Missing {required}")
            if "states/rigid_object" not in episode:
                errors.append("Missing rigid-object state trajectories")
            for key, leaf in leaves.items():
                expected_width = None
                if key.endswith("/root_pose"):
                    expected_width = 7
                elif key.startswith(("ee_pose/", "robot_root_pose/")):
                    expected_width = {"position": 3, "orientation": 4}.get(key.rsplit("/", 1)[-1])
                if expected_width is not None and (leaf.ndim != 2 or leaf.shape[1] != expected_width):
                    errors.append(f"Invalid pose shape: {key} {leaf.shape}")
                elif expected_width in (4, 7):
                    quaternion = leaf[:, -4:]
                    if not np.allclose(np.linalg.norm(quaternion, axis=-1), 1, atol=1e-3):
                        errors.append(f"Non-unit quaternion: {key}")
                if key.startswith(("states/", "ee_pose/", "robot_root_pose/", "obs/")):
                    if leaf.ndim == 0 or leaf.shape[0] != count:
                        errors.append(f"Length mismatch: {key} {leaf.shape}")
                if key.startswith("initial_state/") and (leaf.ndim == 0 or leaf.shape[0] != 1):
                    errors.append(f"Expected one initial-state row: {key}")
                if key.startswith("states/"):
                    initial = leaves.get("initial_state/" + key[len("states/"):])
                    if initial is None or initial.shape[1:] != leaf.shape[1:]:
                        errors.append(f"Missing or incompatible initial state for {key}")
                # Read numeric arrays in bounded chunks; never load all camera images.
                if leaf.dtype.kind in "fc" and leaf.ndim:
                    for start in range(0, leaf.shape[0], 128):
                        if not np.isfinite(leaf[start:start + 128]).all():
                            errors.append(f"Nonfinite values: {key}")
                            break
            if action is not None and action.ndim == 2 and action.shape[1] == 7:
                grip = action[:, 6]
                if not np.isin(grip, [0.0, 1.0]).all():
                    errors.append("Source gripper commands must be float 0 or 1")
                if action.dtype.kind != "f":
                    errors.append("Expected floating-point actions")
            report["episodes"].append({"name": name, "num_steps": count, "recorded_success": success,
                                       "errors": errors, "eligible_for_replay": success and not errors,
                                       "has_mimic_metadata": "obs/datagen_info" in episode})
    report["eligible_episode_count"] = sum(ep["eligible_for_replay"] for ep in report["episodes"])
    if not report["eligible_episode_count"]:
        report["errors"].append("No nonempty successful structurally valid episodes")
    report["ready_for_replay"] = not report["errors"]
    report["generation_ready"] = False
    return report


def audit_tasks(directory: str | Path) -> dict:
    """Inventory direct Task subclasses without importing task code or assets."""
    records = []
    helpers = {"pick_and_place", "pick_and_place_on_surface", "pick_and_place_grouped"}
    for path in sorted(Path(directory).glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or not any(isinstance(b, ast.Name) and b.id == "Task" for b in node.bases):
                continue
            fields = {}
            for statement in node.body:
                if isinstance(statement, ast.Assign):
                    fields.update({t.id: statement.value for t in statement.targets if isinstance(t, ast.Name)})
                elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                    fields[statement.target.id] = statement.value
            subtasks = fields.get("subtasks")
            calls = [n for n in ast.walk(subtasks) if isinstance(n, ast.Call)] if subtasks else []
            names = [n.func.id if isinstance(n.func, ast.Name) else ast.unparse(n.func) for n in calls]
            logicals = sorted({kw.value.value for call in calls for kw in call.keywords
                               if kw.arg == "logical" and isinstance(kw.value, ast.Constant)})
            standard = bool(names) and all(name in helpers for name in names)
            records.append({"task": node.name, "file": str(path), "line": node.lineno,
                            "subtask_groups": len(subtasks.elts) if isinstance(subtasks, (ast.List, ast.Tuple)) else None,
                            "subtask_calls": names, "explicit_logicals": logicals,
                            "status": "standard_helper_candidate" if standard else "override_review_required",
                            "semantics_verified": False, "generation_ready": False})
    return {"scope": "Static census only; success/subtask consistency requires runtime audit",
            "counts": dict(Counter(record["status"] for record in records)), "tasks": records}
