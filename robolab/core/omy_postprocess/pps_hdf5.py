"""Export a published OMY pool as one PPS single-file HDF5, in the pi05 DROID wire convention.

The file follows the PPS "OMY HDF5 export specification": every row holds what the
RoboLab pi05 DROID client sends (``Pi0DroidJointposClient._pack_request``) and every
action what the server returns, before the client's own gripper threshold.

- Numerical rows come from the shared row mapper (``export.iter_pps_rows``) over the
  pool's raw recordings. Arm joints and the 0..1 gripper closure are the recorded
  outputs of the same ``robolab.robots.droid`` observation terms the client reads; the
  only conversion is the gripper command, +1 open/-1 closed -> 0 open/1 closed, which
  both action terms execute as finger_joint 0 / pi/4.
- Camera frames come from the verified training file (the rollout cameras at 720x1280)
  and pass through the client's own ``openpi_client.image_tools.resize_with_pad``.

The file is written with ``training_ready=False``; acceptance evidence is a separate step.
"""

import json
import math
import os
import re
from pathlib import Path
import subprocess
import tempfile
import time

import h5py
import numpy as np

from .export import ARM_JOINT_NAMES, FPS, PPSRecipe, iter_pps_rows
from .io import fsync_directory, publish_directory, sha256_file

FORMAT = "pps_omy_hdf5_v1"
WIRE_CONVENTION = "pi05_droid"
FILE_NAME = "pps.hdf5"
IMAGE_SIZE = (224, 224)
CAMERAS = {"exterior_image_1_left": "over_shoulder_left_camera", "wrist_image_left": "wrist_cam"}
RECIPE = PPSRecipe.from_dict({
    "schema_version": 1, "fps": FPS, "arm_joint_names": list(ARM_JOINT_NAMES),
    "measured_gripper": {"source": "normalized_closure", "units": "normalized_closure"},
    "gripper_command": {"encoding": "zero_one_closed"},
})
# MimicGen plans its commands; no leader device is read, so every generated row lacks
# fresh leader input by construction. Accepted only when it covers every row.
GENERATED_WARNING = "invalid_leader_samples"
GENERATED_REVIEW_REASON = ("MimicGen commands are planned, not read from the OMY leader device; every row "
                           "lacks leader input by construction")
# The only command column that is constant by design (accepted by PPS with this evidence);
# any other constant column is an export bug.
HELD_JOINT = {"joint": "panda_joint3", "column": 2, "hold_value_rad": 0.0, "hold_reference": "DROID home",
              "reason": "the OMY-L100 leader has six axes; teleoperation and MimicGen hold panda_joint3 at the "
                        "DROID home value"}
ALLOWED_CONSTANT = {"actions": [2], "original_actions": [2]}
DATASETS = {  # name: (dtype, trailing shape)
    "joint_position": (np.float32, (7,)), "gripper_position": (np.float32, (1,)), "actions": (np.float32, (8,)),
    "original_actions": (np.float32, (8,)), "raw_finger_position": (np.float32, (1,)),
    "normalized_gripper_position": (np.float32, (1,)), "source_step": (np.int64, ()),
    "timestamp": (np.float64, ()), "source_sim_before_s": (np.float64, ()), "source_sim_origin_s": (np.float64, ()),
}
FRAME_BATCH = 32


def revision():
    root = Path(__file__).resolve().parents[3]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                           capture_output=True, text=True).stdout.strip()
    return {"robolab_revision": head, "robolab_dirty": bool(dirty)}


def family_task(layout_tasks):
    """Stable file-level task: the layout task IDs without their two-digit layout suffix."""
    families = {re.fullmatch(r"(.+?)\d{2}Task", t).group(1) if re.fullmatch(r"(.+?)\d{2}Task", t) else None
                for t in layout_tasks}
    if len(families) != 1 or None in families:
        raise ValueError(f"Layout tasks do not share one <family><NN>Task name: {sorted(layout_tasks)}")
    return families.pop()


def family_id(row):
    """One ID per human source demo, shared by every episode generated from it."""
    if row["kind"] == "generated":
        part, episode = Path(row["parent_source"]), row["parent_episode"]
    else:
        part, episode = Path(row["source_file"]).parent, row["episode"]
    return f"{row['session']}/{part.parent.name}/{part.name}:{episode}"


def review_for(row):
    if row["kind"] != "generated":
        return None
    codes = {w["code"] for w in row.get("warnings", [])}
    if codes - {GENERATED_WARNING}:
        raise ValueError(f"{row['fingerprint']}: unexpected warnings {sorted(codes)}")
    for warning in row.get("warnings", []):
        if sorted(warning.get("indices", [])) != list(range(row["num_samples"])):
            raise ValueError(f"{row['fingerprint']}: {GENERATED_WARNING} does not cover every generated row")
    return {"source_sha256": row["sha256"], "source_group": f"data/{row['episode']}",
            "fingerprint": row["fingerprint"], "reviewer": "omy_dataset_pipeline",
            "reason": GENERATED_REVIEW_REASON, "accepted_warning_codes": sorted(codes)}


def numerical_rows(pool_dir, row):
    rows = list(iter_pps_rows(pool_dir / row["file"], f"data/{row['episode']}", recipe=RECIPE,
                              review=review_for(row)))
    columns = {name: np.asarray([r[name] for r in rows], dtype=dtype).reshape(len(rows), *shape)
               for name, (dtype, shape) in DATASETS.items() if name != "gripper_position"}
    normalized = columns["normalized_gripper_position"]
    # The client sends the DROID observation term, clipped to [0, 1] by its ObsTerm.
    columns["gripper_position"] = np.clip(normalized, 0.0, 1.0).astype(np.float32)
    check_round_trip(columns)
    return columns


def check_round_trip(c):
    """§5.3: original_actions/raw_finger_position reproduce actions/gripper_position."""
    original, actions = c["original_actions"], c["actions"]
    if not np.isin(original[:, 7], (-1.0, 1.0)).all():
        raise ValueError("Recorded gripper command is not exactly +1/-1")
    if not (np.array_equal(actions[:, :7], original[:, :7])
            and np.array_equal(actions[:, 7], ((1.0 - original[:, 7]) / 2.0).astype(np.float32))):
        raise ValueError("Wire actions do not round-trip to the recorded commands")
    if np.abs(c["normalized_gripper_position"] - c["raw_finger_position"] / (math.pi / 4)).max() > 1e-5:
        raise ValueError("Gripper closure does not equal finger_joint / (pi/4)")
    n = len(actions)
    if not (np.array_equal(c["source_step"], np.arange(n)) and np.array_equal(c["timestamp"], np.arange(n) / FPS)):
        raise ValueError("Rows are not the native 15 Hz pre-action grid")


def write_episode(group, columns, cameras, row):
    n = len(columns["actions"])
    for name, values in columns.items():
        group.create_dataset(name, data=values)
    for wire, recorded in CAMERAS.items():
        source = cameras[recorded]
        if source.shape[0] != n or source.dtype != np.uint8 or source.shape[-1] != 3:
            raise ValueError(f"Camera {recorded} does not match the {n} numerical rows")
        target = group.create_dataset(wire, shape=(n, *IMAGE_SIZE, 3), dtype=np.uint8, chunks=(1, *IMAGE_SIZE, 3))
        for start in range(0, n, FRAME_BATCH):
            target[start:start + FRAME_BATCH] = resize(source[start:start + FRAME_BATCH])
    group.attrs.update(success=True, source_family=family_id(row), num_samples=n, layout_task=row["task"],
                       kind=row["kind"], fingerprint=row["fingerprint"])


def resize(frames):
    from openpi_client import image_tools  # the rollout client's own preprocessing

    return image_tools.resize_with_pad(np.asarray(frames), *IMAGE_SIZE)


def export_session(pool_dir, training_dir, output_root):
    """Write OUTPUT_ROOT/SESSION/{pps.hdf5, manifest.json, validation.json}; return the directory."""
    pool_dir, training_dir, output_root = Path(pool_dir), Path(training_dir), Path(output_root)
    pool = json.loads((pool_dir / "dataset.json").read_text())
    training = json.loads((training_dir / "dataset.json").read_text())
    groups = {r["fingerprint"]: r["group"] for r in training["episodes"]}
    if sorted(groups) != sorted(r["fingerprint"] for r in pool["episodes"]):
        raise ValueError("Training file does not hold the pool's episodes")
    target = output_root / pool["session"]
    if target.exists():
        raise FileExistsError(f"Refusing existing output: {target}")
    output_root.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{pool['session']}.partial-", dir=output_root))
    episodes = []
    with h5py.File(training_dir / "training.hdf5", "r") as source, h5py.File(partial / FILE_NAME, "x") as out:
        instructions = {str(source[groups[r["fingerprint"]]].attrs["task_instruction"]) for r in pool["episodes"]}
        if len(instructions) != 1:
            raise ValueError(f"One prompt per file is required; found {sorted(instructions)}")
        prompt = instructions.pop()
        task = family_task({r["task"] for r in pool["episodes"]})
        out.attrs.update(fps=FPS, task=task, prompt=prompt, training_ready=False, format=FORMAT,
                         wire_convention=WIRE_CONVENTION, session=pool["session"])
        data = out.create_group("data")
        for index, row in enumerate(pool["episodes"]):
            recorded = source[groups[row["fingerprint"]]]
            columns = numerical_rows(pool_dir, row)
            if not (np.array_equal(columns["joint_position"], recorded["obs/arm_joint_pos"][()])
                    and np.array_equal(columns["original_actions"], recorded["actions"][()])):
                raise ValueError(f"{row['fingerprint']}: training file and pool recording disagree")
            name = f"episode_{index:03d}"
            write_episode(data.create_group(name), columns, recorded["obs/images"], row)
            episodes.append({"group": f"data/{name}", "kind": row["kind"], "layout_task": row["task"],
                             "source_family": family_id(row), "fingerprint": row["fingerprint"],
                             "num_samples": len(columns["actions"])})
            print(f"{pool['session']}: PPS episode {index + 1}/{len(pool['episodes'])}", flush=True)
    validation = verify_file(partial / FILE_NAME)
    (partial / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "format": FORMAT, "wire_convention": WIRE_CONVENTION, "file": FILE_NAME,
        "sha256": sha256_file(partial / FILE_NAME), "training_ready": False, "session": pool["session"],
        "family": pool["family"], "task": task, "prompt": prompt,
        "pool_manifest_sha256": sha256_file(pool_dir / "dataset.json"),
        "training_manifest_sha256": sha256_file(training_dir / "dataset.json"),
        "training_file_sha256": training["sha256"], **revision(), "created_unix_s": time.time(),
        "selection_rule": pool.get("method"), "episodes": episodes, "conventions": CONVENTIONS,
        "held_joint": validation["held_joint"],
    }
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in partial.iterdir():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    fsync_directory(partial)
    publish_directory(partial, target)
    return target


CONVENTIONS = {
    "client": "policies/pi0_family/client.py:Pi0DroidJointposClient",
    "joint_position": "measured panda_joint1..7 (rad), recorded robolab.robots.droid.arm_joint_pos; identity",
    "gripper_position": "recorded robolab.robots.droid.gripper_pos (finger_joint / (pi/4)), clipped to [0, 1] "
                        "like its ObsTerm; unclipped value in normalized_gripper_position; recorded without the "
                        "rollout's observation noise",
    "actions": "absolute panda_joint1..7 targets (rad), identity; gripper (1 - omy) / 2: 0 open, 1 closed",
    "original_actions": "recorded OMY command: joint targets, gripper +1 open / -1 closed",
    "images": "rollout cameras over_shoulder_left_camera / wrist_cam (720x1280 RGB) through "
              "openpi_client.image_tools.resize_with_pad(224, 224)",
    "timing": "native 15 Hz, pre-action rows, no terminal observation, no label shift",
    "task": "file task is the family name (layout task IDs without their NN suffix); each episode's "
            "layout_task holds its RoboLab task ID",
    "generated_review": {"warning": GENERATED_WARNING, "reason": GENERATED_REVIEW_REASON},
}


def verify_file(path):
    """Spec-level checks of a written file; returns per-channel ranges (§5.4 input)."""
    ranges = {}
    with h5py.File(path, "r") as f:
        if f.attrs["fps"] != FPS or not f.attrs["task"] or not f.attrs["prompt"] or bool(f.attrs["training_ready"]):
            raise ValueError("File attributes fps/task/prompt/training_ready are invalid")
        names = sorted(f["data"])
        if not names:
            raise ValueError("No episodes")
        for name in names:
            group = f["data"][name]
            n = group["actions"].shape[0]
            if n < 1 or int(group.attrs["num_samples"]) != n or not group.attrs["source_family"]:
                raise ValueError(f"{name}: invalid num_samples or source_family")
            for dataset, (dtype, shape) in DATASETS.items():
                values = group[dataset][()]
                if values.dtype != dtype or values.shape != (n, *shape) or not np.isfinite(values).all():
                    raise ValueError(f"{name}/{dataset}: expected finite {np.dtype(dtype)} {(n, *shape)}")
                if values.ndim == 2:
                    low, high = ranges.setdefault(dataset, [np.full(shape, np.inf), np.full(shape, -np.inf)])
                    np.minimum(low, values.min(0), out=low)
                    np.maximum(high, values.max(0), out=high)
            if not np.all(group["actions"][:, HELD_JOINT["column"]] == HELD_JOINT["hold_value_rad"]):
                raise ValueError(f"{name}: {HELD_JOINT['joint']} command is not held at {HELD_JOINT['hold_value_rad']}")
            for wire in CAMERAS:
                image = group[wire]
                if image.dtype != np.uint8 or image.shape != (n, *IMAGE_SIZE, 3):
                    raise ValueError(f"{name}/{wire}: expected uint8 {(n, *IMAGE_SIZE, 3)}")
            if not (np.array_equal(group["source_step"][()], np.arange(n))
                    and np.array_equal(group["timestamp"][()], np.arange(n) / FPS)):
                raise ValueError(f"{name}: source_step/timestamp off the 15 Hz grid")
    ranges = {k: {"min": v[0].tolist(), "max": v[1].tolist()} for k, v in ranges.items()}
    constant = {k: [i for i, (a, b) in enumerate(zip(v["min"], v["max"])) if a == b] for k, v in ranges.items()}
    if any(columns != ALLOWED_CONSTANT.get(k, []) for k, columns in constant.items() if columns) \
            or constant.get("actions") != ALLOWED_CONSTANT["actions"]:
        raise ValueError(f"Unexpected constant channels (export bug): {constant}")
    column = HELD_JOINT["column"]
    measured = [ranges["joint_position"]["min"][column], ranges["joint_position"]["max"][column]]
    return {"episodes": len(names), "ranges": ranges, "constant_channels": constant,
            "held_joint": {**HELD_JOINT, "measured_range_rad": measured}}


def verify_session(directory, pool_dir, training_dir):
    """Cheap check that a published export still matches its pool and training file."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = {"pool_manifest_sha256": sha256_file(Path(pool_dir) / "dataset.json"),
                "training_manifest_sha256": sha256_file(Path(training_dir) / "dataset.json"),
                "sha256": sha256_file(directory / FILE_NAME)}
    if any(manifest.get(k) != v for k, v in expected.items()) or manifest.get("format") != FORMAT:
        raise ValueError(f"PPS export {directory} no longer matches its pool or file")
    return manifest
