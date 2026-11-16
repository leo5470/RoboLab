"""Grasp library: GraspDataGen Isaac Grasp YAML → DROID ``base_link`` poses in the RoboLab object frame.

GraspDataGen writes, per validated grasp:

* ``pregrasp_position`` / ``pregrasp_orientation``: the open-gripper pose it started from (the pose to command);
* ``position`` / ``orientation``: the gripper pose relative to the object after closing and the shake test
  (how the object ends up sitting in the hand);
* ``cspace_position.finger_joint``: the finger angle at the end of the test;
* ``confidence``: 1.0 if the object was still held.

All poses are of GraspDataGen's ``base_link`` in the object frame. ``build`` maps them to DROID's ``base_link``
(D5) and appends the 180° twin of each grasp.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import yaml

from robolab.core.scripted_demos.grasps import frames
from robolab.core.scripted_demos.records import file_sha256


def _pose(entry, position_key: str, orientation_key: str) -> np.ndarray:
    o = entry[orientation_key]
    return frames.pose(entry[position_key], [o["w"], *o["xyz"]])


def parse_isaac_grasp_yaml(path: pathlib.Path) -> dict:
    """Read every grasp of one GraspDataGen output file (successes and failures, if it stored failures)."""
    data = yaml.safe_load(pathlib.Path(path).read_text())
    if data.get("format") != "isaac_grasp":
        raise ValueError(f"{path}: not an isaac_grasp file")
    names, guess, held, confidence, finger = [], [], [], [], []
    for name, entry in (data.get("grasps") or {}).items():
        names.append(name)
        has_pregrasp = "pregrasp_position" in entry
        guess.append(_pose(entry, *(("pregrasp_position", "pregrasp_orientation") if has_pregrasp else ("position", "orientation"))))
        held.append(_pose(entry, "position", "orientation"))
        confidence.append(float(entry.get("confidence", 0.0)))
        finger.append(float(entry.get("cspace_position", {}).get("finger_joint", np.nan)))
    header = {k: v for k, v in data.items() if k != "grasps"}
    return {
        "header": header,
        "names": np.asarray(names),
        "T_obj_gdg_guess": np.asarray(guess).reshape(-1, 4, 4),
        "T_obj_gdg_held": np.asarray(held).reshape(-1, 4, 4),
        "confidence": np.asarray(confidence, dtype=np.float64),
        "finger_joint": np.asarray(finger, dtype=np.float64),
    }


def build(parsed: dict, T_gdg_droid: np.ndarray, *, also_passed: dict[str, np.ndarray] | None = None) -> dict:
    """Library arrays for the grasps with ``confidence > 0``, plus their 180° twins.

    ``also_passed`` maps a label to a per-grasp boolean array from another validation run of the same guesses
    (e.g. the default 1 g tug); it is stored as ``passed_<label>``.
    """
    keep = parsed["confidence"] > 0
    idx = np.nonzero(keep)[0]
    grasp = parsed["T_obj_gdg_guess"][idx] @ T_gdg_droid
    held = parsed["T_obj_gdg_held"][idx] @ T_gdg_droid
    flip = frames.homogeneous(frames.rot(frames.APPROACH_AXIS, np.pi))
    out = {
        "T_obj_grasp": np.concatenate([grasp, grasp @ flip]).astype(np.float64),
        # The held pose of the twin is the held pose of the original turned the same way.
        "T_obj_held": np.concatenate([held, held @ flip]).astype(np.float64),
        "confidence": np.tile(parsed["confidence"][idx], 2),
        "finger_joint": np.tile(parsed["finger_joint"][idx], 2),
        "source_index": np.tile(idx, 2).astype(np.int64),
        "source_name": np.tile(parsed["names"][idx], 2),
        "twin": np.repeat([0, 1], len(idx)).astype(np.int8),
    }
    for label, passed in (also_passed or {}).items():
        out[f"passed_{label}"] = np.tile(np.asarray(passed)[idx], 2).astype(bool)
    return out


def save(path: pathlib.Path, library: dict) -> str:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **library)
    return file_sha256(path)


def load(path: pathlib.Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def write_manifest(path: pathlib.Path, manifest: dict) -> None:
    pathlib.Path(path).write_text(json.dumps(manifest, indent=1, default=str))


# ---- grasp classes -------------------------------------------------------------------------------------------------
# handle / rim / body, from where the pad centre sits on the object.
#
# The body is the cross-section of the lowest part of the object (10-30 % of its height), which a side handle does not
# reach: its centre is the midpoint of that slice's extent and its radius the slice's 95th-percentile radius. The handle
# direction is the mean direction of mid-height points (25-85 %) beyond 1.2x that radius. (RoboLab's
# ``tools/simple_pnp_v2/handle_directions.py`` re-centres on the densest mid-height points instead; for the YCB pitcher
# its densely meshed handle drags that centre onto itself and the estimate, recorded as ≈222° in
# ``docs/simple_pnp_v2_plan.md:114``, points at the spout. The handle is at ≈45° in the object frame.)
#
# A grasp is a *handle* grasp when its pad centre lies outside the body (beyond 1.15x the body radius) within
# ``HANDLE_HALF_WIDTH_DEG`` of the handle direction, so a pitcher's spout does not count; *rim* when it sits on the top
# edge; *body* otherwise. 0° = +X, counter-clockwise, in the object frame.

SIDE_HANDLED = {301, 303}  # objects in the batch with a side handle
HANDLE_HALF_WIDTH_DEG = 45.0


def body_frame(vertices: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(body centre xy, body radius, top height)."""
    z = vertices[:, 2]
    lo, hi = z.min(), z.max()
    low = vertices[(z > lo + 0.10 * (hi - lo)) & (z < lo + 0.30 * (hi - lo))][:, :2]
    if len(low) == 0:  # coarse mesh (e.g. a box with vertices only at its faces): use the whole outline
        low = vertices[:, :2]
    c = (low.min(0) + low.max(0)) / 2
    return c, float(np.percentile(np.linalg.norm(low - c, axis=1), 95)), float(hi)


def handle_direction(vertices: np.ndarray) -> float | None:
    """Handle azimuth (deg) from the body centre, or None when nothing sticks out at mid height."""
    c, r, _ = body_frame(vertices)
    z = vertices[:, 2]
    lo, hi = z.min(), z.max()
    mid = vertices[(z > lo + 0.25 * (hi - lo)) & (z < lo + 0.85 * (hi - lo))][:, :2] - c
    out = mid[np.linalg.norm(mid, axis=1) > 1.2 * r]
    if len(out) == 0:
        return None
    return float(np.degrees(np.arctan2(out[:, 1].mean(), out[:, 0].mean())) % 360)


def classify(T_obj_grasp: np.ndarray, vertices: np.ndarray, catalog_index: int | None = None) -> np.ndarray:
    """Per grasp: "handle", "rim" or "body"."""
    c, body_r, top = body_frame(vertices)
    tcp = np.einsum("nij,j->ni", T_obj_grasp[:, :3, :3], frames.TCP) + T_obj_grasp[:, :3, 3]
    d = tcp[:, :2] - c
    radial = np.linalg.norm(d, axis=1)
    out = np.full(len(tcp), "body", dtype=object)
    out[(tcp[:, 2] > top - 0.035) & (radial <= 1.15 * body_r)] = "rim"
    outside = radial > 1.15 * body_r
    direction = handle_direction(vertices) if catalog_index in SIDE_HANDLED else None
    if direction is not None:
        az = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360
        off = np.abs((az - direction + 180) % 360 - 180)
        out[outside & (off <= HANDLE_HALF_WIDTH_DEG)] = "handle"
        out[outside & (off > HANDLE_HALF_WIDTH_DEG)] = "body"  # e.g. a pitcher's spout
    else:
        out[outside] = "handle"
    return out
