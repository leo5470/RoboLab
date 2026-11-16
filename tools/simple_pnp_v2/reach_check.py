#!/usr/bin/env python3
"""Franka reach gate for Simple-PnP v2 scenes (plan step 4, replaces grasp trials; no Isaac).

    tools/simple_pnp_v2/usd_python.sh tools/simple_pnp_v2/reach_check.py \
        --inputs output/simple_pnp_v2_validation/inputs/inputs.json \
        --handles output/simple_pnp_v2_validation/inputs/handle_records.json \
        --physics output/simple_pnp_v2_validation/physics/pilot3 output/simple_pnp_v2_validation/physics/pilot2 \
        --output output/simple_pnp_v2_validation/pilot3/reach.json

Decided 2026-11-03: OMY limits are ignored; the gate is that the DROID Franka
(franka_robotiq_2f_85_flattened.usd, DROID reset pose, base at the origin) can
reach each scene. Poses are the *settled* ones: the last hold-trace sample of
the nominal physics run (run0) for the exact scene hash; the authored pose is
used only when no matching run exists (recorded per scene).

Points, tool point 15 mm inside the fingertip, top-down (fingers -Z), yaw free,
exact IK (not the generation grid):
  grasp    handle entries: centroid of the handle point set (world);
           others: footprint centre at max(0.03, top - 0.03) m above the table
           (tool point never below 3 cm)
  pregrasp grasp + 0.10 m
  lift     grasp XY at max(grasp z + 0.15, 0.20) m
  release  bin footprint centre at 0.20 m, or 0.35 m for targets taller than 0.18 m
Kinematic only: no arm/table/bin collisions, no wrist-yaw or finger-width
constraint, no grasp trial. Report it as reachability evidence, not grasp evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]
from freeze_inputs import quat_matrix  # noqa: E402
from reach_map import build_solver  # noqa: E402

PARAMS = {"min_tool_z": 0.03, "top_inset": 0.03, "pregrasp_dz": 0.10, "lift_dz": 0.15, "lift_min_z": 0.20,
          "release_z": 0.20, "tall_release_z": 0.35, "tall_threshold": 0.18}


_INDEX = None


def settled_poses(dirs, scene, scene_sha, target, bin_name="grey_bin"):
    global _INDEX
    if _INDEX is None:  # (scene name, sha) -> first matching run0 report, in --physics order
        _INDEX = {}
        for d in dirs:
            for f in sorted(d.glob("*_run0.json")):
                row = json.loads(f.read_text())
                if row.get("trace"):
                    _INDEX.setdefault((Path(row.get("scene", "")).name, row.get("scene_sha256")),
                                      (f, {"status": row["status"], "trace": row["trace"][-1:]}))
    f, row = _INDEX.get((scene, scene_sha), (None, None))
    if row is None:
        return None, None, None
    last = row["trace"][-1]
    return np.array(last[target]["pose"]), np.array(last[bin_name]["pose"]), {
        "source": str(f.resolve().relative_to(ROOT)), "physics_status": row["status"], "t": last["t"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--handles", type=Path, required=True)
    parser.add_argument("--record-dir", type=Path, default=ROOT / "output/simple_pnp_v2_validation/generation")
    parser.add_argument("--physics", type=Path, nargs="*", default=[])
    parser.add_argument("--robot", type=Path, default=Path("assets/robots/franka_robotiq_2f_85_flattened.usd"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    entries = {e["catalog_index"]: e for e in inputs["entries"]}
    handles = json.loads(args.handles.read_text())["records"]
    top = inputs["table"]["top_z"]
    bg = inputs["bin"]["geometry"]
    reachable, meta = build_solver(args.robot)
    rows = []
    for rec_path in sorted(args.record_dir.glob("*.json")):
        rec = json.loads(rec_path.read_text())
        if rec.get("status") != "authored":
            continue
        entry = entries[rec["catalog_index"]]
        g = entry["geometry"]
        scene = ROOT / rec["scene"]
        scene_sha = hashlib.sha256(scene.read_bytes()).hexdigest()
        tpose, bpose, source = settled_poses(args.physics, scene.name, scene_sha, entry["prim_name"])
        if tpose is None:
            t = rec["target_pose"]; b = rec["bin_pose"]
            tpose = np.array([*t["translate"], *t["orient"]]); bpose = np.array([*b["translate"], *b["orient"]])
            source = {"source": "authored pose (no matching nominal physics run)"}
        R = quat_matrix(tpose[3:]) @ quat_matrix(g["root_rotation_wxyz"]).T
        Rb = quat_matrix(bpose[3:]) @ quat_matrix(bg["root_rotation_wxyz"]).T
        height = g["rest_size_m"][2]
        handle = handles.get(str(rec["catalog_index"]))
        if handle:
            grasp = (np.array(handle["points"]) @ R.T + tpose[:3]).mean(0)
            grasp[2] = max(grasp[2], top + PARAMS["min_tool_z"])
            basis = f"handle class {handle['class']} centroid"
        else:
            c = R @ np.array(g["rest_center_offset_m"]) + tpose[:3]
            grasp = np.array([c[0], c[1], top + max(PARAMS["min_tool_z"], height - PARAMS["top_inset"])])
            basis = "footprint centre, top - 0.03"
        bin_c = Rb @ np.array(bg["rest_center_offset_m"]) + bpose[:3]
        rz = PARAMS["tall_release_z"] if height > PARAMS["tall_threshold"] else PARAMS["release_z"]
        points = {"grasp": grasp, "pregrasp": grasp + [0, 0, PARAMS["pregrasp_dz"]],
                  "lift": np.array([grasp[0], grasp[1], max(grasp[2] + PARAMS["lift_dz"], top + PARAMS["lift_min_z"])]),
                  "release": np.array([bin_c[0], bin_c[1], top + rz])}
        ok = {k: bool(reachable(v)) for k, v in points.items()}
        failed = [k for k, v in ok.items() if not v]
        rows.append({"slot": rec["slot"], "scene": rec["scene"], "scene_sha256": scene_sha, "pose": source,
                     "grasp_basis": basis, "points": {k: v.round(4).tolist() for k, v in points.items()},
                     "reachable": ok, "status": "failed" if failed else "passed",
                     "reasons": [f"{k} {points[k].round(3).tolist()} unreachable" for k in failed]})
        print(f"{rec['slot']}: {rows[-1]['status']} {rows[-1]['reasons']}", flush=True)
    summary = {s: sum(r["status"] == s for r in rows) for s in ("passed", "failed")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"solver": meta, "params": PARAMS, "summary": summary,
                                       "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                       "results": rows}, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
