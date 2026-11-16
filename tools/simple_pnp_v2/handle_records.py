#!/usr/bin/env python3
"""Build handle records for the handle-visibility entries from confirmed regions.

    tools/simple_pnp_v2/usd_python.sh tools/simple_pnp_v2/handle_records.py \
        --inputs output/simple_pnp_v2_validation/inputs/inputs.json \
        --output output/simple_pnp_v2_validation/inputs/handle_records.json

Points are in the *rest frame* v2 places with: the payload's children, then
its preserved root rotation and intended scale, root at the origin, yaw 0
(world = T * R(yaw) * rest). Directions: degrees, 0 = +X, CCW, Z up.

Classes, directions, and handle regions come from handle_regions.json, which
was confirmed from three-view point plots (plot_handles.py). The direction
from the body centre to the region's points is recomputed as a cross-check and
must agree with the confirmed direction within 30 degrees.
- Class A (side handle): points inside the confirmed region.
- Class B (top handle): the top fraction of height.
- Class C (utensil lying flat): all points; occlusion-only screen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REGIONS = Path(__file__).resolve().parent / "handle_regions.json"
MAX_POINTS = 200


def rest_points(path, root_rotation, scale, normals=False):
    """Mesh points (and optionally area-weighted vertex normals) in the rest frame."""
    from pxr import Usd, UsdGeom
    from freeze_inputs import quat_matrix
    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    to_root = np.array(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(0).GetInverse())
    R = quat_matrix(root_rotation)
    pts, nrm = [], []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        P = np.array(points, float)
        M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)) @ to_root
        pts.append(((np.c_[P, np.ones(len(P))] @ M)[:, :3] * np.asarray(scale)) @ R.T)
        if normals:
            counts = np.array(mesh.GetFaceVertexCountsAttr().Get() or [], int)
            index = np.array(mesh.GetFaceVertexIndicesAttr().Get() or [], int)
            N = np.zeros_like(P)
            start = 0
            for c in counts:  # fan triangulation, area-weighted face normals
                face = index[start:start + c]; start += c
                for i in range(1, c - 1):
                    a_, b_, c_ = P[face[0]], P[face[i]], P[face[i + 1]]
                    n = np.cross(b_ - a_, c_ - a_)
                    N[face[[0, i, i + 1]]] += n
            if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
                N = -N
            L = M[:3, :3]  # row-vector convention: n' = n @ inv(L).T
            N = N @ np.linalg.inv(L).T
            N = (N / np.asarray(scale)) @ R.T
            N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
            nrm.append(N)
    X = np.vstack(pts)
    return (X, np.vstack(nrm)) if normals else X


def subsample_index(n, rng):
    return np.arange(n) if n <= MAX_POINTS else np.sort(rng.choice(n, MAX_POINTS, replace=False))


def in_region(X, region):
    keep = np.ones(len(X), bool)
    for axis, i in (("x", 0), ("y", 1), ("z", 2)):
        if axis in region:
            lo, hi = region[axis]
            keep &= (X[:, i] >= (-np.inf if lo is None else lo)) & (X[:, i] <= (np.inf if hi is None else hi))
    if "xy_along_deg" in region:
        deg, lo, hi = region["xy_along_deg"]
        s = X[:, 0] * np.cos(np.radians(deg)) + X[:, 1] * np.sin(np.radians(deg))
        keep &= (s >= (-np.inf if lo is None else lo)) & (s <= (np.inf if hi is None else hi))
    if "top_fraction" in region:
        h = X[:, 2]
        keep &= h >= h.max() - region["top_fraction"] * (h.max() - h.min())
    return keep


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    entries = {e["catalog_index"]: e for e in inputs["entries"]}
    config = json.loads(REGIONS.read_text())
    spec = {int(k): v for k, v in config["regions"].items()}
    for k in config["class_c"]["indices"]:
        spec[k] = {"class": "C", "region": {}, "note": config["class_c"]["note"]}
    rng = np.random.default_rng(0)
    records, problems = {}, []
    for index, item in sorted(spec.items()):
        if index not in entries:
            problems.append(f"{index}: not a retained entry")
            continue
        e = entries[index]
        X, N = rest_points(ROOT / e["payload"], e["geometry"]["root_rotation_wxyz"], e["scale"], normals=True)
        keep = in_region(X, item["region"])
        # Winding check: share of vertex normals pointing away from the asset centre.
        centre = (X.min(0) + X.max(0)) / 2
        outward = float(np.mean(np.einsum("ij,ij->i", N, X - centre) > 0))
        if keep.sum() < 10:
            problems.append(f"{index} {e['prim_name']}: only {keep.sum()} points in the handle region")
            continue
        row = {"catalog_index": index, "prim_name": e["prim_name"], "class": item["class"], "payload": e["payload"],
               "payload_sha256": e["payload_sha256"], "region": item["region"], "note": item.get("note"),
               "confirmed": "three-view point plots, 2026-11-02 (render-based visibility gate still required)",
               "regions_sha256": hashlib.sha256(REGIONS.read_bytes()).hexdigest()}
        if item["class"] == "A":
            body = X[~keep]
            center = (body[:, :2].min(0) + body[:, :2].max(0)) / 2
            v = X[keep, :2].mean(0) - center
            measured = float(np.degrees(np.arctan2(v[1], v[0])) % 360)
            diff = abs((measured - item["direction_deg"] + 180) % 360 - 180)
            row.update(handle_direction_local_deg=float(item["direction_deg"]), region_centroid_direction_deg=measured,
                       body_center_xy=center.tolist())
            if diff > 30:
                problems.append(f"{index} {e['prim_name']}: region centroid direction {measured:.0f} vs confirmed "
                                f"{item['direction_deg']} deg")
        pick = subsample_index(int(keep.sum()), rng)
        row["points"] = np.round(X[keep][pick], 5).tolist()
        row["normals"] = np.round(N[keep][pick], 4).tolist()
        row["outward_normal_fraction_all_points"] = outward
        records[str(index)] = row
        print(f"{item['class']} {index:>3} {e['prim_name']:24s} dir={row.get('handle_direction_local_deg')} "
              f"centroid={row.get('region_centroid_direction_deg') and round(row['region_centroid_direction_deg'])} "
              f"region_points={int(keep.sum())} outward={outward:.2f}")
    if problems:
        print("\n".join(problems))
    out = {"schema_version": 1, "inputs_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
           "regions_sha256": hashlib.sha256(REGIONS.read_bytes()).hexdigest(),
           "records": records, "problems": problems}
    args.output.write_text(json.dumps(out, indent=1))
    print("wrote", args.output, hashlib.sha256(args.output.read_bytes()).hexdigest())
    return 1 if problems else 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
