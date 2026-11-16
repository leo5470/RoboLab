#!/usr/bin/env python3
"""Estimate each asset's handle direction from its mesh outline (preliminary).

Run with pxr available, e.g.:
    tools/simple_pnp_v2/usd_python.sh tools/simple_pnp_v2/handle_directions.py \
        --catalog-index 230 231 303 --report output/simple_pnp_v2_validation/handle_directions.json

Points are taken in the asset's composed frame, including any rotation its root
authors (v2 scenes preserve that rotation and apply yaw on top). Mid-height
points (25-85 % of height) are recentred on the body. The handle direction is
the mean direction of points whose radius exceeds 1.15x the median of per-10°
maximum radii. ``lobe_ratio`` (max/median of those radii) near 1 means no clear
side handle. Directions are degrees, 0 = +X, counter-clockwise, Z up.
This is an estimate: confirm each direction with a render before freezing it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def mesh_points(stage):
    from pxr import Usd, UsdGeom
    pts = []
    for prim in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh):
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if points:
                M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0))
                pts.append((np.c_[np.array(points, float), np.ones(len(points))] @ M)[:, :3])
    return np.vstack(pts) if pts else None


def estimate(path):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(path))
    if UsdGeom.GetStageUpAxis(stage) != "Z":
        return {"error": f"stage up axis {UsdGeom.GetStageUpAxis(stage)} is not Z"}
    X = mesh_points(stage)
    if X is None:
        return {"error": "no mesh points"}
    h = X[:, 2]; lo, hi = h.min(), h.max()
    band = X[(h > lo + .25 * (hi - lo)) & (h < lo + .85 * (hi - lo))][:, :2]
    c = np.median(band, 0)
    for _ in range(3):
        r = np.linalg.norm(band - c, axis=1); c = band[r < np.percentile(r, 70)].mean(0)
    d = band - c; r = np.linalg.norm(d, axis=1); az = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 360
    bins = np.array([r[(az >= b) & (az < b + 10)].max() if ((az >= b) & (az < b + 10)).any() else 0 for b in range(0, 360, 10)])
    sel = r > np.median(bins) * 1.15
    direction = float(np.degrees(np.arctan2(d[sel, 1].mean(), d[sel, 0].mean())) % 360) if sel.any() else None
    return {"bounds_min": X.min(0).round(4).tolist(), "bounds_max": X.max(0).round(4).tolist(),
            "body_center_xy": c.round(4).tolist(), "lobe_ratio": round(float(bins.max() / np.median(bins)), 3),
            "handle_direction_deg": None if direction is None else round(direction, 1),
            "peak_bin_deg": int(bins.argmax() * 10 + 5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog-index", type=int, nargs="*", default=[])
    parser.add_argument("--asset", type=Path, nargs="*", default=[], help="USD paths relative to the repo root")
    parser.add_argument("--report", type=Path, help="Write JSON results here")
    args = parser.parse_args()
    catalog = json.loads((ROOT / "assets/objects/object_catalog.json").read_text())
    items = catalog if isinstance(catalog, list) else catalog["objects"]
    targets = [(i, items[i]["name"], Path(items[i]["usd_path"])) for i in args.catalog_index]
    targets += [(None, p.stem, p) for p in args.asset]
    if not targets:
        parser.error("give --catalog-index and/or --asset")
    rows = []
    for index, name, rel in targets:
        path = ROOT / rel
        row = {"catalog_index": index, "name": name, "usd_path": str(rel),
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **estimate(path)}
        rows.append(row)
        print(f"{str(index):>4} {name:28s} dir={row.get('handle_direction_deg')} lobe={row.get('lobe_ratio')} {row.get('error', '')}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"method": "mesh-outline lobe estimate (preliminary)", "rows": rows}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
