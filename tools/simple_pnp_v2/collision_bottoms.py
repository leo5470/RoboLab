#!/usr/bin/env python3
"""Compare each asset's collision-shape bottom with its visual bottom.

Run with pxr available, e.g.:
    tools/simple_pnp_v2/usd_python.sh tools/simple_pnp_v2/collision_bottoms.py \
        --retained --report output/simple_pnp_v2_validation/collision_bottoms.json

A contact start (clearance 0) is safe from asset geometry only if no authored
collision shape extends below the visual bottom (``delta_mm`` >= 0). Hulls that
PhysX cooks at load time (``convexDecomposition``) are not represented in the
authored file; measure start-up motion in simulation as well. Also reports
any root rotation the payload authors, which scenes must preserve.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROTATION_OPS = ("rotateX", "rotateY", "rotateZ", "rotateXYZ", "rotateXZY", "rotateYXZ",
                "rotateYZX", "rotateZXY", "rotateZYX", "orient", "transform")


def inspect(path):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(path)); root = stage.GetDefaultPrim()
    visual = UsdGeom.BBoxCache(0, ["default", "render", "proxy"], useExtentsHint=False).ComputeWorldBound(root).ComputeAlignedRange()
    guide = UsdGeom.BBoxCache(0, ["default", "render", "proxy", "guide"], useExtentsHint=False)
    bottoms, approximations, offsets = [], set(), set()
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        attr = prim.GetAttribute("physics:approximation")
        approximations.add(str(attr.Get()) if attr and attr.Get() else ("none" if prim.IsA(UsdGeom.Mesh) else prim.GetTypeName()))
        for name in ("physxCollision:contactOffset", "physxCollision:restOffset"):
            a = prim.GetAttribute(name)
            if a and a.Get() is not None:
                offsets.add(f"{name.split(':')[1]}={float(a.Get())}")
        if prim.IsA(UsdGeom.Mesh):
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if points:
                M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0))
                bottoms.append(float((np.c_[np.array(points, float), np.ones(len(points))] @ M)[:, 2].min()))
        elif prim.IsA(UsdGeom.Gprim):
            bottoms.append(float(guide.ComputeWorldBound(prim).ComputeAlignedRange().GetMin()[2]))
    rotation = Gf.Matrix4d(1); rotation_ops = []
    for op in UsdGeom.Xformable(root).GetOrderedXformOps():
        if op.GetOpName().split(":")[1] in ROTATION_OPS:
            rotation_ops.append(op.GetOpName()); rotation = op.GetOpTransform(0) * rotation
    up = rotation.TransformDir(Gf.Vec3d(0, 0, 1))
    collision_bottom = min(bottoms) if bottoms else None
    return {"visual_bottom_m": round(visual.GetMin()[2], 5),
            "collision_bottom_m": None if collision_bottom is None else round(collision_bottom, 5),
            "delta_mm": None if collision_bottom is None else round((collision_bottom - visual.GetMin()[2]) * 1000, 3),
            "approximations": sorted(approximations), "offsets": sorted(offsets),
            "root_rotation_ops": rotation_ops, "root_rotation_deg": round(rotation.ExtractRotation().GetAngle(), 3),
            "root_rotation_tilts_up_axis": abs(up[2] - 1) > 1e-6}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retained", action="store_true", help="All retained v1 entries plus the grey bin")
    parser.add_argument("--asset", type=Path, nargs="*", default=[], help="USD paths relative to the repo root")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    paths = list(args.asset)
    if args.retained:
        catalog = json.loads((ROOT / "assets/objects/object_catalog.json").read_text())
        items = catalog if isinstance(catalog, list) else catalog["objects"]
        manifest = json.loads((ROOT / "assets/scenes/simple_pnp/manifest.json").read_text())
        paths += [Path(items[i]["usd_path"]) for i in sorted({s["catalog_index"] for s in manifest["scenes"]})]
        paths.append(Path(manifest["destination"]["usd_path"]))
    if not paths:
        parser.error("give --retained and/or --asset")
    rows = [{"usd_path": str(p), **inspect(ROOT / p)} for p in paths]
    deltas = [r["delta_mm"] for r in rows if r["delta_mm"] is not None]
    below = [r["usd_path"] for r in rows if r["delta_mm"] is not None and r["delta_mm"] < 0]
    tilted = [r["usd_path"] for r in rows if r["root_rotation_tilts_up_axis"]]
    print(f"{len(rows)} assets; collision minus visual bottom: min {min(deltas):.3f} mm, max {max(deltas):.3f} mm")
    print(f"collision below visual: {below or 'none'}")
    print(f"non-identity root rotations: {[(r['usd_path'], r['root_rotation_deg']) for r in rows if r['root_rotation_deg'] > 1e-3] or 'none'}")
    print(f"root rotations that tilt the up axis: {tilted or 'none'}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"rows": rows}, indent=1))
    return 1 if below else 0


if __name__ == "__main__":
    sys.exit(main())
