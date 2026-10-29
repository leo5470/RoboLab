#!/usr/bin/env python3
"""Audit Simple-PnP libraries or a single flat tabletop scene without physics.

Requires numpy and pxr (USD). Exit 1 means layouts need repair/validation; this
never modifies scenes. Passing static checks does not certify graspability.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robolab.core.scenes.placement import flat_layout_issues, footprint_separation


def world_corners(cache, prim):
    from pxr import Gf, Usd, UsdGeom

    if not prim:
        raise ValueError("Missing scene prim")
    bounds = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
    if bounds.IsEmpty():
        raise ValueError(f"Missing geometry: {prim.GetPath()}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array([matrix.Transform(Gf.Vec3d(*point)) for point in
                     product(*zip(bounds.GetMin(), bounds.GetMax()))])


def audit_flat_scene(path, objects, *, support="table", pair_clearance=0.02, edge_clearance=0.01):
    """Check separate objects on an axis-aligned rectangular support.

    This deliberately does not certify stacked or contained arrangements.
    Keep the returned stage alive while reading USD layer provenance.
    """
    from pxr import Usd, UsdGeom

    path = Path(path)
    if not objects or len(set(objects)) != len(objects) or support in objects:
        raise ValueError("List distinct objects separately from the support")
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Cannot open {path}")
    root = stage.GetDefaultPrim()
    if not root:
        raise ValueError("Scene has no default prim")
    # Inspect Pcp errors directly: Isaac's bundled USD does not expose a
    # Stage.GetCompositionErrors method.
    errors = list(stage.GetPseudoRoot().GetPrimIndex().rootNode.layerStack.localErrors)
    for prim in stage.TraverseAll():
        errors.extend(prim.GetPrimIndex().localErrors)
    if errors:
        raise ValueError("USD composition errors: " + "; ".join(sorted({str(e) for e in errors})))
    if UsdGeom.GetStageUpAxis(stage) != "Z" or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1):
        raise ValueError("Flat scene audit requires Z-up and metersPerUnit=1")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    points = {name: world_corners(cache, root.GetChild(name)) for name in (support, *objects)}
    table = points.pop(support)
    if any(len(np.unique(table[:, axis].round(7))) != 2 for axis in range(3)):
        raise ValueError("Flat scene audit requires an axis-aligned rectangular support")
    issues = flat_layout_issues(points, table, pair_clearance=pair_clearance, edge_clearance=edge_clearance)
    surface = float(table[:, 2].max())
    bottoms = {name: float(p[:, 2].min()) - surface for name, p in points.items()}
    # Starts should be close to support before a settle pass, never embedded.
    for name, gap in bottoms.items():
        if gap < -0.001 or gap > 0.005:
            issues.append(f"{name}: authored bottom gap {gap:.4f} m outside [-0.001, 0.005] m")
    return {
        "scene": path.name, "objects": list(objects), "support": support,
        "scene_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "issues": issues, "bottom_gaps_m": bottoms,
        "pair_separations_m": {f"{first}/{second}": footprint_separation(points[first], points[second])
                               for i, first in enumerate(objects) for second in objects[i+1:]},
        "bounds": {name: {"lower": p.min(axis=0).tolist(), "upper": p.max(axis=0).tolist()}
                   for name, p in points.items()},
    }, stage


def audit_scene(path, target, *, pair_clearance=0.02, edge_clearance=0.01):
    """Compatibility wrapper for the two-object Simple-PnP library."""
    row, stage = audit_flat_scene(path, [target, "grey_bin"], pair_clearance=pair_clearance,
                                  edge_clearance=edge_clearance)
    row.update(target=target, footprint_separation_m=row["pair_separations_m"][f"{target}/grey_bin"])
    return row, stage


def layer_hashes(stage):
    return {str(Path(layer.realPath).resolve()): hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
            for layer in stage.GetUsedLayers() if layer.realPath}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--scene-dir", type=Path, help="Library with manifest.json; defaults to Simple-PnP")
    source.add_argument("--scene", type=Path, help="One scene with separate tabletop objects")
    parser.add_argument("--objects", nargs="+", help="All tabletop object/container prim names for --scene")
    parser.add_argument("--support", default="table", help="Rectangular support prim for --scene")
    parser.add_argument("--pattern", default="*", help="Scene filename glob within the manifest")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pair-clearance", type=float, default=0.02)
    parser.add_argument("--edge-clearance", type=float, default=0.01)
    args = parser.parse_args()
    if any(not np.isfinite(v) or v < 0 for v in (args.pair_clearance, args.edge_clearance)):
        parser.error("Clearances must be finite and nonnegative")
    if args.scene:
        if not args.objects or args.pattern != "*":
            parser.error("--scene requires --objects and does not accept --pattern")
        try:
            row, stage = audit_flat_scene(args.scene, args.objects, support=args.support,
                                          pair_clearance=args.pair_clearance, edge_clearance=args.edge_clearance)
            report = {"schema_version": 1, "scope": "static flat-table geometry; no physics or grasp validation",
                      "scene_path": str(args.scene.resolve()), "pair_clearance_m": args.pair_clearance,
                      "edge_clearance_m": args.edge_clearance, "asset_layer_sha256": layer_hashes(stage), **row}
            report["status"] = "failed" if row["issues"] else "passed"
        except Exception as error:
            report = {"schema_version": 1, "status": "failed", "scene_path": str(args.scene.resolve()),
                      "issues": [f"audit error: {type(error).__name__}: {error}"]}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("status", "scene_path", "issues")}, indent=2))
        return 1 if report["status"] == "failed" else 0
    if args.objects or args.support != "table":
        parser.error("--objects and --support require --scene")
    args.scene_dir = args.scene_dir or ROOT / "assets/scenes/simple_pnp"
    manifest_path = args.scene_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows, hashes = [], {}
    for item in manifest["scenes"]:
        if not Path(item["scene"]).match(args.pattern):
            continue
        path = args.scene_dir / item["scene"]
        try:
            row, stage = audit_scene(path, item["target"]["name"],
                                    pair_clearance=args.pair_clearance, edge_clearance=args.edge_clearance)
            # Read provenance while the composing stage still owns its layers.
            for layer in stage.GetUsedLayers():
                if layer.realPath and Path(layer.realPath).parent != args.scene_dir.resolve():
                    asset_path = Path(layer.realPath)
                    key = str(asset_path.relative_to(ROOT)) if asset_path.is_relative_to(ROOT) else str(asset_path)
                    if key not in hashes:
                        hashes[key] = hashlib.sha256(asset_path.read_bytes()).hexdigest()
        except Exception as error:
            row = {"scene": path.name, "target": item["target"]["name"],
                   "issues": [f"audit error: {type(error).__name__}: {error}"]}
        rows.append(row)
        if len(rows) % 250 == 0:
            print(f"Audited {len(rows)} scenes", flush=True)
    if not rows:
        parser.error("No scenes matched")
    counts = Counter()
    for row in rows:
        for kind in ("footprint clearance", "outside table", "authored bottom gap", "audit error"):
            if any(kind in issue for issue in row["issues"]):
                counts[kind] += 1
        if row.get("footprint_separation_m", 1) < 0:
            counts["overlapping footprint envelopes"] += 1
    report = {
        "schema_version": 1, "scope": "static composed geometry; no physics or grasp validation",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "pair_clearance_m": args.pair_clearance, "edge_clearance_m": args.edge_clearance,
        "scene_count": len(rows), "flagged_count": sum(bool(r["issues"]) for r in rows),
        "issue_counts": dict(counts), "asset_layer_sha256": hashes, "scenes": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("scenes", "asset_layer_sha256")}, indent=2))
    return 1 if report["flagged_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
