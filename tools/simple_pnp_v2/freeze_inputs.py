#!/usr/bin/env python3
"""Freeze Simple-PnP v2 generation inputs (plan step 1). Static: no physics.

    tools/simple_pnp_v2/usd_python.sh tools/simple_pnp_v2/freeze_inputs.py \
        --reach-map output/simple_pnp_v2_validation/inputs/reach_map.json \
        --output output/simple_pnp_v2_validation/inputs/inputs.json

Records, from the actual files:
- the retained-entry allowlist (catalog index, catalog path, the payload the
  retained v1 scenes actually use, prim name, scale) and the excluded indices;
- payload and composed-dependency hashes, and the grey-bin pin;
- each payload's geometry: children-only bounds (the frame v2 places with),
  the root rotation v2 must preserve (yaw is applied on top), and the resting
  height/footprint that results;
- the measured table, mount, and ground plane from base_empty.usda;
- every v1 layout for each retained entry (current library plus the removed
  layouts in --removed-dir), as geometry-based footprint centres computed from
  the transform v1 actually evaluated (translate, yaw orient, scale; v1 dropped
  root rotations), plus target yaw.
Fails if any retained entry has inconsistent payloads/scales across layouts,
or if the bin differs from the v1 pin.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
V1_DIR = ROOT / "assets/scenes/simple_pnp"
BIN_PIN = {"name": "grey_bin", "usd_path": "assets/fixtures/grey_bin.usd", "scale": [0.007, 0.007, 0.007], "yaw_deg": 0.0}
FIXTURES = {"table", "franka_table", "grey_bin"}
PRIM_RE = re.compile(r'def "(\w+)" \(\s*prepend payload = @([^@]+)@\s*\)\s*\{(.*?)\n    \}', re.S)
OP_RE = re.compile(r'xformOp:(translate|orient|scale) = \(([^)]*)\)')
NAME_RE = re.compile(r"pnp_(\d+)_(.+)_(\d+)\.usda$")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return os.path.relpath(Path(path).resolve(), ROOT)


def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return [w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2]


def quat_matrix(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def yaw_quat(deg):
    a = math.radians(deg) / 2
    return [math.cos(a), 0.0, 0.0, math.sin(a)]


def quat_yaw_deg(q):
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)))


def parse_scene(path):
    text = Path(path).read_text()
    prims = {}
    for name, payload, body in PRIM_RE.findall(text):
        ops = {k: [float(v) for v in vals.split(",")] for k, vals in OP_RE.findall(body)}
        order = re.search(r"xformOpOrder = \[([^\]]*)\]", body)
        prims[name] = {"payload": payload, "ops": ops, "order": order.group(1) if order else None}
    ground = re.search(r'def Xform "GroundPlane".*?xformOp:translate = \(([^)]*)\)', text, re.S)
    return prims, [float(v) for v in ground.group(1).split(",")] if ground else None


def open_checked(path):
    from pxr import Usd
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Cannot open {path}")
    errors = list(stage.GetPseudoRoot().GetPrimIndex().rootNode.layerStack.localErrors)
    for prim in stage.TraverseAll():
        errors.extend(prim.GetPrimIndex().localErrors)
    if errors:
        raise ValueError(f"{path}: composition errors: " + "; ".join(sorted({str(e) for e in errors})))
    return stage


def layer_hashes(stage):
    return {rel(layer.realPath): sha256(layer.realPath) for layer in stage.GetUsedLayers()
            if layer.realPath and Path(layer.realPath).is_file()}


def measure_payload(path, intended_scale):
    """Geometry in the frame v2 places with: world = T * R(yaw ⊗ root_rot) * S * children."""
    from pxr import Gf, Usd, UsdGeom
    stage = open_checked(path)
    root = stage.GetDefaultPrim()
    if not root:
        raise ValueError(f"{path}: no default prim")
    if UsdGeom.GetStageUpAxis(stage) != "Z" or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1):
        raise ValueError(f"{path}: needs Z-up, metersPerUnit=1")
    xf = UsdGeom.Xformable(root)
    ops = [{"name": op.GetOpName(), "value": np.asarray(op.Get(), float).ravel().tolist() if not isinstance(
        op.Get(), (Gf.Quatf, Gf.Quatd)) else [op.Get().GetReal(), *op.Get().GetImaginary()]}
        for op in xf.GetOrderedXformOps()]
    local = Gf.Transform(xf.GetLocalTransformation())
    rq = local.GetRotation().GetQuat()
    root_rot = [rq.GetReal(), *rq.GetImaginary()]
    if root_rot[0] < 0:
        root_rot = [-v for v in root_rot]
    from robolab.core.scenes.placement import prim_local_bounds
    lower, upper = prim_local_bounds(root)  # exact, from geometry points
    corners = np.array([[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])
    rest = (corners * np.asarray(intended_scale)) @ quat_matrix(root_rot).T  # yaw 0, root at origin
    up = quat_matrix(root_rot) @ np.array([0, 0, 1.0])
    return {
        "root_xform_ops": ops,
        "root_rotation_wxyz": root_rot,
        "root_rotation_tilt_deg": math.degrees(math.acos(max(-1, min(1, float(up[2]))))),
        "root_rotation_yaw_deg": quat_yaw_deg(root_rot),
        "children_bounds_local": {"lower": lower.tolist(), "upper": upper.tolist()},
        "rest_bounds_at_origin": {"lower": rest.min(0).tolist(), "upper": rest.max(0).tolist()},
        "rest_size_m": (rest.max(0) - rest.min(0)).tolist(),
        "rest_center_offset_m": ((rest.max(0) + rest.min(0)) / 2).tolist(),
        "layer_sha256": layer_hashes(stage),
    }


def v1_center(geometry, ops, rotation_preserved=False):
    """Footprint centre and bottom of a v1 layout from the transform v1 evaluated."""
    lo = np.array(geometry["children_bounds_local"]["lower"]); hi = np.array(geometry["children_bounds_local"]["upper"])
    q = ops["orient"]
    if rotation_preserved:
        q = quat_mul(q, geometry["root_rotation_wxyz"])
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    world = (corners * np.asarray(ops["scale"])) @ quat_matrix(q).T + np.asarray(ops["translate"])
    return ((world.min(0) + world.max(0)) / 2)[:2].tolist(), float(world[:, 2].min())


def measure_base():
    from pxr import Usd, UsdGeom
    stage = open_checked(ROOT / "assets/scenes/base_empty.usda")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    world = stage.GetDefaultPrim()
    out = {}
    for name in ("table", "franka_table"):
        r = cache.ComputeWorldBound(world.GetChild(name)).ComputeAlignedRange()
        out[name] = {"lower": list(r.GetMin()), "upper": list(r.GetMax())}
    ground = UsdGeom.Xformable(world.GetChild("GroundPlane")).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    out["ground_plane_z_authored"] = float(ground.ExtractTranslation()[2])
    out["layer_sha256"] = layer_hashes(stage)
    return out


def environment():
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return None
    import importlib.metadata as md
    versions = {}
    for pkg in ("isaacsim", "isaaclab"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = None
    return {"host": platform.node(), "python": sys.version.split()[0], "packages": versions,
            "git_head": run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
            "gpus": run(["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--removed-dir", type=Path,
                        default=ROOT / "output/simple_pnp_cleanup_qhncc8ix/assets/scenes/simple_pnp",
                        help="Removed v1 layouts (launch-filtered and excluded targets), at the original depth")
    parser.add_argument("--reach-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    catalog_path = ROOT / "assets/objects/object_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    items = catalog if isinstance(catalog, list) else catalog["objects"]
    manifest_path = V1_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    retained = sorted({s["catalog_index"] for s in manifest["scenes"]})
    excluded = sorted({e["catalog_index"] for e in manifest["excluded_targets"]})
    if set(retained) & set(excluded):
        raise SystemExit(f"Retained/excluded overlap: {sorted(set(retained) & set(excluded))}")
    if len(retained) != manifest["target_count"]:
        raise SystemExit("Retained entry count disagrees with the v1 manifest")

    sources = {"current": sorted(V1_DIR.glob("pnp_*.usda")), "removed": sorted(args.removed_dir.glob("pnp_*.usda"))}
    layouts = collections.defaultdict(list)
    problems = []
    for source, files in sources.items():
        for path in files:
            m = NAME_RE.search(path.name)
            index = int(m.group(1))
            if index not in retained:
                continue
            prims, ground = parse_scene(path)
            targets = [n for n in prims if n not in FIXTURES]
            if len(targets) != 1 or "grey_bin" not in prims:
                problems.append(f"{path.name}: expected one target and grey_bin, found {sorted(prims)}")
                continue
            layouts[index].append({"source": source, "scene": path.name, "path": rel(path),
                                   "sha256": sha256(path), "target_prim": targets[0],
                                   "target": prims[targets[0]], "bin": prims["grey_bin"], "ground": ground})

    measured = {}

    def geometry(payload_rel, scale):
        key = (payload_rel, tuple(scale))
        if key not in measured:
            measured[key] = measure_payload(ROOT / payload_rel, scale)
        return measured[key]

    bin_geometry = geometry(BIN_PIN["usd_path"], BIN_PIN["scale"])
    entries, references = [], {}
    for index in retained:
        rows = layouts[index]
        payloads = {os.path.normpath(V1_DIR / r["target"]["payload"]) for r in rows}
        scales = {tuple(r["target"]["ops"]["scale"]) for r in rows}
        prims = {r["target_prim"] for r in rows}
        if len(payloads) != 1 or len(scales) != 1 or len(prims) != 1:
            problems.append(f"{index}: inconsistent payload/scale/prim across v1 layouts: {payloads} {scales} {prims}")
            continue
        for r in rows:
            b = r["bin"]
            if (os.path.normpath(V1_DIR / b["payload"]) != str(ROOT / BIN_PIN["usd_path"])
                    or not np.allclose(b["ops"]["scale"], BIN_PIN["scale"])):
                problems.append(f"{r['scene']}: bin differs from the v1 pin")
        payload_rel = rel(next(iter(payloads)))
        scale = list(next(iter(scales)))
        item = items[index]
        g = geometry(payload_rel, scale)
        entries.append({
            "catalog_index": index, "catalog_name": item["name"], "catalog_usd_path": item["usd_path"],
            "payload": payload_rel, "substituted": payload_rel != item["usd_path"],
            "prim_name": next(iter(prims)), "scale": scale, "catalog_dims": item.get("dims"),
            "payload_sha256": sha256(ROOT / payload_rel), "geometry": g,
        })
        refs = []
        for r in sorted(rows, key=lambda r: r["scene"]):
            t_xy, t_bottom = v1_center(g, r["target"]["ops"])
            b_xy, _ = v1_center(bin_geometry, r["bin"]["ops"])
            refs.append({"scene": r["scene"], "source": r["source"], "path": r["path"], "sha256": r["sha256"],
                         "target_center_xy": t_xy, "bin_center_xy": b_xy,
                         "offset_xy": [t_xy[0] - b_xy[0], t_xy[1] - b_xy[1]],
                         "target_yaw_deg": quat_yaw_deg(r["target"]["ops"]["orient"]),
                         "target_root_translate": r["target"]["ops"]["translate"],
                         "target_bottom_z_v1": t_bottom})
        references[str(index)] = refs

    if problems:
        print("\n".join(problems[:50]), file=sys.stderr)
        raise SystemExit(f"{len(problems)} input problems; nothing frozen")
    coverage = collections.Counter(len(v) for v in references.values())
    base = measure_base()
    table = base["table"]
    reach = args.reach_map.resolve()
    out = {
        "schema_version": 1,
        "description": "Frozen Simple-PnP v2 generation inputs (plan step 1)",
        "environment": environment(),
        "files": {rel(p): sha256(p) for p in (catalog_path, manifest_path, ROOT / "robolab/tasks/single_pnp/exclusions.json",
                                               ROOT / "assets/scenes/base_empty.usda",
                                               ROOT / "assets/robots/franka_robotiq_2f_85_flattened.usd", reach)},
        "reach_map": {"path": rel(reach), "sha256": sha256(reach)},
        "table": {**table, "top_z": table["upper"][2], "bottom_z": table["lower"][2],
                  "required_ground_plane_z": table["lower"][2]},
        "base_scene": base,
        "bin": {**BIN_PIN, "payload_sha256": sha256(ROOT / BIN_PIN["usd_path"]), "geometry": bin_geometry},
        "retained_indices": retained, "excluded_indices": excluded,
        "entries": entries,
        "v1_references": references,
        "v1_reference_coverage": {"layouts_per_entry": dict(coverage), "sources": {k: len(v) for k, v in sources.items()},
                                  "removed_dir": rel(args.removed_dir),
                                  "complete": set(coverage) == {manifest["generated_variants_per_target"]}},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    tilted = [(e["catalog_index"], e["prim_name"], round(e["geometry"]["root_rotation_tilt_deg"], 2),
               round(e["geometry"]["root_rotation_yaw_deg"], 2)) for e in entries
              if e["geometry"]["root_rotation_tilt_deg"] > 0.01 or abs(e["geometry"]["root_rotation_yaw_deg"]) > 0.01]
    print(f"{len(entries)} entries, references {dict(coverage)}, substituted "
          f"{[e['catalog_index'] for e in entries if e['substituted']]}")
    print(f"table x {table['lower'][0]:.4f}-{table['upper'][0]:.4f} y {table['lower'][1]:.4f}-{table['upper'][1]:.4f} "
          f"z {table['lower'][2]:.4f}-{table['upper'][2]:.4f}; bin rest size {np.round(bin_geometry['rest_size_m'], 4).tolist()}")
    print("root rotations to preserve:", tilted)
    print("wrote", args.output, sha256(args.output))


if __name__ == "__main__":
    main()
