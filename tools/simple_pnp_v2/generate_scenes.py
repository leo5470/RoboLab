#!/usr/bin/env python3
"""Generate Simple-PnP v2 layouts from frozen inputs (plan step 3). Static only.

    tools/simple_pnp_v2/usd_python.sh tools/simple_pnp_v2/generate_scenes.py \
        --inputs output/simple_pnp_v2_validation/inputs/inputs.json \
        --handles output/simple_pnp_v2_validation/inputs/handle_records.json \
        --layouts 1                       # pilot: one layout per retained entry

For each requested slot (entry, layout number) it tries up to --tries
deterministic candidates. Each candidate samples from computed feasible sets:
first the bin centre (table support + edge clearance, release reach, v1 bin
novelty), then a target yaw and the target footprint centre (support, pair
clearance to the bin, grasp/lift reach, v1 target and offset novelty, within-v2
diversity, handle yaw targeting and occlusion screen). Positions are footprint
centres (transformed geometry), matching the frozen v1 references.

Accepted candidates are written to --out-dir with the payload root rotation
preserved (orient = yaw ⊗ root rotation), objects in contact with the measured
tabletop, and GroundPlane at the measured table bottom. Each written scene is
reopened and independently audited (composition, footprints, clearances,
bottom gaps, ground plane, composed resting height, novelty from the composed
poses). Slots with no accepted candidate are recorded as unresolved.

One JSON record per slot goes to --record-dir. Physics, grasp, and visibility
gates are recorded as pending: this tool does not run them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "tools")]

from freeze_inputs import quat_matrix, quat_mul, quat_yaw_deg, sha256, yaw_quat, rel  # noqa: E402

CAMERA = (0.05, 0.57, 0.66)  # OverShoulderLeftCameraCfg position (robolab/variations/camera.py)
GRID_STEP = 0.005
SCREEN_POINTS = 24


def defaults():
    return {
        "tries": 20, "edge_clearance": 0.01, "pair_clearance": 0.03, "audit_pair_clearance": 0.02,
        "novelty_m": 0.03, "diversity_m": 0.03, "theta_gen_deg": 45.0, "theta_screen_deg": 60.0,
        "grasp_heights": [0.03, 0.20], "release_height": 0.20, "tall_release_height": 0.35,
        "tall_threshold": 0.18, "contact_clearance": 0.0,
        # Perturbed physics runs move the bin +-2 mm XY (yaw fixed) and the
        # target +-2 mm XY / +-2 deg yaw about its root; keep perturbed starts
        # inside the preparation preflight's edge clearance (decided 2026-11-03).
        "perturb_xy_m": 0.002, "perturb_yaw_deg": 2.0,
    }


class Reach:
    def __init__(self, path):
        data = json.loads(Path(path).read_text())
        self.xs, self.ys = np.array(data["xs"]), np.array(data["ys"])
        self.grids = {round(float(k), 3): np.array(v, bool) for k, v in data["reachable"].items()}

    def ok(self, z, x, y):
        """Conservative: all four surrounding grid cells reachable."""
        G = self.grids[round(z, 3)]
        xs, ys = self.xs, self.ys
        i = np.clip(np.floor((x - xs[0]) / (xs[1] - xs[0])).astype(int), 0, len(xs) - 2)
        j = np.clip(np.floor((y - ys[0]) / (ys[1] - ys[0])).astype(int), 0, len(ys) - 2)
        inside = (x >= xs[0]) & (x <= xs[-1]) & (y >= ys[0]) & (y <= ys[-1])
        return inside & G[i, j] & G[i + 1, j] & G[i, j + 1] & G[i + 1, j + 1]


def rect_local(lower, upper, center_offset):
    """Footprint corners relative to the footprint centre, yaw 0."""
    x0, y0, x1, y1 = lower[0], lower[1], upper[0], upper[1]
    c = np.array(center_offset[:2])
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) - c


def rotate(points, yaw):
    """points (4,2), yaw (N,) radians -> (N,4,2)."""
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    return np.stack([points[None, :, 0] * c - points[None, :, 1] * s,
                     points[None, :, 0] * s + points[None, :, 1] * c], -1)


def separation_to_box(P, corners, yaw, box_lo, box_hi):
    """SAT gap between rotated rectangles (P + corners) and an axis-aligned box."""
    A = P[:, None, :] + corners
    B = np.array([[box_lo[0], box_lo[1]], [box_hi[0], box_lo[1]], [box_hi[0], box_hi[1]], [box_lo[0], box_hi[1]]])
    axes = [np.broadcast_to([1.0, 0.0], (len(P), 2)), np.broadcast_to([0.0, 1.0], (len(P), 2)),
            np.stack([np.cos(yaw), np.sin(yaw)], -1), np.stack([-np.sin(yaw), np.cos(yaw)], -1)]
    gaps = []
    for ax in axes:
        a = np.einsum("nkd,nd->nk", A, ax)
        b = B @ ax.T  # (4, N)
        gaps.append(np.maximum(b.min(0) - a.max(1), a.min(1) - b.max(0)))
    return np.max(gaps, 0)


def segment_hits_box(points, lo, hi, cam):
    """points (N,M,3) -> (N,M) bool: segment point->cam intersects AABB [lo, hi]."""
    d = np.asarray(cam) - points
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (np.asarray(lo) - points) / d
        t2 = (np.asarray(hi) - points) / d
    tmin = np.where(np.isfinite(t1), np.minimum(t1, t2), -np.inf)
    tmax = np.where(np.isfinite(t1), np.maximum(t1, t2), np.inf)
    # Axis-parallel segments outside the slab never hit.
    outside = (d == 0) & ((points < lo) | (points > hi))
    enter, leave = tmin.max(-1), tmax.min(-1)
    return (enter <= leave) & (leave > 1e-6) & (enter < 1.0) & ~outside.any(-1)


def wrap_deg(a):
    return (np.asarray(a) + 180) % 360 - 180


class Generator:
    def __init__(self, inputs, handles, reach, params, robot_boxes=()):
        self.inputs, self.handles, self.reach, self.p = inputs, handles, reach, params
        self.robot_boxes = [(np.array(b["position"]), quat_matrix(b["quat_wxyz"]), np.array(b["local_lower"]),
                             np.array(b["local_upper"])) for b in robot_boxes]
        t = inputs["table"]
        self.table_lo, self.table_hi = np.array(t["lower"]), np.array(t["upper"])
        self.top = t["top_z"]
        bg = inputs["bin"]["geometry"]
        self.bin_half = np.array(bg["rest_size_m"][:2]) / 2
        self.bin_off = np.array(bg["rest_center_offset_m"])
        self.bin_height = bg["rest_size_m"][2]
        self.bin_lower_z = bg["rest_bounds_at_origin"]["lower"][2]
        e = self.p["edge_clearance"]
        xs = np.arange(self.table_lo[0] + e, self.table_hi[0] - e + 1e-9, GRID_STEP)
        ys = np.arange(self.table_lo[1] + e, self.table_hi[1] - e + 1e-9, GRID_STEP)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        self.grid = np.stack([X.ravel(), Y.ravel()], -1)

    # ---- feasible sets -------------------------------------------------
    def bin_mask(self, P, entry, refs):
        e, half = self.p["edge_clearance"] + self.p["perturb_xy_m"], self.bin_half
        lo, hi = self.table_lo[:2] + e + half, self.table_hi[:2] - e - half
        m = {"support": np.all((P >= lo) & (P <= hi), 1)}
        z = self.p["tall_release_height"] if entry["geometry"]["rest_size_m"][2] > self.p["tall_threshold"] \
            else self.p["release_height"]
        m["release_reach"] = self.reach.ok(z, P[:, 0], P[:, 1])
        R = np.array([r["bin_center_xy"] for r in refs])
        m["v1_bin_novelty"] = np.min(np.linalg.norm(P[:, None] - R[None], axis=-1), 1) >= self.p["novelty_m"]
        return m

    def target_yaw(self, P, entry, handle, delta_deg, yaw_deg):
        if handle and handle["class"] == "A":
            bearing = np.degrees(np.arctan2(CAMERA[1] - P[:, 1], CAMERA[0] - P[:, 0]))
            return np.radians(bearing - handle["handle_direction_local_deg"] + delta_deg)
        return np.full(len(P), math.radians(yaw_deg))

    def target_mask(self, P, yaw, entry, b, refs, accepted, handle):
        g = entry["geometry"]
        rest = g["rest_bounds_at_origin"]
        corners = rotate(rect_local(rest["lower"], rest["upper"], g["rest_center_offset_m"]), yaw)
        # Perturbation margin: XY offset plus the farthest corner's sweep for the
        # yaw perturbation about the root pivot.
        reach_r = np.linalg.norm(rect_local(rest["lower"], rest["upper"], [0, 0]), axis=1).max()
        e = (self.p["edge_clearance"] + self.p["perturb_xy_m"]
             + 2 * reach_r * math.sin(math.radians(self.p["perturb_yaw_deg"]) / 2))
        A = P[:, None, :] + corners
        m = {"support": np.all((A >= self.table_lo[:2] + e) & (A <= self.table_hi[:2] - e), (1, 2))}
        sep = separation_to_box(P, corners, yaw, b - self.bin_half, b + self.bin_half)
        m["bin_clearance"] = sep >= self.p["pair_clearance"]
        reach = np.ones(len(P), bool)
        for z in self.p["grasp_heights"]:
            reach &= self.reach.ok(z, P[:, 0], P[:, 1])
        m["grasp_reach"] = reach
        RT = np.array([r["target_center_xy"] for r in refs]); RO = np.array([r["offset_xy"] for r in refs])
        m["v1_target_novelty"] = np.min(np.linalg.norm(P[:, None] - RT[None], axis=-1), 1) >= self.p["novelty_m"]
        m["v1_offset_novelty"] = np.min(np.linalg.norm((P - b)[:, None] - RO[None], axis=-1), 1) >= self.p["novelty_m"]
        div = np.ones(len(P), bool)
        for a in accepted:
            if np.linalg.norm(b - a["bin_center_xy"]) < self.p["diversity_m"]:
                div &= np.linalg.norm(P - a["target_center_xy"], axis=1) >= self.p["diversity_m"]
        m["v2_diversity"] = div
        if handle:
            # All screen points: mesh normals proved unreliable for a camera-facing
            # subset (visibility pilot 2), so no handle point may be blocked.
            m["handle_occlusion"] = ~self.occluded(P, yaw, entry, b, handle).any(1)
        return m, sep

    def screen_points(self, handle):
        pts = np.array(handle["points"])
        idx = np.linspace(0, len(pts) - 1, min(SCREEN_POINTS, len(pts))).round().astype(int)
        return pts[idx]

    def occluded(self, P, yaw, entry, b, handle):
        """(N, M) bool: handle point rays to the camera blocked by the bin or own body."""
        g = entry["geometry"]
        off = np.array(g["rest_center_offset_m"][:2])
        pts = self.screen_points(handle)
        c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        # root = P - Rz off; world = root + Rz pts_xy; z = pts_z + root_z
        rel_xy = pts[None, :, :2] - off[None, None, :]
        wx = P[:, None, 0] + rel_xy[..., 0] * c - rel_xy[..., 1] * s
        wy = P[:, None, 1] + rel_xy[..., 0] * s + rel_xy[..., 1] * c
        root_z = self.top - g["rest_bounds_at_origin"]["lower"][2]
        W = np.stack([wx, wy, np.broadcast_to(pts[None, :, 2] + root_z, wx.shape)], -1)
        bin_lo = [b[0] - self.bin_half[0], b[1] - self.bin_half[1], self.top]
        bin_hi = [b[0] + self.bin_half[0], b[1] + self.bin_half[1], self.top + self.bin_height]
        # Bin only: self-occlusion is addressed by yawing class-A handles toward
        # the camera (theta limits); a body-cylinder model fails for handles
        # inside the body footprint (jugs). The render-based visibility gate
        # measures actual occlusion, including self-occlusion.
        hit = segment_hits_box(W, bin_lo, bin_hi, CAMERA)
        # Robot links at the frozen start pose, as oriented boxes.
        cam = np.asarray(CAMERA, float)
        for pos, R, lo, hi in self.robot_boxes:
            hit |= segment_hits_box((W - pos) @ R, lo, hi, (cam - pos) @ R)
        return hit

    # ---- one slot -------------------------------------------------------
    def candidate(self, entry, refs, accepted, handle, rng):
        info = {}
        bm = self.bin_mask(self.grid, entry, refs)
        info["bin_feasible_cells"] = int(np.all(list(bm.values()), 0).sum())
        info["bin_eliminated_by"] = {k: int((~v).sum()) for k, v in bm.items()}
        cells = self.grid[np.all(list(bm.values()), 0)]
        if not len(cells):
            return None, info | {"reason": "no feasible bin region"}
        b = cells[rng.integers(len(cells))] + rng.uniform(-GRID_STEP / 2, GRID_STEP / 2, 2)
        if not np.all([v[0] for v in self.bin_mask(b[None], entry, refs).values()]):
            b = cells[rng.integers(len(cells))]
        delta = float(rng.uniform(-self.p["theta_gen_deg"], self.p["theta_gen_deg"]))
        yaw_deg = float(rng.uniform(0, 360))
        yaw = self.target_yaw(self.grid, entry, handle, delta, yaw_deg)
        tm, _ = self.target_mask(self.grid, yaw, entry, b, refs, accepted, handle)
        ok = np.all(list(tm.values()), 0)
        info.update(bin_center_xy=b.tolist(), delta_deg=delta if handle and handle["class"] == "A" else None,
                    sampled_yaw_deg=None if handle and handle["class"] == "A" else yaw_deg,
                    target_feasible_cells=int(ok.sum()),
                    target_eliminated_by={k: int((~v).sum()) for k, v in tm.items()})
        if not ok.any():
            return None, info | {"reason": "no feasible target region for sampled bin/yaw"}
        index = np.flatnonzero(ok)
        P = self.grid[index[rng.integers(len(index))]]
        Pj = P + rng.uniform(-GRID_STEP / 2, GRID_STEP / 2, 2)
        yj = self.target_yaw(Pj[None], entry, handle, delta, yaw_deg)
        mj, _ = self.target_mask(Pj[None], yj, entry, b, refs, accepted, handle)
        if all(v[0] for v in mj.values()):
            P = Pj
        yaw = float(self.target_yaw(P[None], entry, handle, delta, yaw_deg)[0])
        mf, sep = self.target_mask(P[None], np.array([yaw]), entry, b, refs, accepted, handle)
        failed = [k for k, v in mf.items() if not v[0]]
        if failed:
            return None, info | {"reason": "final point failed: " + ",".join(failed)}
        info.update(target_center_xy=P.tolist(), target_yaw_deg=math.degrees(yaw) % 360,
                    planned_bin_separation_m=float(sep[0]))
        return {"target_center_xy": P, "bin_center_xy": b, "yaw": yaw}, info

    def poses(self, entry, c):
        g = entry["geometry"]
        yaw = c["yaw"]
        off = np.array(g["rest_center_offset_m"][:2])
        Rz = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        root_xy = c["target_center_xy"] - Rz @ off
        root_z = self.top + self.p["contact_clearance"] - g["rest_bounds_at_origin"]["lower"][2]
        q = quat_mul(yaw_quat(math.degrees(yaw)), g["root_rotation_wxyz"])
        bg = self.inputs["bin"]["geometry"]
        bin_root = c["bin_center_xy"] - self.bin_off[:2]
        bq = quat_mul(yaw_quat(0.0), bg["root_rotation_wxyz"])
        return ({"translate": [*root_xy.tolist(), root_z], "orient": q, "scale": entry["scale"]},
                {"translate": [*bin_root.tolist(), self.top + self.p["contact_clearance"] - self.bin_lower_z],
                 "orient": bq, "scale": self.inputs["bin"]["scale"]})


# ---- USD authoring and independent re-audit -----------------------------
def author_scene(path, entry, target_pose, bin_pose, inputs):
    from pxr import Gf, Sdf, Usd, UsdUtils
    base_path = ROOT / "assets/scenes/base_empty.usda"
    base = Sdf.Layer.FindOrOpen(str(base_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    layer = Sdf.Layer.CreateNew(str(path))
    layer.TransferContent(base)

    def rebase(p):
        if not p or os.path.isabs(p) or "://" in p or p.startswith("omniverse:"):
            return p
        return os.path.relpath(os.path.normpath(os.path.join(base_path.parent, p)), path.parent)
    UsdUtils.ModifyAssetPaths(layer, rebase)
    stage = Usd.Stage.Open(layer)
    world = stage.GetDefaultPrim()
    ground = world.GetChild("GroundPlane").GetAttribute("xformOp:translate")
    g = ground.Get()
    ground.Set(type(g)(g[0], g[1], inputs["table"]["required_ground_plane_z"]))
    for name, payload, pose in ((entry["prim_name"], entry["payload"], target_pose),
                                ("grey_bin", inputs["bin"]["usd_path"], bin_pose)):
        prim = stage.DefinePrim(world.GetPath().AppendChild(name))
        prim.GetPayloads().AddPayload(os.path.relpath(ROOT / payload, path.parent))
        defaults = {"translate": Sdf.ValueTypeNames.Double3, "orient": Sdf.ValueTypeNames.Quatf,
                    "scale": Sdf.ValueTypeNames.Float3}
        for op, vt in defaults.items():
            attr = prim.GetAttribute("xformOp:" + op)
            if not attr:
                attr = prim.CreateAttribute("xformOp:" + op, vt)
            tn = str(attr.GetTypeName())
            v = pose[op]
            if op == "orient":
                Q = Gf.Quatd if tn == "quatd" else Gf.Quath if tn == "quath" else Gf.Quatf
                V = Gf.Vec3d if tn == "quatd" else Gf.Vec3h if tn == "quath" else Gf.Vec3f
                attr.Set(Q(float(v[0]), V(*map(float, v[1:]))))
            else:
                V = {"double3": Gf.Vec3d, "float3": Gf.Vec3f, "half3": Gf.Vec3h}[tn]
                attr.Set(V(*map(float, v)))
        prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray, False, Sdf.VariabilityUniform).Set(
            ["xformOp:translate", "xformOp:orient", "xformOp:scale"])
    layer.Save()


def reaudit(path, entry, inputs, refs, params, handle, target_pose):
    """Independent checks on the written file; returns (issues, measurements)."""
    from pxr import Usd, UsdGeom
    from audit_simple_pnp_scenes import audit_flat_scene, layer_hashes, world_corners
    issues = []
    row, stage = audit_flat_scene(path, [entry["prim_name"], "grey_bin"],
                                  pair_clearance=params["audit_pair_clearance"],
                                  edge_clearance=params["edge_clearance"])
    issues += row["issues"]
    world = stage.GetDefaultPrim()
    gz = UsdGeom.Xformable(world.GetChild("GroundPlane")).ComputeLocalToWorldTransform(0).ExtractTranslation()[2]
    if abs(gz - inputs["table"]["required_ground_plane_z"]) > 1e-6:
        issues.append(f"GroundPlane z {gz} != table bottom {inputs['table']['required_ground_plane_z']}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    tc = world_corners(cache, world.GetChild(entry["prim_name"]))
    bc = world_corners(cache, world.GetChild("grey_bin"))
    size = tc.max(0) - tc.min(0)
    expect = entry["geometry"]["rest_size_m"][2]
    if abs(size[2] - expect) > 0.001:
        issues.append(f"composed height {size[2]:.4f} != resting height {expect:.4f} (dropped root rotation?)")
    bsize = bc.max(0) - bc.min(0)
    if np.abs(bsize - inputs["bin"]["geometry"]["rest_size_m"]).max() > 1e-4:
        issues.append(f"bin composed size {bsize.tolist()} differs from the pin")
    # Root rotation preserved: composed orientation == yaw ⊗ root rotation.
    prim = world.GetChild(entry["prim_name"])
    order = [op.GetOpName() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]
    if order != ["xformOp:translate", "xformOp:orient", "xformOp:scale"]:
        issues.append(f"unexpected op order {order}")
    t_xy = ((tc.min(0) + tc.max(0)) / 2)[:2]; b_xy = ((bc.min(0) + bc.max(0)) / 2)[:2]
    RT = np.array([r["target_center_xy"] for r in refs]); RB = np.array([r["bin_center_xy"] for r in refs])
    RO = np.array([r["offset_xy"] for r in refs])
    dT, dB, dO = (np.linalg.norm(v, axis=1) for v in (RT - t_xy, RB - b_xy, RO - (t_xy - b_xy)))
    nov = {"target_m": float(dT.min()), "bin_m": float(dB.min()), "offset_m": float(dO.min()),
           "nearest": {"target": refs[int(dT.argmin())]["scene"], "bin": refs[int(dB.argmin())]["scene"],
                       "offset": refs[int(dO.argmin())]["scene"]}}
    for k in ("target_m", "bin_m", "offset_m"):
        if nov[k] < params["novelty_m"]:
            issues.append(f"composed v1 novelty {k} {nov[k]:.4f} < {params['novelty_m']}")
    meas = {"geometry_audit": {k: row[k] for k in ("issues", "bottom_gaps_m", "pair_separations_m", "bounds")},
            "composed_target_center_xy": t_xy.tolist(), "composed_bin_center_xy": b_xy.tolist(),
            "composed_target_size_m": size.tolist(), "v1_novelty": nov,
            "scene_sha256": row["scene_sha256"], "layer_sha256": layer_hashes(stage)}
    if handle and handle["class"] == "A":
        yaw = quat_yaw_deg(quat_mul(target_pose["orient"], [entry["geometry"]["root_rotation_wxyz"][0],
                                                            *[-v for v in entry["geometry"]["root_rotation_wxyz"][1:]]]))
        world_dir = (yaw + handle["handle_direction_local_deg"]) % 360
        bearing = math.degrees(math.atan2(CAMERA[1] - t_xy[1], CAMERA[0] - t_xy[0]))
        angle = abs(float(wrap_deg(world_dir - bearing)))
        meas["handle"] = {"predicted_world_direction_deg": world_dir, "camera_bearing_deg": bearing,
                          "angle_to_camera_deg": angle}
        if angle > params["theta_screen_deg"]:
            issues.append(f"handle {angle:.1f} deg from camera bearing > {params['theta_screen_deg']}")
    return issues, meas


def scene_name(entry, layout):
    return f"pnp_{entry['catalog_index']:03d}_{entry['prim_name']}_{layout:02d}.usda"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--handles", type=Path, required=True)
    parser.add_argument("--entries", type=int, nargs="*", help="Catalog indices (default: all retained)")
    parser.add_argument("--layouts", type=int, default=1, help="Layouts per entry")
    parser.add_argument("--layout-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "assets/scenes/simple_pnp_v2")
    parser.add_argument("--record-dir", type=Path, default=ROOT / "output/simple_pnp_v2_validation/generation")
    parser.add_argument("--params", type=Path, help="JSON overriding generation parameters")
    parser.add_argument("--unresolved", type=Path, default=HERE / "unresolved_entries.json",
                        help="Entries recorded as unresolved without generation")
    parser.add_argument("--robot-occluders", type=Path,
                        default=ROOT / "output/simple_pnp_v2_validation/inputs/robot_occluders.json")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate slots that already have a record")
    parser.add_argument("--replace-slots", type=Path,
                        help="Top-up: JSON {slot: [failed gates]}; regenerate only these slots with fresh "
                             "candidates (--try-offset), diversity against every other authored layout")
    parser.add_argument("--try-offset", type=int, default=0,
                        help="First candidate index (top-up rounds use 20, 40, ... so no earlier candidate recurs)")
    parser.add_argument("--archive-dir", type=Path,
                        help="With --replace-slots: where the replaced scene and record are moved")
    args = parser.parse_args()
    params = defaults() | (json.loads(args.params.read_text()) if args.params else {})
    inputs = json.loads(args.inputs.read_text())
    handles_doc = json.loads(args.handles.read_text())
    if handles_doc["inputs_sha256"] != sha256(args.inputs):
        raise SystemExit("handle records were built from different inputs")
    reach_path = ROOT / inputs["reach_map"]["path"]
    if sha256(reach_path) != inputs["reach_map"]["sha256"]:
        raise SystemExit("reach map changed since inputs were frozen")
    for entry in inputs["entries"]:
        if sha256(ROOT / entry["payload"]) != entry["payload_sha256"]:
            raise SystemExit(f"{entry['payload']} changed since inputs were frozen")
    unresolved = json.loads(args.unresolved.read_text())["entries"]
    robot = json.loads(args.robot_occluders.read_text())
    gen = Generator(inputs, handles_doc["records"], Reach(reach_path), params, robot["boxes"])
    entries = [e for e in inputs["entries"] if args.entries is None or e["catalog_index"] in args.entries]
    provenance = {"inputs_sha256": sha256(args.inputs), "handles_sha256": sha256(args.handles),
                  "reach_map_sha256": inputs["reach_map"]["sha256"], "params": params, "seed": args.seed,
                  "generator_sha256": sha256(__file__), "camera_position": CAMERA, "grid_step_m": GRID_STEP,
                  "robot_occluders_sha256": sha256(args.robot_occluders)}
    args.record_dir.mkdir(parents=True, exist_ok=True)
    summary = {"authored": 0, "unresolved": 0, "skipped": 0}
    replace = json.loads(args.replace_slots.read_text()) if args.replace_slots else None
    if replace is not None and not args.archive_dir:
        raise SystemExit("--replace-slots needs --archive-dir")
    for entry in entries:
        k = entry["catalog_index"]
        refs = inputs["v1_references"][str(k)]
        handle = handles_doc["records"].get(str(k))
        accepted = []
        layouts = range(args.layout_start + args.layouts)
        if replace is not None:
            todo = []
            for layout in layouts:
                name = scene_name(entry, layout)
                record_path = args.record_dir / name.replace(".usda", ".json")
                if name.removesuffix(".usda") in replace:
                    todo.append(layout)
                elif record_path.exists():
                    prior = json.loads(record_path.read_text())
                    if prior["status"] == "authored":
                        accepted.append({"target_center_xy": np.array(prior["measurements"]["composed_target_center_xy"]),
                                         "bin_center_xy": np.array(prior["measurements"]["composed_bin_center_xy"])})
            layouts = todo
        for layout in layouts:
            name = scene_name(entry, layout)
            record_path = args.record_dir / name.replace(".usda", ".json")
            replaces = None
            if replace is not None:
                args.archive_dir.mkdir(parents=True, exist_ok=True)
                replaces = {"failed_gates": replace[name.removesuffix(".usda")], "try_offset": args.try_offset}
                if record_path.exists():
                    replaces["record_sha256"] = sha256(record_path)
                    record_path.rename(args.archive_dir / record_path.name)
                if (args.out_dir / name).exists():
                    replaces["scene_sha256"] = sha256(args.out_dir / name)
                    (args.out_dir / name).rename(args.archive_dir / name)
            elif record_path.exists():
                prior = json.loads(record_path.read_text())
                if layout < args.layout_start or not args.overwrite:
                    if prior["status"] == "authored":
                        accepted.append({"target_center_xy": np.array(prior["measurements"]["composed_target_center_xy"]),
                                         "bin_center_xy": np.array(prior["measurements"]["composed_bin_center_xy"])})
                    if layout >= args.layout_start:
                        summary["skipped"] += 1
                    continue
            if layout < args.layout_start:
                continue
            if str(k) in unresolved:
                record = {"slot": name.removesuffix(".usda"), "catalog_index": k, "prim_name": entry["prim_name"],
                          "layout": layout, "status": "unresolved", "tries": [],
                          "unresolved_reason": "entry excluded from v2 generation: " + unresolved[str(k)]["reason"],
                          "unresolved_config_sha256": sha256(args.unresolved), "provenance": provenance,
                          "gates": {g: "not_run" for g in ("geometry", "physics", "grasp", "visibility")}}
                record_path.write_text(json.dumps(record, indent=1))
                (args.out_dir / name).unlink(missing_ok=True)
                summary["unresolved"] += 1
                continue
            t0 = time.time()
            tries = []
            record = {"slot": name.removesuffix(".usda"), "catalog_index": k, "prim_name": entry["prim_name"],
                      "layout": layout, "payload": entry["payload"], "payload_sha256": entry["payload_sha256"],
                      "scale": entry["scale"], "bin": {k2: inputs["bin"][k2] for k2 in ("usd_path", "scale", "payload_sha256")},
                      "provenance": provenance,
                      "handle": None if handle is None else {
                          "class": handle["class"], "confirmed": handle["confirmed"],
                          "record_sha256": hashlib.sha256(json.dumps(handle, sort_keys=True).encode()).hexdigest(),
                          "handle_direction_local_deg": handle.get("handle_direction_local_deg"),
                          "theta_gen_deg": params["theta_gen_deg"], "theta_screen_deg": params["theta_screen_deg"]},
                      "v1_reference_count": len(refs)}
            if replaces:
                record["replaces"] = replaces
            status = "unresolved"
            scene_path = args.out_dir / name
            for t in range(args.try_offset, args.try_offset + params["tries"]):
                rng = np.random.default_rng([args.seed, k, layout, t])
                c, info = gen.candidate(entry, refs, accepted, handle, rng)
                info["try"] = t
                if c is None:
                    tries.append(info)
                    continue
                target_pose, bin_pose = gen.poses(entry, c)
                author_scene(scene_path, entry, target_pose, bin_pose, inputs)
                issues, meas = reaudit(scene_path, entry, inputs, refs, params, handle, target_pose)
                if issues:
                    info["reason"] = "re-audit: " + "; ".join(issues)
                    tries.append(info)
                    scene_path.unlink()
                    continue
                tries.append(info | {"accepted": True})
                diversity = [float(min(np.linalg.norm(c["target_center_xy"] - a["target_center_xy"]),
                                       np.linalg.norm(c["bin_center_xy"] - a["bin_center_xy"]))) for a in accepted]
                record.update(scene=rel(scene_path), target_pose=target_pose, bin_pose=bin_pose,
                              target_yaw_deg=info["target_yaw_deg"], measurements=meas,
                              v2_diversity_min_m=min(diversity) if diversity else None)
                accepted.append({"target_center_xy": c["target_center_xy"], "bin_center_xy": c["bin_center_xy"]})
                status = "authored"
                break
            record.update(status=status, tries=tries, elapsed_s=time.time() - t0,
                          gates={"geometry": "passed" if status == "authored" else "pending",
                                 "physics": "pending", "grasp": "pending",
                                 "visibility": "pending" if handle else "not_applicable"})
            if status != "authored":
                record["unresolved_reason"] = tries[-1].get("reason") if tries else "no tries"
            record_path.write_text(json.dumps(record, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
            summary[status] += 1
            print(f"{name}: {status} after {len(tries)} tries ({time.time() - t0:.1f}s)"
                  + ("" if status == "authored" else f" — {record['unresolved_reason']}"), flush=True)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
