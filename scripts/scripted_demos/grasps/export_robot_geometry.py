"""Phase 2: export what the selector and the library need from the robot and gripper USDs (pxr only, no simulator).

* ``robot_chain.npz``: the DROID arm's joint chain for numpy FK/IK (``robolab.core.scripted_demos.kinematics``), from
  RoboLab's ``tools/simple_pnp_v2/reach_map.py:load_chain``, with FK checked against the authored ``base_link``.
* ``gripper_geometry.json``: the open DROID Robotiq 2F-85 as 6 mm slice boxes in ``base_link`` (each link's surface
  sampled area-weighted, cut along the approach axis +X); the inner fingers are split into the contact pad
  (x >= 0.128 m) and the finger body. Written into the package (``robolab/core/scripted_demos/``), where ``select`` loads it.
* ``gripper_frame_map.json``: the fixed transform from GraspDataGen's Robotiq ``base_link`` to DROID's (D5). Both
  grippers are Robotiq 2F-85 models authored in different frames: GraspDataGen's fingers extend along +Z with the left
  finger at -Y, DROID's along +X with the left finger at +Y. The rotation is fixed (DROID x = GDG z, DROID y = -GDG y,
  DROID z = GDG x); the translation is measured so the open fingers' bounding boxes coincide.

    scripts/scripted_demos/run.sh --pxr scripts/scripted_demos/grasps/export_robot_geometry.py [--out-dir DIR] [--geometry-out FILE]
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from robolab.core.scripted_demos.grasps import frames, objects
from robolab.core.scripted_demos import kinematics, paths
from robolab.core.scripted_demos.records import file_sha256

ROBOT_USD = objects.ROBOLAB_DIR / "assets/robots/franka_robotiq_2f_85_flattened.usd"
PACKAGE_GEOMETRY = pathlib.Path(kinematics.__file__).with_name("gripper_geometry.json")


def _repo_relative(path: pathlib.Path) -> str:
    """A path inside this checkout relative to it, so the tracked gripper_geometry.json names no local directory."""
    path = pathlib.Path(path).resolve()
    return str(path.relative_to(objects.ROBOLAB_DIR)) if path.is_relative_to(objects.ROBOLAB_DIR) else str(path)


# ---- arm chain -----------------------------------------------------------------------------------------------------

def export_chain(out: pathlib.Path) -> None:
    from pxr import Usd
    from tools.simple_pnp_v2.reach_map import load_chain

    usd = ROBOT_USD
    chain, T_root, T_base_authored, _ = load_chain(Usd.Stage.Open(str(usd)))
    revolute = np.array([c[2] for c in chain])
    limits = np.array([c[3] if c[2] else (0.0, 0.0) for c in chain])
    # The chain ends at panda_hand_joint's child; a fixed correction maps that frame onto base_link exactly.
    raw = kinematics.Chain(np.stack([c[0] for c in chain]), np.stack([c[1] for c in chain]), revolute, limits,
                           T_root, np.eye(4))
    T_end_base = np.linalg.inv(raw.fk(kinematics.HOME)) @ T_base_authored
    np.savez(out, T0=np.stack([c[0] for c in chain]), T1inv=np.stack([c[1] for c in chain]), revolute=revolute,
             limits=limits, T_root=T_root, T_end_base=T_end_base, usd=str(usd),
             usd_sha256=file_sha256(usd))
    loaded = kinematics.Chain.load(out)
    err = np.abs(loaded.fk(kinematics.HOME) - T_base_authored).max()
    print(f"wrote {out}; T_end_base translation {T_end_base[:3, 3].round(5)} rotation angle "
          f"{np.degrees(np.arccos(np.clip((np.trace(T_end_base[:3, :3]) - 1) / 2, -1, 1))):.3f} deg; "
          f"FK(HOME) vs authored base_link max abs {err:.2e}")
    print("T_root (panda_link0 in USD world):\n", T_root.round(5))
    print("base_link at HOME:\n", T_base_authored.round(5))


# ---- gripper slice geometry --------------------------------------------------------------------------------------

SLICE = 0.006
PAD_START_X = 0.128
LINKS = {"palm": ["base_link"], "fingers": ["left_inner_finger", "right_inner_finger"],
         "outer": ["left_outer_finger", "right_outer_finger", "left_outer_knuckle", "right_outer_knuckle"],
         "knuckles": ["left_inner_knuckle", "right_inner_knuckle"]}


def surface_points(prim, base_inv, rng, n=40000):
    from pxr import Usd, UsdGeom

    tris = []
    for q in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not (q.IsA(UsdGeom.Mesh) and q.GetAttribute("points").Get()):
            continue
        m = UsdGeom.Mesh(q)
        v = np.array(m.GetPointsAttr().Get(), float)
        M = base_inv @ np.array(UsdGeom.Xformable(q).ComputeLocalToWorldTransform(0)).T
        v = (M @ np.c_[v, np.ones(len(v))].T).T[:, :3]
        counts, idx, s = np.array(m.GetFaceVertexCountsAttr().Get()), np.array(m.GetFaceVertexIndicesAttr().Get()), 0
        for k in counts:
            f = idx[s : s + k]
            s += k
            tris.extend(v[[f[0], f[j], f[j + 1]]] for j in range(1, k - 1))
    T = np.array(tris)
    area = np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1) / 2
    pick = rng.choice(len(T), n, p=area / area.sum())
    u = rng.random((n, 2))
    flip = u.sum(1) > 1
    u[flip] = 1 - u[flip]
    t = T[pick]
    return np.vstack([t[:, 0] + u[:, :1] * (t[:, 1] - t[:, 0]) + u[:, 1:] * (t[:, 2] - t[:, 0]), T.reshape(-1, 3)])


def slices(points, start=None, stop=None):
    x = points[:, 0]
    lo_x = x.min() if start is None else start
    hi_x = x.max() if stop is None else stop
    boxes = []
    for x0 in np.arange(lo_x, hi_x, SLICE):
        x1 = min(x0 + SLICE, hi_x)
        s = points[(x >= x0) & (x <= x1)]
        if len(s):
            boxes.append([[float(x0), *s[:, 1:].min(0).round(5).tolist()], [float(x1), *s[:, 1:].max(0).round(5).tolist()]])
    return boxes


def export_gripper_geometry(out: pathlib.Path) -> None:
    from pxr import Usd, UsdGeom

    usd = ROBOT_USD
    stage = Usd.Stage.Open(str(usd))
    prims = {}
    for p in stage.Traverse():
        if "Robotiq" in str(p.GetPath()):
            prims.setdefault(p.GetName(), p)
    base_inv = np.linalg.inv(np.array(UsdGeom.Xformable(prims["base_link"]).ComputeLocalToWorldTransform(0)).T)
    rng = np.random.default_rng(0)
    parts: dict[str, list] = {"palm": [], "pads": [], "fingers": [], "outer": [], "knuckles": []}
    for part, links in LINKS.items():
        for link in links:
            pts = surface_points(prims[link], base_inv, rng)
            if part == "palm":
                parts["palm"].append([pts.min(0).round(5).tolist(), pts.max(0).round(5).tolist()])
            elif part == "fingers":
                parts["pads"] += slices(pts, start=PAD_START_X)
                parts["fingers"] += slices(pts[pts[:, 0] < PAD_START_X], stop=PAD_START_X)
            else:
                parts[part] += slices(pts)
    out.write_text(json.dumps({"frame": "DROID Robotiq base_link, open (finger_joint 0)", "slice_m": SLICE,
                               "robot_usd": _repo_relative(usd), "robot_usd_sha256": file_sha256(usd),
                               "parts": parts}, indent=1))
    print({k: len(v) for k, v in parts.items()}, "->", out)


# ---- GraspDataGen -> DROID gripper frame map ---------------------------------------------------------------------

FINGERS = ("left_inner_finger", "right_inner_finger")


def finger_boxes(usd_path, robotiq_scope: str | None):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    prims = {}
    for p in stage.Traverse():
        if p.GetName() in ("base_link", *FINGERS) and (robotiq_scope is None or robotiq_scope in str(p.GetPath())):
            prims.setdefault(p.GetName(), p)

    def world(prim):  # column-vector convention
        return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T

    base_inv = np.linalg.inv(world(prims["base_link"]))
    boxes = {}
    for finger in FINGERS:
        pts = []
        for q in Usd.PrimRange(prims[finger], Usd.TraverseInstanceProxies()):
            if q.IsA(UsdGeom.Mesh) and q.GetAttribute("points").Get():
                P = np.c_[np.array(q.GetAttribute("points").Get(), float), np.ones(len(q.GetAttribute("points").Get()))]
                pts.append((base_inv @ world(q) @ P.T).T[:, :3])
        P = np.vstack(pts)
        boxes[finger] = (P.min(0), P.max(0))
    return boxes, {k: str(v.GetPath()) for k, v in prims.items()}


def export_frame_map(out: pathlib.Path) -> None:
    gdg_usd = paths.graspdatagen_dir() / "bots/robotiq_2f_85.usd"
    droid_usd = ROBOT_USD
    gdg, gdg_paths = finger_boxes(gdg_usd, None)
    droid, droid_paths = finger_boxes(droid_usd, "Robotiq")

    R = frames.R_GDG_DROID  # columns: DROID axes in GDG coordinates
    # DROID box corners mapped into the GDG frame (rotation only), per finger. Left/right names follow each asset.
    report = {"rotation_gdg_from_droid": R.tolist(), "fingers": {}}
    centers_gdg, centers_droid = [], []
    for finger in FINGERS:
        lo_d, hi_d = droid[finger]
        corners = np.array([[x, y, z] for x in (lo_d[0], hi_d[0]) for y in (lo_d[1], hi_d[1]) for z in (lo_d[2], hi_d[2])])
        mapped = corners @ R.T
        # The mirrored y axis swaps which physical finger is "left"; match boxes by side, not by name.
        centers_droid.append(mapped.mean(0))
        report["fingers"][f"droid_{finger}"] = {"min": lo_d.round(5).tolist(), "max": hi_d.round(5).tolist(),
                                                "in_gdg_axes_min": mapped.min(0).round(5).tolist(),
                                                "in_gdg_axes_max": mapped.max(0).round(5).tolist()}
        lo_g, hi_g = gdg[finger]
        report["fingers"][f"gdg_{finger}"] = {"min": lo_g.round(5).tolist(), "max": hi_g.round(5).tolist()}
    gdg_centers = np.array([(gdg[f][0] + gdg[f][1]) / 2 for f in FINGERS])
    droid_centers = np.array(centers_droid)
    # Pair each DROID finger with the GDG finger on the same side (sign of GDG y).
    order = np.argsort(gdg_centers[:, 1]), np.argsort(droid_centers[:, 1])
    gdg_sorted, droid_sorted = gdg_centers[order[0]], droid_centers[order[1]]
    t = (gdg_sorted - droid_sorted).mean(0)  # DROID origin in GDG coordinates
    residual = gdg_sorted - (droid_sorted + t)

    sizes_g = np.array([gdg[f][1] - gdg[f][0] for f in FINGERS])
    sizes_d = np.array([np.abs(R @ (droid[f][1] - droid[f][0])) for f in FINGERS])
    report.update({
        "translation_gdg_from_droid_m": t.round(6).tolist(),
        "pair_residual_mm": (residual * 1000).round(3).tolist(),
        "finger_box_size_difference_mm": ((sizes_g - sizes_d) * 1000).round(2).tolist(),
        "open_gap_mm": {"gdg": float(abs(np.diff(gdg_sorted[:, 1]))[0] * 1000),
                        "droid": float(abs(np.diff(droid_sorted[:, 1]))[0] * 1000)},
        "gdg_usd": str(gdg_usd), "gdg_usd_sha256": file_sha256(gdg_usd),
        "droid_usd": str(droid_usd), "droid_usd_sha256": file_sha256(droid_usd),
        "prims": {"gdg": gdg_paths, "droid": droid_paths},
    })
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("translation_gdg_from_droid_m", "pair_residual_mm",
                                             "finger_box_size_difference_mm", "open_gap_mm")}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None, help="default: <grasp_data>")
    parser.add_argument("--geometry-out", type=pathlib.Path, default=PACKAGE_GEOMETRY)
    args = parser.parse_args()
    out = args.out_dir or objects.grasp_data_root()
    out.mkdir(parents=True, exist_ok=True)
    export_chain(out / "robot_chain.npz")
    export_gripper_geometry(args.geometry_out)
    export_frame_map(out / "gripper_frame_map.json")


if __name__ == "__main__":
    main()
