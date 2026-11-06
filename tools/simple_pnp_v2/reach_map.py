#!/usr/bin/env python3
"""Kinematic top-down reach map for the DROID Franka + Robotiq 2F-85 (no collisions).

Run with pxr available, e.g.:
    tools/simple_pnp_v2/usd_python.sh tools/simple_pnp_v2/reach_map.py \
        --heights 0.03 0.20 0.35 --output output/simple_pnp_v2_validation/reach_map.json

Reads the joint chain (local frames, axes, limits) from the robot USD and checks
that forward kinematics at the DROID reset pose reproduces the authored
base_link pose. The tool point lies on base_link +X (the fingers' direction in
this asset), --tip-inset metres inside the farthest finger-mesh point. For each
grid cell and tool height, damped-least-squares IK seeks a pose with the fingers
pointing world -Z and free yaw. This ignores arm/table/bin collisions and
non-top-down approaches: it is a placement prefilter, not a grasp check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HOME = np.array([0, -np.pi / 5, 0, -4 * np.pi / 5, 0, 3 * np.pi / 5, 0])  # DroidCfg init_state


def load_chain(stage):
    from pxr import Gf, UsdGeom, UsdPhysics
    prims = {}
    for p in stage.Traverse():  # names repeat in this asset; keep the first, as Traverse orders them
        prims.setdefault(p.GetName(), p)

    def pose(p, q):
        q = Gf.Quatd(q.GetReal(), *q.GetImaginary()).GetNormalized()
        T = np.eye(4); T[:3, :3] = np.array(Gf.Matrix3d(q)).T; T[:3, 3] = p
        return T

    chain = []
    for name in [f"panda_joint{i}" for i in range(1, 9)] + ["panda_hand_joint"]:
        p = prims[name]; j = UsdPhysics.Joint(p); revolute = p.IsA(UsdPhysics.RevoluteJoint)
        if revolute and p.GetAttribute("physics:axis").Get() != "X":
            raise ValueError(f"{name}: expected joint axis X")
        limits = (np.radians(p.GetAttribute("physics:lowerLimit").Get()),
                  np.radians(p.GetAttribute("physics:upperLimit").Get())) if revolute else None
        chain.append((pose(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()),
                      np.linalg.inv(pose(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get())), revolute, limits))
    world = lambda prim: np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
    return chain, world(prims["panda_link0"]), world(prims["base_link"]), prims


def build_solver(robot, tip_inset=0.015):
    """Top-down IK for the DROID Franka: returns (reachable(target_xyz) -> bool, metadata)."""
    from pxr import Usd, UsdGeom
    path = ROOT / robot
    stage = Usd.Stage.Open(str(path))
    chain, T_root, T_base_authored, prims = load_chain(stage)
    limits = np.array([c[3] for c in chain if c[2]])

    def rotx(a):
        c, s = np.cos(a), np.sin(a); T = np.eye(4); T[1:3, 1:3] = [[c, -s], [s, c]]; return T

    def fk(q):
        T = T_root.copy(); k = 0
        for T0, T1inv, revolute, _ in chain:
            T = T @ T0 @ (rotx(q[k]) if revolute else np.eye(4)) @ T1inv; k += revolute
        return T

    fk_error = float(np.abs(fk(HOME)[:3, 3] - T_base_authored[:3, 3]).max())
    if fk_error > 1e-4:
        raise SystemExit(f"FK at the DROID reset pose misses the authored base_link by {fk_error:.4g} m")
    inverse = np.linalg.inv(T_base_authored); tip = -np.inf
    for p in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
        if p.IsA(UsdGeom.Mesh) and "finger" in str(p.GetPath()):
            P = np.array(UsdGeom.Mesh(p).GetPointsAttr().Get() or [], float)
            if len(P):
                W = (np.c_[P, np.ones(len(P))] @ np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0))) @ inverse.T
                tip = max(tip, float(W[:, 0].max()))
    tool = tip - tip_inset

    def error(q, target):
        T = fk(q)
        return np.r_[target - (T[:3, :3] @ [tool, 0, 0] + T[:3, 3]), 0.3 * np.cross(T[:3, 0], [0, 0, -1])]

    rng = np.random.default_rng(0)
    seeds = [HOME] + [HOME + rng.normal(0, 0.6, 7) for _ in range(4)]

    def reachable(target):
        target = np.asarray(target, float)
        for q in seeds:
            q = q.copy()
            for _ in range(150):
                e = error(q, target)
                if np.linalg.norm(e[:3]) < 2e-3 and np.linalg.norm(e[3:]) < 0.3 * np.sin(np.radians(3)):
                    return True
                J = np.stack([(error(q + d, target) - e) / 1e-5 for d in np.eye(7) * 1e-5], 1)
                q = np.clip(q - J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e), limits[:, 0], limits[:, 1])
        return False

    meta = {"robot": str(robot), "robot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fk_check_error_m": fk_error, "fingertip_along_base_link_x_m": tip, "tool_point_m": tool,
            "approach": "top-down, fingers -Z, yaw free; kinematic only (no collisions)"}
    return reachable, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", type=Path, default=Path("assets/robots/franka_robotiq_2f_85_flattened.usd"))
    parser.add_argument("--heights", type=float, nargs="+", default=[0.03, 0.20, 0.35], help="Tool-point heights (m)")
    parser.add_argument("--x", type=float, nargs=3, default=[0.20, 0.90, 0.05], metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--y", type=float, nargs=3, default=[-0.50, 0.50, 0.05], metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--tip-inset", type=float, default=0.015)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reachable, meta = build_solver(args.robot, args.tip_inset)

    xs = np.round(np.arange(args.x[0], args.x[1] + 1e-9, args.x[2]), 4)
    ys = np.round(np.arange(args.y[0], args.y[1] + 1e-9, args.y[2]), 4)
    grids = {}
    for z in args.heights:
        grid = [[reachable(np.array([x, y, z])) for y in ys] for x in xs]
        grids[f"{z:.3f}"] = grid
        print(f"\ntool z={z:.2f} m (rows x {xs[0]}..{xs[-1]}, cols y {ys[0]}..{ys[-1]}; #=reachable)")
        for x, row in zip(xs, grid):
            print(f"x={x:.2f} " + "".join("#" if r else "." for r in row))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({**meta, "xs": xs.tolist(), "ys": ys.tolist(), "reachable": grids}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
