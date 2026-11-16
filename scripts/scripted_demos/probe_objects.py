"""Phases 0-1 in one simulator process: per batch object, on one training layout of the stock registration,

* the simulated mass PhysX gives it (``masses.json``; the jugs' catalog mass is 0, so PhysX derives it);
* a smoke test: cameras present, control rate, VRAM and step time over ``--steps`` held steps (``probe.json``);
* the frame check of the exported OBJ (Phase 1 gate: < 1 mm): our object-frame box, carried by
  ``WorldState.get_pose`` right after reset, against the object's bound in the live stage
  (``UsdGeom.BBoxCache.ComputeWorldBound``). For the YCB assets that bound comes back as a world-axis box (identity
  matrix), which ``WorldState.get_bbox`` inherits, so for a yawed YCB object its "object-frame" corners are inflated;
  the check compares both forms of our box (oriented, and boxed along the world axes) and records the matching one
  (``meshes/frame_check.json``).

    scripts/scripted_demos/run.sh scripts/scripted_demos/probe_objects.py [--objects 303 ...] [--layout 01] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
import subprocess
import time

import numpy as np

from robolab.core.scripted_demos.grasps import frames, objects
from robolab.core.scripted_demos import sim


def oriented_bound_corners(prim) -> np.ndarray:
    """World corners of a prim's bound: its range carried by its matrix (identity for a world-axis bound)."""
    from pxr import Usd, UsdGeom

    bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(prim)
    lo, hi = np.array(bound.GetRange().GetMin()), np.array(bound.GetRange().GetMax())
    local = np.array(list(itertools.product(*zip(lo, hi))))
    return (np.c_[local, np.ones(8)] @ np.array(bound.GetMatrix()))[:, :3]  # row-vector convention


def nearest_corner_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None] - b[None], axis=-1).min(axis=1)


def frame_score(T: np.ndarray, vertices: np.ndarray, reference: np.ndarray) -> dict:
    """Max corner error of ``reference`` against our object-frame box carried by ``T``, as that oriented box and as
    its world-axis bounding box."""
    lo, hi = vertices.min(0), vertices.max(0)
    oriented = frames.transform_points(T, np.array(list(itertools.product(*zip(lo, hi)))))
    axis = np.array(list(itertools.product(*zip(oriented.min(0), oriented.max(0)))))
    e_oriented = float(nearest_corner_error(oriented, reference).max())
    e_axis = float(nearest_corner_error(axis, reference).max())
    return {"error_oriented_mm": e_oriented * 1000, "error_world_axis_mm": e_axis * 1000,
            "reference_form": "oriented" if e_oriented <= e_axis else "world-axis",
            "max_corner_error_mm": min(e_oriented, e_axis) * 1000, "obj_corners": oriented.round(5).tolist()}


def gpu_memory_mb() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    for line in out.splitlines():
        pid, used = (x.strip() for x in line.split(","))
        if int(pid) == os.getpid():
            return float(used)
    return None


def update_json(path: pathlib.Path, rows: dict) -> None:
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objects", nargs="*", default=[str(o.index) for o in objects.BATCH])
    parser.add_argument("--layout", default="01")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--out-dir", type=pathlib.Path, default=None, help="default: <grasp_data>")
    args = parser.parse_args()
    out = args.out_dir or objects.grasp_data_root()

    sim.launch_app()
    import torch
    from robolab.core.world.world_state import get_world

    masses, probes, frame_checks = {}, {}, {}
    for key in args.objects:
        obj = objects.get(key)
        task = obj.task(args.layout)
        t0 = time.time()
        env, env_cfg = sim.make_env(task, out / "robolab_output" / "probe" / task)
        created_s = time.time() - t0
        obs, _ = env.reset()
        world = get_world(env)

        # Frame check, before anything moves.
        T = sim.object_pose(world, obj.name)
        v = objects.mesh_vertices(obj)
        reference = oriented_bound_corners(world._get_prim(obj.name, env_id=0)) - env.scene.env_origins[0].cpu().numpy()
        scored = frame_score(T, v, reference)
        robolab = nearest_corner_error(np.array(scored["obj_corners"]), sim.bbox_corners(world, obj.name)).max()
        frame_checks[obj.key] = {"task": task, **scored, "pose": T.round(6).tolist(),
                                 "stage_bound_corners": reference.round(5).tolist(),
                                 "robolab_get_bbox_corner_error_mm": float(robolab * 1000),
                                 "table_top_z": sim.table_top(world),
                                 "bin_corners": sim.bbox_corners(world, sim.BIN).round(5).tolist(),
                                 "passed": bool(scored["max_corner_error_mm"] < 1.0)}

        # Smoke test: hold the arm for a number of steps.
        robot = env.scene["robot"]
        arm_ids, _ = robot.find_joints([f"panda_joint{i}" for i in range(1, 8)], preserve_order=True)
        action = torch.zeros(1, 8, device=env.device)
        action[0, :7] = robot.data.joint_pos[0, arm_ids]
        t0 = time.time()
        for _ in range(args.steps):
            obs, _, term, trunc, _ = env.step(action)
        masses[obj.key] = float(env.scene[obj.name].root_physx_view.get_masses().reshape(-1)[0])
        probes[obj.key] = {
            "task": task, "env_create_s": round(created_s, 1), "step_ms": round((time.time() - t0) / args.steps * 1000, 1),
            "terminated": bool(term.any()), "truncated": bool(trunc.any()),
            "image_obs": {k: list(x.shape) for k, x in obs["image_obs"].items()},
            "viewport_cam": {k: list(x.shape) for k, x in obs.get("viewport_cam", {}).items()},
            "gpu_memory_mb": gpu_memory_mb(), "control_hz": 1.0 / (env_cfg.sim.dt * env_cfg.decimation),
            "max_episode_length": int(env.max_episode_length), "instruction": env_cfg.instruction,
            "simulated_mass_kg": masses[obj.key]}
        print(f"{obj.key}: mass {masses[obj.key]:.3f} kg, frame error {scored['max_corner_error_mm']:.3f} mm "
              f"({scored['reference_form']}), step {probes[obj.key]['step_ms']} ms", flush=True)
        env.close()

    update_json(out / "masses.json", masses)
    update_json(out / "probe.json", probes)
    update_json(out / "meshes" / "frame_check.json", frame_checks)


if __name__ == "__main__":
    status = 0
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        status = 1
    finally:
        sim.close_app(status)
