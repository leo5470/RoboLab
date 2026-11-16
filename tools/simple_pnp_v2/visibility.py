#!/usr/bin/env python3
"""Handle-visibility gate for Simple-PnP v2 handle entries (plan step 4). Needs Isaac + GPU.

    .venv/bin/python tools/simple_pnp_v2/run_pinned.py cuda:0 tools/simple_pnp_v2/visibility.py \
        --tasks SimplePnpV2230MilkjugA0100Task ... \
        --handles output/simple_pnp_v2_validation/inputs/handle_records.json \
        --inputs output/simple_pnp_v2_validation/inputs/inputs.json \
        --output output/simple_pnp_v2_validation/visibility/<batch>

For each task: fresh OMY environment with a qualification copy of
OverShoulderLeftCameraCfg (same pose and intrinsics, plus depth), robot at its
frozen start pose, ScenePreparation with clearance 0, then the same 5.5 s hold
as physics qualification. The handle point set is moved with the *simulated*
final pose and projected into the rendered image; a point is visible when the
rendered depth anywhere in its 3x3 pixel neighbourhood matches within the
tolerance (thin handles straddle background pixels).

Area-based rule (decided 2026-11-03): the visible handle area is the convex
hull of the visible projected points. A task passes when
  * area >= LIMITS["min_pixel_area"] native px (~3 px at the 224x224 policy input; decided 2026-11-03),
  * area x depth^2 >= LIMITS["min_reference_ratio"] x the entry's best-case
    value from the reference sweep (--reference; distance-normalised, so it is
    comparable across layouts), which captures occlusion and turning away,
  * class A: the simulated handle direction is within theta_screen of the
    camera bearing.
The visible point fraction is recorded as a diagnostic only.

Sweep mode (--yaw-offsets) yaws the target about its footprint centre and
records the same measurements per offset; visibility_reference.py turns sweeps
into the reference file. Writes and flushes each report before teardown.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402 -- must precede Isaac
from isaaclab.app import AppLauncher  # noqa: E402

LIMITS = {"depth_tolerance_m": 0.01, "depth_window_px": 3, "min_pixel_area": 100.0,
          "min_reference_ratio": 0.5, "hold_s": 5.5, "theta_screen_deg": 60.0}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--tasks", nargs="+", required=True)
parser.add_argument("--task-dirs", nargs="+", default=["single_pnp_v2_drafts"])
parser.add_argument("--handles", type=Path, required=True)
parser.add_argument("--inputs", type=Path, required=True)
parser.add_argument("--reference", type=Path, help="visibility_reference.py output (required unless sweeping)")
parser.add_argument("--seed", type=int, default=20261102)
parser.add_argument("--yaw-offsets", type=float, nargs="*", help="Sweep study: yaw offsets (deg) about the footprint centre")
parser.add_argument("--sweep-hold-s", type=float, default=1.0)
parser.add_argument("--sweep-at", type=float, nargs=2, metavar=("X", "Y"),
                    help="Sweep: move the target footprint centre here and park the bin (--park-bin) first")
parser.add_argument("--park-bin", type=float, nargs=2, default=[0.66, -0.34], metavar=("X", "Y"),
                    help="Sweep: bin footprint centre while sweeping at --sweep-at (behind the target, seen from the camera)")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
args.kit_args = (args.kit_args or "") + " --/app/fastShutdown=false"
app = AppLauncher(args).app


def qmat(q):
    import numpy as np
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def move_footprint(asset, offset_rest, rest_rot, xy, env, torch, np):
    """Translate an asset so its footprint centre is at xy (orientation and height kept)."""
    pose = asset.data.root_pose_w.clone()
    R = qmat(pose[0, 3:].cpu().numpy()) @ qmat(rest_rot).T
    centre = pose[0, :3].cpu().numpy() + R @ np.array(offset_rest)
    pose[0, 0] += float(xy[0] - centre[0]); pose[0, 1] += float(xy[1] - centre[1])
    asset.write_root_pose_to_sim(pose)
    env.scene.write_data_to_sim(); env.sim.forward(); env.scene.update(dt=0.0)


def rotate_about_footprint(asset, entry, delta_deg, env, torch, np):
    """Yaw the target by delta about its footprint centre (sweep studies only)."""
    pose = asset.data.root_pose_w.clone()
    q = pose[0, 3:].cpu().numpy()
    R = qmat(q) @ qmat(entry["geometry"]["root_rotation_wxyz"]).T
    centre = pose[0, :3].cpu().numpy() + R @ np.array(entry["geometry"]["rest_center_offset_m"])
    a = math.radians(delta_deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    root = centre + Rz @ (pose[0, :3].cpu().numpy() - centre)
    w, x, y, z = q; c, s = math.cos(a / 2), math.sin(a / 2)
    qn = [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w]  # (c,0,0,s) ⊗ q
    pose[0, :3] = torch.tensor(root, dtype=pose.dtype, device=pose.device)
    pose[0, 3:] = torch.tensor(qn, dtype=pose.dtype, device=pose.device)
    asset.write_root_pose_to_sim(pose)
    env.scene.write_data_to_sim(); env.sim.forward(); env.scene.update(dt=0.0)


def measure(env, camera, target, entry, handle, np, cv2, reference=None):
    rgb = camera.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
    depth = camera.data.output["distance_to_image_plane"][0].cpu().numpy().squeeze(-1)
    K = camera.data.intrinsic_matrices[0].cpu().numpy()
    cam_pos = camera.data.pos_w[0].cpu().numpy(); cam_q = camera.data.quat_w_ros[0].cpu().numpy()
    pose = env.scene.rigid_objects[target].data.root_pose_w[0].cpu().numpy()
    # rest frame -> world: world = T + R(orient) R(root_rot)^-1 p_rest
    R = qmat(pose[3:]) @ qmat(entry["geometry"]["root_rotation_wxyz"]).T
    P = np.array(handle["points"]) @ R.T + pose[:3]
    C = (P - cam_pos) @ qmat(cam_q)  # ROS optical frame: x right, y down, z forward
    z = C[:, 2]
    u = K[0, 0] * C[:, 0] / z + K[0, 2]; v = K[1, 1] * C[:, 1] / z + K[1, 2]
    H, W = depth.shape
    inside = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ui, vi = np.clip(u.astype(int), 0, W - 1), np.clip(v.astype(int), 0, H - 1)
    r = LIMITS["depth_window_px"] // 2
    err = np.full(len(P), np.inf)
    for dv in range(-r, r + 1):
        for du in range(-r, r + 1):
            err = np.minimum(err, np.abs(depth[np.clip(vi + dv, 0, H - 1), np.clip(ui + du, 0, W - 1)] - z))
    visible = inside & (err <= LIMITS["depth_tolerance_m"])
    area = float(cv2.contourArea(cv2.convexHull(np.stack([u[visible], v[visible]], 1).astype(np.float32)))) \
        if visible.sum() >= 3 else 0.0
    depth_m = float(np.median(z[inside])) if inside.any() else None
    normalized = area * depth_m ** 2 if depth_m else 0.0  # px * m^2: invariant to camera distance
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    final_dir = None if handle.get("handle_direction_local_deg") is None else (yaw + handle["handle_direction_local_deg"]) % 360
    c = P.mean(0)
    bearing = math.degrees(math.atan2(cam_pos[1] - c[1], cam_pos[0] - c[0]))
    angle = None if final_dir is None else abs((final_dir - bearing + 180) % 360 - 180)
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for x, y, ok in zip(ui[inside], vi[inside], visible[inside]):
        cv2.circle(overlay, (int(x), int(y)), 2, (0, 255, 0) if ok else (0, 0, 255), -1)
    if visible.sum() >= 3:
        hull = cv2.convexHull(np.stack([u[visible], v[visible]], 1).astype(np.int32))
        cv2.polylines(overlay, [hull], True, (255, 255, 0), 1)
    ratio = None if not reference else normalized / reference["best_normalized_area"]
    reasons = []
    if not inside.any():
        reasons.append("handle outside the image")
    if area < LIMITS["min_pixel_area"]:
        reasons.append(f"pixel area {area:.0f} < {LIMITS['min_pixel_area']:.0f}")
    if reference is not None and ratio < LIMITS["min_reference_ratio"]:
        reasons.append(f"area {ratio:.2f} of best case < {LIMITS['min_reference_ratio']}")
    if handle["class"] == "A" and angle is not None and angle > LIMITS["theta_screen_deg"]:
        reasons.append(f"handle {angle:.1f} deg from camera bearing > {LIMITS['theta_screen_deg']}")
    return {"status": "passed" if not reasons else "failed", "reasons": reasons,
            "pixel_area": area, "handle_depth_m": depth_m, "normalized_area": normalized, "reference_ratio": ratio,
            "visible_fraction": float(visible.mean()), "points": len(P), "in_image_fraction": float(inside.mean()),
            "final_handle_direction_deg": final_dir, "camera_bearing_deg": bearing, "angle_to_camera_deg": angle,
            "camera_pos": cam_pos.tolist(), "camera_quat_ros": cam_q.tolist(), "intrinsics": K.tolist(),
            "final_pose": pose.tolist(), "per_point": {"visible": visible.tolist(), "inside": inside.tolist()},
            "_rgb": cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "_overlay": overlay}


def main():
    import numpy as np
    import torch
    import robolab.constants
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.runtime import create_env
    from robolab.core.omy_teleop.collection import build_parser
    from robolab.core.omy_teleop.config import configure_device, configure_performance, register_omy_envs
    from robolab.core.omy_teleop.environment import OmyEnv
    from robolab.core.omy_teleop.options import ARM_JOINTS
    from robolab.core.omy_teleop.preparation_runtime import ScenePreparation
    from robolab.core.task.task_index import resolve_selected_task

    robolab.constants.VERBOSE = False
    args.output.mkdir(parents=True, exist_ok=True)
    handles = json.loads(args.handles.read_text())["records"]
    # Key by catalog index: prim names repeat across entries (mug, pitcher, bowl, ...).
    inputs_doc = json.loads(args.inputs.read_text())
    entries = {e["catalog_index"]: e for e in inputs_doc["entries"]}
    bin_geometry = inputs_doc["bin"]["geometry"]
    if not args.yaw_offsets and args.reference is None:
        raise SystemExit("--reference is required for the gate (build it with visibility_reference.py)")
    references = json.loads(args.reference.read_text())["references"] if args.reference else {}
    for task in args.tasks:
        out = args.output / f"{task}.json"
        if out.exists():
            continue
        row = {"task": task, "limits": LIMITS, "status": "error", "camera": "OverShoulderLeftCameraCfg + depth copy"}
        env = None
        started = time.time()
        try:
            options = build_parser().parse_args([
                "--desktop", "--source", "tcp", "--device", args.device, "--control-hz", "15",
                "--recording-profile", "compact", "--stepping-profile", "held-target",
                "--contact-sensors", "required", "--contact-history-length", "0",
                "--solver-position-iterations", "16", "--episode-length-s", "120",
                "--max-attempt-steps", "1800", "--max-attempt-seconds", "120", "--scene-settle-clearance-m", "0"])
            options.task, options.task_dirs = task, args.task_dirs
            options.task_resolution = resolve_selected_task(task, options.task_dirs)
            robolab.constants.set_output_dir(str((args.output / "isaac" / task).resolve()))
            register_omy_envs(task, task_dirs=options.task_dirs, resolution=options.task_resolution,
                              qualification_camera=True)
            cfg = parse_env_cfg(task + "OMYL100", device=args.device, num_envs=1)
            cfg.episode_length_s = 120
            configure_performance(cfg, options)
            configure_device(cfg, options)
            scene_path = Path(str(cfg.scene.scene.spawn.usd_path))
            row.update(scene=str(scene_path), scene_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest())
            env, cfg = create_env(cfg, policy="simple_pnp_v2_visibility", rendering_mode="performance", env_class=OmyEnv)
            target = [n for n in env.scene.rigid_objects if n not in ("table", "grey_bin")][0]
            index = int(scene_path.stem.split("_")[1])  # pnp_<index>_<name>_<layout>
            entry = entries[index]
            if entry["prim_name"] != target:
                raise ValueError(f"scene {scene_path.name} target {target} != catalog entry {entry['prim_name']}")
            handle = handles[str(entry["catalog_index"])]
            reference = None
            if not args.yaw_offsets:
                if str(index) not in references:
                    raise ValueError(f"no visibility reference for catalog entry {index}")
                reference = references[str(index)]
                row.update(reference=reference, reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest())
            row.update(target=target, handle_class=handle["class"],
                       handle_record_sha256=hashlib.sha256(json.dumps(handle, sort_keys=True).encode()).hexdigest())
            action = torch.tensor([[*[cfg.scene.robot.init_state.joint_pos[n] for n in ARM_JOINTS], 1.0]], device=env.device)
            camera = env.scene.sensors["over_shoulder_left_camera"]
            offsets = args.yaw_offsets if args.yaw_offsets else [None]
            results = []
            for delta in offsets:
                with torch.inference_mode():
                    env.reset(seed=args.seed)
                    if delta is not None and args.sweep_at:
                        bg = bin_geometry
                        move_footprint(env.scene.rigid_objects["grey_bin"], bg["rest_center_offset_m"],
                                       bg["root_rotation_wxyz"], args.park_bin, env, torch, np)
                        move_footprint(env.scene.rigid_objects[target], entry["geometry"]["rest_center_offset_m"],
                                       entry["geometry"]["root_rotation_wxyz"], args.sweep_at, env, torch, np)
                    if delta is not None:
                        rotate_about_footprint(env.scene.rigid_objects[target], entry, delta, env, torch, np)
                    prepare = ScenePreparation(env, options)
                    while app.is_running():
                        result = prepare.advance()
                        if result["status"] != "settling":
                            break
                    if result["status"] != "ready":
                        msg = "preparation " + result["status"] + ": " + "; ".join(result.get("reasons", []))
                        if delta is None:
                            raise RuntimeError(msg)
                        results.append({"yaw_offset_deg": delta, "status": "error", "reasons": [msg]})
                        continue
                    hold = LIMITS["hold_s"] if delta is None else args.sweep_hold_s
                    for _ in range(math.ceil(hold / env.step_dt)):
                        env.step(action)
                    for _ in range(4):  # let the renderer converge on the final state
                        env.sim.render()
                    camera.update(dt=env.step_dt)
                    m = measure(env, camera, target, entry, handle, np, cv2, reference)
                tag = task if delta is None else f"{task}_yaw{delta:+04.0f}"
                img = args.output / f"{tag}_rgb.png"; ov = args.output / f"{tag}_overlay.png"
                cv2.imwrite(str(img), m.pop("_rgb")); cv2.imwrite(str(ov), m.pop("_overlay"))
                m.update(yaw_offset_deg=delta, rgb=str(img), overlay=str(ov))
                results.append(m)
            if args.yaw_offsets:
                row.update(status="sweep", sweep=results, sweep_at=args.sweep_at,
                           park_bin=args.park_bin if args.sweep_at else None)
            else:
                row.update(results[0])
                row["renderer"] = "RTX performance preset"
        except Exception:
            row["error"] = traceback.format_exc()
            traceback.print_exc()
        finally:
            row["wall_s"] = time.time() - started
            tmp = out.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(row, f, indent=1); f.flush(); os.fsync(f.fileno())
            tmp.rename(out)
            print(f"{task}: {row.get('status')} area={row.get('pixel_area')} ratio={row.get('reference_ratio')} "
                  f"sweep={[(r['yaw_offset_deg'], round(r.get('pixel_area') or 0)) for r in row.get('sweep', [])]} "
                  f"angle={row.get('angle_to_camera_deg')} {row.get('reasons', '')} ({row['wall_s']:.1f}s)", flush=True)
            if env is not None:
                env.close()
    (args.output / "done").write_text("ok\n")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    # Every report was fsynced before teardown. Kit's shutdown can hang or
    # crash (exit 139), so bound it: exit without app.close().
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
