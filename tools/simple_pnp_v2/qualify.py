#!/usr/bin/env python3
"""Physics qualification of Simple-PnP v2 draft tasks (plan step 4). Needs Isaac and a GPU.

    .venv/bin/python tools/simple_pnp_v2/run_pinned.py cuda:0 tools/simple_pnp_v2/qualify.py \
        --tasks SimplePnpV2230MilkjugA0100Task ... --runs 5 \
        --output output/simple_pnp_v2_validation/physics/<batch>

Uses the OMY collection profile (OmyDroidCfg, 15 Hz, held-target stepping,
required contact sensors, 16 solver position iterations) and ScenePreparation
with clearance 0, exactly as collection would start an episode. Run 0 is the
nominal start; runs 1..N-1 apply a seeded perturbation (±2 mm XY, ±2° yaw,
to the target; ±2 mm XY only to the bin, which stays at yaw 0) to the authored
start before preparation.

Every case gets a fresh environment (created and closed per case). Each case
report is written and flushed before teardown; judge from reports, not exit
codes. Per run it records poses every physics step during the hold, the
preparation report, and measured maxima against the frozen limits below.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2  # noqa: F401,E402 -- must precede Isaac
from isaaclab.app import AppLauncher  # noqa: E402

LIMITS = {  # proposed in the plan; frozen after the pilot
    "hold_s": 5.5, "stable_window_s": 0.5,
    "max_translation_m": 0.020, "max_tilt_deg": 15.0,
    "final_linear_m_s": 0.01, "final_angular_rad_s": 0.5,
    "window_drift_m": 0.001, "window_drift_deg": 0.5,
    "max_yaw_drift_deg_handle": 10.0, "max_table_motion_m": 0.0005,
    "perturb_xy_m": 0.002, "perturb_yaw_deg": 2.0,
}

SUPPORT_FORCE_N = 0.001  # PreparationSettings.support_force_n

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--tasks", nargs="+", required=True)
parser.add_argument("--task-dirs", nargs="+", default=["single_pnp_v2_drafts"])
parser.add_argument("--runs", type=int, default=5, help="1 = nominal only; N = nominal + N-1 perturbed")
parser.add_argument("--seed", type=int, default=20261102)
parser.add_argument("--handle-tasks", nargs="*", default=[], help="Tasks that also get the yaw-drift limit")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.kit_args = (args.kit_args or "") + " --/app/fastShutdown=false"
app = AppLauncher(args).app


def host_info():
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return None
    import importlib.metadata as md
    import torch
    return {"host": platform.node(), "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "driver": run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
            "isaacsim": md.version("isaacsim"), "isaaclab": md.version("isaaclab"),
            "git_head": run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]), "kit_args": args.kit_args}


def quat_yaw(q):
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


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
    from robolab.core.scenes.placement import tilt_change_deg
    from robolab.core.task.task_index import resolve_selected_task
    from robolab.core.world.world_state import get_world

    robolab.constants.VERBOSE = False
    args.output.mkdir(parents=True, exist_ok=True)
    info = host_info()
    (args.output / f"meta_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{os.getpid()}.json").write_text(json.dumps({"argv": sys.argv, "limits": LIMITS, "host": info,
                                                       "qualify_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1))
    for task in args.tasks:
        for run in range(args.runs):
            out = args.output / f"{task}_run{run}.json"
            if out.exists():
                continue
            row = {"task": task, "run": run, "kind": "nominal" if run == 0 else "perturbed",
                   "limits": LIMITS, "host": info, "status": "error"}
            env = None
            started = time.time()
            try:
                options = build_parser().parse_args([
                    "--desktop", "--source", "tcp", "--device", args.device, "--control-hz", "15",
                    "--recording-profile", "compact", "--stepping-profile", "held-target",
                    "--contact-sensors", "required", "--contact-history-length", "0",
                    "--solver-position-iterations", "16", "--episode-length-s", "120",
                    "--max-attempt-steps", "1800", "--max-attempt-seconds", "120",
                    "--scene-settle-clearance-m", "0"])
                options.task, options.task_dirs = task, args.task_dirs
                options.task_resolution = resolve_selected_task(task, options.task_dirs)
                robolab.constants.set_output_dir(str((args.output / "isaac" / task).resolve()))
                register_omy_envs(task, task_dirs=options.task_dirs, resolution=options.task_resolution)
                cfg = parse_env_cfg(task + "OMYL100", device=args.device, num_envs=1)
                cfg.episode_length_s = 120
                configure_performance(cfg, options)
                configure_device(cfg, options)
                scene_path = Path(str(cfg.scene.scene.spawn.usd_path))
                row.update(scene=str(scene_path), scene_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest())
                env, cfg = create_env(cfg, policy="simple_pnp_v2_qualification", rendering_mode="performance",
                                      env_class=OmyEnv)
                row["physics"] = {"physics_dt": env.physics_dt, "step_dt": env.step_dt, "decimation": env.cfg.decimation,
                                  "solver_position_iterations": 16, "device": args.device}
                names = list(env.scene.rigid_objects)
                movable = [n for n in names if n != "table"]
                target = [n for n in movable if n != "grey_bin"][0]
                action = torch.tensor([[*[cfg.scene.robot.init_state.joint_pos[n] for n in ARM_JOINTS], 1.0]],
                                      device=env.device)
                world = get_world(env)
                with torch.inference_mode():
                    env.reset(seed=args.seed)
                    row["reset_seed"] = args.seed
                    perturbation = {}
                    if run > 0:
                        rng = np.random.default_rng([args.seed, run, int(hashlib.sha256(task.encode()).hexdigest()[:8], 16)])
                        for n in movable:
                            a = env.scene.rigid_objects[n]
                            pose = a.data.root_pose_w.clone()
                            dx, dy = rng.uniform(-LIMITS["perturb_xy_m"], LIMITS["perturb_xy_m"], 2)
                            dyaw = rng.uniform(-LIMITS["perturb_yaw_deg"], LIMITS["perturb_yaw_deg"])
                            if n == "grey_bin":
                                dyaw = 0.0  # the bin stays at yaw 0 (decided 2026-11-03); XY only
                            # Yaw about the root; XY offset in world. (Pivots sit <= 6 cm from the
                            # footprint centre, so the centre moves < 2 mm extra at 2 deg.)
                            ang = math.radians(dyaw)
                            pose[0, :2] += torch.tensor([dx, dy], device=pose.device, dtype=pose.dtype)
                            qz = torch.tensor([math.cos(ang / 2), 0, 0, math.sin(ang / 2)], device=pose.device, dtype=pose.dtype)
                            w1, x1, y1, z1 = qz; w2, x2, y2, z2 = pose[0, 3:]
                            pose[0, 3:] = torch.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                                                       w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])
                            a.write_root_pose_to_sim(pose)
                            perturbation[n] = {"dx_m": float(dx), "dy_m": float(dy), "dyaw_deg": float(dyaw)}
                        env.scene.write_data_to_sim(); env.sim.forward(); env.scene.update(dt=0.0)
                    row["perturbation"] = perturbation
                    prepare = ScenePreparation(env, options)
                    start = {n: env.scene.rigid_objects[n].data.root_pose_w[0].cpu().numpy().copy() for n in names}
                    robot = env.scene["robot"]
                    # Robot start clearance: no robot body origin within 5 cm of an object envelope.
                    body = robot.data.body_pos_w[0].cpu().numpy()
                    near = []
                    for n in movable:
                        corners = prepare._bounds(n)
                        lo, hi = corners.min(0) - 0.05, corners.max(0) + 0.05
                        inside = np.all((body >= lo) & (body <= hi), 1)
                        near += [f"{robot.body_names[i]} near {n}" for i in np.flatnonzero(inside)]
                    row["robot_start_proximity"] = near
                    trace = []

                    def sample(phase, t):
                        s = {"phase": phase, "t": t}
                        for n in names:
                            a = env.scene.rigid_objects[n]
                            s[n] = {"pose": a.data.root_pose_w[0].cpu().numpy().tolist(),
                                    "lin": float(torch.linalg.vector_norm(a.data.root_lin_vel_w[0])),
                                    "ang": float(torch.linalg.vector_norm(a.data.root_ang_vel_w[0]))}
                        return s

                    trace.append(sample("start", 0.0))
                    t = 0.0
                    while app.is_running():
                        result = prepare.advance()
                        t = result["elapsed_s"]
                        trace.append(sample("prep", t))
                        if result["status"] != "settling":
                            break
                    row["preparation"] = {k: v for k, v in result.items() if k not in ("bodies",)}
                    contacts = {}
                    if result["status"] == "ready":
                        steps = math.ceil(LIMITS["hold_s"] / env.physics_dt)
                        env.action_manager.process_action(action)
                        for i in range(steps):
                            env._sim_step_counter += 1
                            env.action_manager.apply_action()
                            env.scene.write_data_to_sim()
                            env.sim.step(render=False)
                            env.scene.update(dt=env.physics_dt)
                            t += env.physics_dt
                            trace.append(sample("hold", t))
                            if (i + 1) % env.cfg.decimation == 0:
                                env.action_manager.process_action(action)
                        world.clear_contact_cache()
                        for pair in ((target, "grey_bin"), (target, "table"), ("grey_bin", "table")):
                            try:
                                # Same threshold as ScenePreparation's support check: light
                                # utensils weigh < 0.1 N, the in_contact default.
                                contacts["/".join(pair)] = bool(world.in_contact(*pair, force_threshold=SUPPORT_FORCE_N,
                                                                                 env_id=0))
                            except Exception as error:  # report, do not guess
                                contacts["/".join(pair)] = f"unavailable: {type(error).__name__}: {error}"
                        for n in movable:
                            try:
                                contacts[f"{n}/gripper"] = bool(world.in_contact(n, "gripper", force_threshold=SUPPORT_FORCE_N,
                                                                                 env_id=0))
                            except Exception as error:
                                contacts[f"{n}/gripper"] = f"unavailable: {type(error).__name__}"
                    row["final_contacts"] = contacts
                row.update(evaluate(trace, start, names, target, contacts, result, task in args.handle_tasks,
                                    env.physics_dt, near, tilt_change_deg, np, env.cfg.decimation))
                row["trace"] = trace
            except Exception:
                row["error"] = traceback.format_exc()
                traceback.print_exc()
            finally:
                row["wall_s"] = time.time() - started
                tmp = out.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(row, f)
                    f.flush(); os.fsync(f.fileno())
                tmp.rename(out)
                print(f"{task} run{run}: {row.get('status')} {row.get('reasons', '')} ({row['wall_s']:.1f}s)", flush=True)
                if env is not None:
                    env.close()
    (args.output / "done").write_text("ok\n")


def evaluate(trace, start, names, target, contacts, prep, handle, dt, near, tilt_change_deg, np, lag):
    L = LIMITS
    reasons, maxima = [], {}
    if prep["status"] != "ready":
        return {"status": "failed", "reasons": ["preparation " + prep["status"] + ": " + "; ".join(prep.get("reasons", []))],
                "maxima": maxima}
    hold = [s for s in trace if s["phase"] == "hold"]
    if hold[-1]["t"] - hold[0]["t"] + dt < L["hold_s"] - 1e-6:
        reasons.append("hold shorter than required")
    window = [s for s in hold if s["t"] >= hold[-1]["t"] - L["stable_window_s"] + 1e-9]
    for n in names:
        s0 = np.array(start[n])
        pos = np.array([s[n]["pose"][:3] for s in trace]); quat = np.array([s[n]["pose"][3:] for s in trace])
        trans = float(np.linalg.norm(pos - s0[:3], axis=1).max())
        tilt = float(max(tilt_change_deg(s0[3:], q) for q in quat))
        yaw = float(max(abs((math.degrees(math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2)))
                             - math.degrees(math.atan2(2*(s0[3]*s0[6]+s0[4]*s0[5]), 1-2*(s0[5]**2+s0[6]**2))) + 180) % 360 - 180)
                        for q in quat))
        wpos = np.array([s[n]["pose"][:3] for s in window]); wq = np.array([s[n]["pose"][3:] for s in window])
        wdrift = float(np.linalg.norm(wpos - wpos[0], axis=1).max())
        wrot = float(max(tilt_change_deg(wq[0], q) for q in wq))
        wyaw = float(max(abs(((math.degrees(math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2)))
                               - math.degrees(math.atan2(2*(wq[0][0]*wq[0][3]+wq[0][1]*wq[0][2]), 1-2*(wq[0][2]**2+wq[0][3]**2)))
                               + 180) % 360 - 180)) for q in wq))
        # Speeds from poses one control period apart (decided 2026-11-03): PhysX can
        # report non-zero velocities for bodies whose poses do not change.
        from robolab.core.omy_teleop.preparation import rotation_angle
        lin = ang = 0.0
        for a, b in zip(window, window[lag:]):
            step = b["t"] - a["t"]
            lin = max(lin, float(np.linalg.norm(np.array(b[n]["pose"][:3]) - np.array(a[n]["pose"][:3]))) / step)
            ang = max(ang, rotation_angle(a[n]["pose"][3:], b[n]["pose"][3:]) / step)
        rep_lin = float(max(s[n]["lin"] for s in window)); rep_ang = float(max(s[n]["ang"] for s in window))
        peak_lin = float(max(s[n]["lin"] for s in trace)); peak_ang = float(max(s[n]["ang"] for s in trace))
        maxima[n] = {"translation_m": trans, "tilt_deg": tilt, "yaw_drift_deg": yaw, "window_drift_m": wdrift,
                     "window_rotation_deg": max(wrot, wyaw), "final_linear_m_s": lin, "final_angular_rad_s": ang,
                     "final_reported_linear_m_s": rep_lin, "final_reported_angular_rad_s": rep_ang,
                     "peak_reported_linear_m_s": peak_lin, "peak_reported_angular_rad_s": peak_ang,
                     "speed_source": f"pose deltas over {lag} physics steps"}
        if n == "table":
            if trans > L["max_table_motion_m"]:
                reasons.append(f"table moved {trans * 1000:.2f} mm (base-scene setup error)")
            continue
        if trans > L["max_translation_m"]:
            reasons.append(f"{n}: translated {trans * 1000:.1f} mm")
        if tilt > L["max_tilt_deg"]:
            reasons.append(f"{n}: tilted {tilt:.1f} deg")
        if lin > L["final_linear_m_s"] or ang > L["final_angular_rad_s"]:
            reasons.append(f"{n}: moving in final window ({lin:.3g} m/s, {ang:.3g} rad/s)")
        if wdrift > L["window_drift_m"] or max(wrot, wyaw) > L["window_drift_deg"]:
            reasons.append(f"{n}: drift in final window ({wdrift * 1000:.2f} mm, {max(wrot, wyaw):.2f} deg)")
        if handle and n == target and yaw > L["max_yaw_drift_deg_handle"]:
            reasons.append(f"{n}: yaw drift {yaw:.1f} deg > {L['max_yaw_drift_deg_handle']}")
    if contacts.get(f"{target}/grey_bin") is True:
        reasons.append("target in contact with the bin")
    for k in (f"{target}/table", "grey_bin/table"):
        if contacts.get(k) is not True:
            reasons.append(f"{k} support contact: {contacts.get(k)}")
    for k, v in contacts.items():
        if k.endswith("/gripper") and v is True:
            reasons.append(f"{k} contact")
    if near:
        reasons.append("robot start proximity: " + ", ".join(near))
    return {"status": "passed" if not reasons else "failed", "reasons": reasons, "maxima": maxima}


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
