#!/usr/bin/env python3
"""Diagnostic scene trials with DROID/OMY; never writes demonstrations or library scenes.

Run with --host-gpu N outside a device-restricted sandbox. Each invocation uses
one fresh simulator process. Reports distinguish start physics, semantics, render,
and the limited set of attempted robot grasps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
import faulthandler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", type=Path, required=True)
parser.add_argument("--kind", choices=("pail", "stack"), required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--host-gpu", type=int)
parser.add_argument("--stability-only", action="store_true")
parser.add_argument("--grasp-only", action="store_true")
parser.add_argument("--grasps", type=int, default=8)
parser.add_argument("--candidate", type=int, help="Repeat only this grasp candidate (0..7)")
parser.add_argument("--grasp-depth", type=float, help="Fingertip depth below the object's top, meters")
parser.add_argument("--transfer", action="store_true", help="After a pail lift, test transport, held rejection, and release")
parser.add_argument("--stabilization", action="store_true", help="Compare an explicitly different PhysX stabilization profile")
args = parser.parse_args()
if args.grasps < 0 or args.grasps > 8 or args.candidate is not None and not 0 <= args.candidate <= 7:
    parser.error("Use 0..8 grasps and candidate 0..7")
if args.host_gpu is not None:
    import subprocess
    import os
    import signal
    from robolab.core.omy_postprocess.gpu_isolation import isolated_gpu_launch
    from robolab.core.omy_postprocess.render import _gpu_slot
    gpu = isolated_gpu_launch(f"cuda:{args.host_gpu}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gpu.json").write_text(json.dumps(gpu.metadata, indent=2))
    argv = sys.argv[1:]
    index = argv.index("--host-gpu")
    del argv[index:index + 2]
    with _gpu_slot(f"cuda:{args.host_gpu}"):
        child = subprocess.Popen([*gpu.prefix, sys.executable, "-u", "-B", __file__, *argv], env=gpu.env, start_new_session=True)
        try:
            status = child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
            raise SystemExit("Scene trial exceeded the 600-second wall-clock limit")
        raise SystemExit(status)

faulthandler.enable()
faulthandler.dump_traceback_later(120, repeat=True)

import cv2  # noqa: E402 -- before Isaac
from isaaclab.app import AppLauncher  # noqa: E402

launcher = AppLauncher(headless=True, device="cuda:0", enable_cameras=True,
    kit_args="--/app/settings/persistent=false --/renderer/multiGpu/enabled=false "
             "--/plugins/carb.tasking.plugin/threadCount=8 "
             "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8 "
             "--/persistent/physics/numThreads=2 --/app/fastShutdown=false")


def main():
    import math
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils, Gf
    import omni.physics.tensors as physics_tensors
    import omni.usd
    import robolab.constants as constants
    from robolab.core.environments.factory import create_env_cfg
    from robolab.core.environments.runtime import create_env
    from robolab.core.observations.observation_utils import generate_obs_cfg
    from robolab.core.omy_teleop.config import (OmyDroidCfg, OmyDroidActionsCfg,
        OmyOperatorCamerasCfg, configure_performance)
    from robolab.core.omy_teleop.environment import OmyEnv
    from robolab.core.omy_teleop.collection import build_parser
    from robolab.core.omy_teleop.options import ARM_JOINTS
    from robolab.core.world.world_state import get_world
    from robolab.core.task.conditionals import object_resting_in_container
    from robolab.core.events.subtask_recorder import SubtaskCompletionRecorderTerm
    from robolab.robots.droid import ProprioceptionObservationCfg, contact_gripper
    from robolab.variations.lighting import SphereLightCfg
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from tools.audit_simple_pnp_scenes import layer_hashes

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.scene.resolve()
    scene = out / source.name
    layer = Sdf.Layer.CreateAnonymous()
    layer.TransferContent(Sdf.Layer.FindOrOpen(str(source)))
    def rebase(path):
        return str((source.parent / path).resolve()) if path and not ":" in path and not Path(path).is_absolute() else path
    UsdUtils.ModifyAssetPaths(layer, rebase)
    layer.Export(str(scene))
    stage = Usd.Stage.Open(str(scene))
    target = "plasticpail_a01" if args.kind == "pail" else "blue_block"
    names = [target, "grey_bin"] if args.kind == "pail" else ["red_block", "blue_block"]
    supports = {name: ("red_block" if name == "blue_block" else "table") for name in names}
    report = {"diagnostic_only": True, "source": str(source), "scene": str(scene),
        "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "layers": layer_hashes(stage), "kind": args.kind,
        "profile": {"robot": "DROID/OMY", "physics_hz": 120, "control_hz": 15,
            "solver_position_cap": 16, "enable_stabilization": args.stabilization,
            "contacts": "full diagnostic", "reset_seeds": [0, 1, 2],
            "seconds_per_reset": 5, "stable_window_s": .5,
            "limits": {"translation_m": .02, "tilt_deg": 15, "linear_m_s": .01,
                "angular_rad_s": .5, "window_translation_m": .001, "window_rotation_deg": .5}},
        "physics": {"status": "pending", "trials": []}, "grasp": {"status": "pending", "trials": []},
        "semantics": {"status": "pending"}, "render": {"status": "pending"}}
    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2))
    save()
    task = out / "qualification_task.py"
    task.write_text('''from dataclasses import dataclass
import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from robolab.core.scenes.utils import import_scene
from robolab.core.task.task import Task
from robolab.core.task.conditionals import object_resting_in_container, pick_and_place, stacked, object_dropped
def released_stack(env):
    return stacked(env, objects=["red_block", "blue_block"], order="bottom_to_top") & object_dropped(env, object=["red_block", "blue_block"])
PARAMS = dict(object="plasticpail_a01", container="grey_bin", gripper_name="gripper", exclude_contact_with="table", settle_time_s=.5, linear_threshold=.05, angular_threshold=.5, position_tolerance=.001, rotation_tolerance_deg=.5)
@configclass
class Terminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=''' + ('object_resting_in_container, params=PARAMS' if args.kind == "pail" else 'released_stack') + ''')
@dataclass
class QualificationTask(Task):
    contact_object_list = ''' + repr(names + ["table"]) + '''
    scene = import_scene(''' + repr(str(scene)) + ''', contact_object_list)
    terminations = Terminations
    instruction = {"default": "Diagnostic qualification fixture"}
    episode_length_s: int = 120
    subtasks = ''' + ('[pick_and_place(object=["plasticpail_a01"], container="grey_bin", logical="all", score=1., placement_func=object_resting_in_container, placement_params={k:v for k,v in PARAMS.items() if k not in ("object", "container")})]' if args.kind == "pail" else 'None') + '\n')
    constants.VERBOSE = False
    constants.DEBUG = False
    constants.RECORD_IMAGE_DATA = False
    constants.set_output_dir(str(out))
    cfg_class = create_env_cfg(str(task), task_dir=str(out),
        observations_cfg=generate_obs_cfg({"proprio_obs": ProprioceptionObservationCfg()})(),
        actions_cfg=OmyDroidActionsCfg(), robot_cfg=OmyDroidCfg, camera_cfg=[OmyOperatorCamerasCfg],
        lighting_cfg=SphereLightCfg, background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper, dt=1/120, decimation=8, render_interval=8, seed=0)
    cfg = cfg_class()
    cfg.sim.device = "cuda:0"
    cfg.scene.num_envs = 1
    options = build_parser().parse_args(["--desktop", "--source", "tcp", "--control-hz", "15",
        "--recording-profile", "compact", "--stepping-profile", "held-target", "--contact-sensors", "full",
        "--contact-history-length", "0", "--solver-position-iterations", "16"])
    configure_performance(cfg, options)
    cfg.sim.physx.enable_stabilization = args.stabilization
    env = None
    try:
        env, cfg = create_env(cfg, policy="scene_qualification_diagnostic", rendering_mode="performance", env_class=OmyEnv)
        world = get_world(env)
        robot = env.scene["robot"]
        home = torch.tensor([[*[cfg.scene.robot.init_state.joint_pos[n] for n in ARM_JOINTS], 1.]], device=env.device)
        sim_view = physics_tensors.create_simulation_view("torch")
        sim_view.set_subspace_roots("/")
        robot_paths = [str(p.GetPath()) for p in omni.usd.get_context().get_stage().Traverse()
                       if str(p.GetPath()).startswith("/World/envs/env_0/robot/") and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        filter_paths = [str(world._get_prim(n, env_id=0).GetPath()) for n in names + ["table"]]
        contact = sim_view.create_rigid_contact_view(robot_paths, filter_patterns=[filter_paths] * len(robot_paths), max_contact_data_count=8192)
        report["robot_contact_bodies"] = robot_paths
        report["robot_body_names"] = robot.body_names
        report["contact_filters"] = names + ["table"]
        micro = 0
        last_force_peaks = np.zeros((len(robot_paths), len(filter_paths)))
        def advance(action=home, steps=8):
            nonlocal micro, last_force_peaks
            last_force_peaks = np.zeros((len(robot_paths), len(filter_paths)))
            env.action_manager.process_action(action)
            for _ in range(steps):
                env._sim_step_counter += 1
                env.action_manager.apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)
                last_force_peaks = np.maximum(last_force_peaks, np.linalg.norm(force_matrix(), axis=-1))
                micro += 1
                if micro % cfg.decimation == 0:
                    env.episode_length_buf += 1
                    env.common_step_counter += 1
            world.clear_contact_cache()
        def reset(seed=0):
            nonlocal micro
            env.reset(seed=seed)
            micro = 0
            world.reset_predicate_state([0])
            world.clear_contact_cache()
        def state(name):
            body = env.scene[name]
            return np.r_[body.data.root_pose_w[0].cpu().numpy(), body.data.root_vel_w[0].cpu().numpy()]
        def force_matrix():
            return contact.get_contact_force_matrix(dt=env.physics_dt).cpu().numpy()
        def snapshot(label):
            camera = env.scene.sensors["view_cam"]
            camera.set_world_poses_from_view(
                eyes=torch.tensor([[.9,-1.1,1.0]], device=env.device),
                targets=torch.tensor([[.52,-.08,.08]], device=env.device))
            for _ in range(24):
                env.sim.render()
            camera.update(env.physics_dt, force_recompute=True)
            rgb = camera.data.output["rgb"][0].cpu().numpy()[:, :, :3]
            cv2.imwrite(str(out / (label + ".png")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            report["render"].update(status="rendered_review_required", last_image=label + ".png",
                                    pixel_std=float(rgb.std()))
        def angular(q1, q2):
            dots = np.abs(np.sum(q1 * q2, axis=-1))
            return np.rad2deg(2 * np.arccos(np.clip(dots, 0, 1)))
        if not args.grasp_only:
            for seed in (0, 1, 2):
                reset(seed)
                initial = {n: state(n) for n in names}
                trace = {n: [] for n in names}
                supported = {n: [] for n in names}
                max_robot_force = 0.
                unwanted_force = 0.
                for step in range(600):
                    advance(steps=1)
                    max_robot_force = max(max_robot_force, float(np.linalg.norm(force_matrix(), axis=-1).max(initial=0)))
                    for name in names:
                        trace[name].append(state(name))
                        if step >= 540:
                            supported[name].append(bool(world.is_supported_on_surface(name, supports[name], force_threshold=.01, env_id=0)))
                    if args.kind == "pail":
                        unwanted_force = max(unwanted_force, float(world.get_contact_force(target, "grey_bin").abs().max()))
                row = {"seed": seed, "bodies": {}, "robot_contact_force_max_n": max_robot_force,
                       "unintended_object_contact_force_max_n": unwanted_force}
                for name in names:
                    data = np.asarray(trace[name]); last = data[-60:]
                    r0 = Rotation.from_quat(initial[name][[4, 5, 6, 3]])
                    r = Rotation.from_quat(data[:, [4, 5, 6, 3]])
                    tilt = np.rad2deg(np.arccos(np.clip(r.apply([0,0,1]) @ r0.apply([0,0,1]), -1, 1)))
                    metrics = dict(max_translation_m=float(np.linalg.norm(data[:, :3]-initial[name][:3], axis=1).max()),
                        max_tilt_deg=float(tilt.max()), final_linear_m_s=float(np.linalg.norm(last[:,7:10], axis=1).max()),
                        final_angular_rad_s=float(np.linalg.norm(last[:,10:13], axis=1).max()),
                        window_translation_m=float(np.linalg.norm(last[:,:3]-last[0,:3], axis=1).max()),
                        window_rotation_deg=float(angular(last[:,3:7], last[0,3:7]).max()),
                        supported_entire_final_window=all(supported[name]), final_pose=data[-1,:7].tolist())
                    metrics["passed"] = (metrics["max_translation_m"] <= .02 and metrics["max_tilt_deg"] <= 15
                        and metrics["final_linear_m_s"] <= .01 and metrics["final_angular_rad_s"] <= .5
                        and metrics["window_translation_m"] <= .001 and metrics["window_rotation_deg"] <= .5
                        and metrics["supported_entire_final_window"])
                    row["bodies"][name] = metrics
                row["passed"] = all(v["passed"] for v in row["bodies"].values()) and max_robot_force < .01 and unwanted_force < .01
                report["physics"]["trials"].append(row)
                np.savez_compressed(out / f"stability_seed{seed}.npz", **{n: np.asarray(v) for n,v in trace.items()})
                print("PHYSICS " + json.dumps(row), flush=True)
                save()
            report["physics"]["status"] = "passed" if all(r["passed"] for r in report["physics"]["trials"]) else "failed"
            snapshot("settled")
            settled = out / (scene.stem + ".settled.usda")
            stage.GetRootLayer().Export(str(settled))
            saved_stage = Usd.Stage.Open(str(settled))
            for name in names:
                pose = state(name)[:7]
                prim = saved_stage.GetPrimAtPath("/world/" + name)
                for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        op.Set(Gf.Vec3d(*map(float, pose[:3])))
                    elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                        quaternion = Gf.Quatf if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat else Gf.Quatd
                        op.Set(quaternion(float(pose[3]), *map(float, pose[4:7])))
            saved_stage.GetRootLayer().Save()
            report["settled_scene"] = str(settled)
            save()
        if args.stability_only:
            return

        def teleport(name, pose):
            body = env.scene[name]
            body.write_root_pose_to_sim(torch.tensor([pose], dtype=torch.float32, device=env.device))
            body.write_root_velocity_to_sim(torch.zeros((1,6), device=env.device))
            env.scene.write_data_to_sim(); env.sim.forward(); env.scene.update(0.)
            world.clear_contact_cache()
        term = cfg.terminations.success
        def success():
            return bool(term.func(env, **term.params)[0])
        if not args.grasp_only:
            semantic_rows = []
            for case in ("initial", "airborne", "released_drop", "reset_timer"):
                reset()
                for _ in range(15): advance()
                pose = state(target)[:7].copy()
                if case != "initial":
                    if args.kind == "pail":
                        _, center = world.get_bbox("grey_bin")
                        pose[:2] = center[0,:2].cpu().numpy()
                        pose[2] = .17 if case != "airborne" else .5
                    else:
                        pose[2] += .05 if case != "airborne" else .4
                    teleport(target, pose)
                    world.reset_predicate_state([0])
                values = []; times = []; scores = []
                recorder = next((r for r in env.recorder_manager._terms.values() if isinstance(r, SubtaskCompletionRecorderTerm)), None)
                for step in range(75):
                    if case == "airborne": teleport(target, pose)
                    advance()
                    value = success(); values.append(value)
                    times.append((step+1) * env.step_dt)
                    if recorder is not None:
                        recorder.record_post_step(); scores.append(recorder.infos[0]["score"])
                expected = case in ("released_drop", "reset_timer") or (args.kind == "stack" and case == "initial")
                row = dict(case=case, expected=expected, final=values[-1], ever=any(values),
                    first_true_s=next((t for t,v in zip(times,values) if v), None),
                    passed=(all(values[-8:]) if expected else not any(values)), final_pose=state(target)[:7].tolist(),
                    final_score=scores[-1] if scores else None)
                if args.kind == "pail" and expected:
                    row["passed"] &= row["first_true_s"] is not None and row["first_true_s"] >= .5
                    row["score_agrees"] = all((s == 1.) == v for s,v in zip(scores,values))
                    row["passed"] &= row["score_agrees"]
                semantic_rows.append(row)
                snapshot("semantics_" + case)
                print("SEMANTICS " + json.dumps(row), flush=True)
            report["semantics"] = {"status": "passed" if all(r["passed"] for r in semantic_rows) else "failed",
                "trials": semantic_rows, "scope": "physical drop/airborne controls; stack initial is intentionally already stacked",
                "release_mapping": dict(contact_gripper)}
            save()

        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=1, device=env.device)
        arm_ids, _ = robot.find_joints(list(ARM_JOINTS), preserve_order=True)
        ee = robot.find_bodies("base_link")[0][0]
        report["robot_kinematics"] = {"fixed_base": robot.is_fixed_base, "ee_index": ee,
            "joint_indices": arm_ids, "jacobian_shape": list(robot.root_physx_view.get_jacobians().shape)}
        target_index = names.index(target)
        finger_rows = [i for i,p in enumerate(robot_paths) if p.endswith(("left_inner_finger", "right_inner_finger"))]
        report["grasp"]["finger_contact_paths"] = [robot_paths[i] for i in finger_rows]
        report["grasp"]["controller"] = "absolute pose DLS IK, base_link, OMY joint targets; scripted candidates"
        def move(position, quat, seconds, closed=False):
            start = robot.data.body_pos_w[0,ee].clone()
            end = torch.tensor(position, device=env.device, dtype=torch.float32)
            q = torch.tensor([quat], device=env.device, dtype=torch.float32)
            max_unintended = 0.; samples = []
            planned_steps = round(seconds/env.step_dt)
            for step in range(planned_steps + 45):
                fraction = min(1., (step+1)/max(1, planned_steps*.8))
                fraction = fraction*fraction*(3-2*fraction)
                desired = start + (end-start)*fraction
                ik.set_command(torch.cat((desired[None],q), dim=1))
                joints = robot.data.joint_pos[:,arm_ids]
                jac_ids = arm_ids if robot.is_fixed_base else [i+6 for i in arm_ids]
                jac = robot.root_physx_view.get_jacobians()[:,ee-1 if robot.is_fixed_base else ee,:,jac_ids]
                wanted = ik.compute(robot.data.body_pos_w[:,ee], robot.data.body_quat_w[:,ee], jac, joints)
                command = joints + (wanted-joints).clamp(-.04,.04)
                command = torch.maximum(torch.minimum(command, robot.data.soft_joint_pos_limits[:,arm_ids,1]), robot.data.soft_joint_pos_limits[:,arm_ids,0])
                action = torch.cat((command, torch.tensor([[-1. if closed else 1.]], device=env.device)), dim=1)
                advance(action)
                forces = np.linalg.norm(force_matrix(), axis=-1)
                unwanted = last_force_peaks.copy()
                unwanted[finger_rows, target_index] = 0
                max_unintended = max(max_unintended, float(unwanted.max(initial=0)))
                samples.append(dict(pose=state(target)[:7].tolist(), finger_forces=forces[finger_rows,target_index].tolist(),
                    ee=robot.data.body_pos_w[0,ee].cpu().tolist(), ee_quat=robot.data.body_quat_w[0,ee].cpu().tolist(),
                    joints=robot.data.joint_pos[0,arm_ids].cpu().tolist(), success=success(), unwanted_force_n=float(unwanted.max(initial=0))))
                if step+1 >= planned_steps:
                    position_error = float(torch.linalg.vector_norm(robot.data.body_pos_w[0,ee]-end))
                    orientation_error = float(angular(robot.data.body_quat_w[0,ee].cpu().numpy(), np.asarray(quat)))
                    if position_error < .005 and orientation_error < 3.:
                        break
            return dict(position_error_m=float(torch.linalg.vector_norm(robot.data.body_pos_w[0,ee]-end)),
                        orientation_error_deg=float(angular(robot.data.body_quat_w[0,ee].cpu().numpy(), np.asarray(quat))),
                        duration_s=len(samples)*env.step_dt, max_unintended_force_n=max_unintended, samples=samples)
        for attempt in range(args.grasps):
            reset(attempt)
            for _ in range(30): advance()
            initial = state(target)[:3].copy()
            corners, center = world.get_bbox(target)
            center = center[0].cpu().numpy(); top = float(corners[0,:,2].max())
            candidate = attempt if args.candidate is None else args.candidate
            yaw = (candidate % 4) * math.pi/2
            offset = np.zeros(2)
            if args.kind == "pail":
                # Gripper local Y is the jaw closing axis; pinch a wall radially.
                radius = .053 if candidate < 4 else .060
                offset = radius*np.array([-math.sin(yaw), math.cos(yaw)])
            depth = args.grasp_depth if args.grasp_depth is not None else (.015 if args.kind == "pail" else .018)
            finger_z = top - depth
            grasp = np.r_[center[:2] + offset, finger_z + .1628]
            # Inspect this USD, not the generic Robotiq frame convention: its
            # finger meshes extend along base_link +X, and jaws close along Y.
            q = (Rotation.from_euler("z",yaw) * Rotation.from_euler("y",math.pi/2)).as_quat()[[3,0,1,2]].tolist()
            pre = grasp + [0,0,.16]
            row = dict(attempt=attempt, candidate=candidate, depth_m=depth, yaw_rad=yaw, grasp_base_position=grasp.tolist(), quaternion=q, phases={})
            row["phases"]["pregrasp"] = move(pre,q,5.)
            row["phases"]["approach"] = move(grasp,q,3.)
            snapshot(f"grasp_{attempt}_approach")
            row["phases"]["close"] = move(grasp,q,1.5,True)
            row["phases"]["lift"] = move(grasp+[0,0,.12],q,2.,True)
            row["phases"]["hold"] = move(grasp+[0,0,.12],q,1.,True)
            snapshot(f"grasp_{attempt}_hold")
            hold = row["phases"]["hold"]["samples"]
            heights = [s["pose"][2]-initial[2] for s in hold]
            row["minimum_hold_lift_m"] = min(heights)
            row["hold_drift_m"] = float(np.linalg.norm(np.asarray([s["pose"][:3] for s in hold])-hold[0]["pose"][:3], axis=1).max())
            row["both_finger_contact_fraction"] = float(np.mean([all(f>.01 for f in s["finger_forces"]) for s in hold]))
            row["max_unintended_force_n"] = max(p["max_unintended_force_n"] for p in row["phases"].values())
            row["passed"] = bool(min(heights) > .05 and row["hold_drift_m"] < .01
                and row["both_finger_contact_fraction"] >= .8 and row["max_unintended_force_n"] < .1
                and row["phases"]["pregrasp"]["position_error_m"] < .01
                and row["phases"]["approach"]["position_error_m"] < .01
                and row["phases"]["approach"]["orientation_error_deg"] < 5.)
            (out / f"grasp_{attempt}.json").write_text(json.dumps(row,indent=2))
            short = {k:v for k,v in row.items() if k != "phases"}
            report["grasp"]["trials"].append(short)
            print("GRASP " + json.dumps(short), flush=True)
            save()
            if row["passed"]:
                report["grasp"]["status"] = "passed"
                if args.transfer and args.kind == "pail":
                    _, destination = world.get_bbox("grey_bin")
                    desired = np.r_[destination[0,:2].cpu().numpy() + offset, grasp[2]+.18]
                    phases = {"raise": move(grasp+[0,0,.18],q,2.,True),
                              "transport": move(desired,q,6.,True)}
                    phases["lower_held"] = move(desired-[0,0,.10],q,3.,True)
                    snapshot("held_in_bin")
                    held = phases["lower_held"]["samples"][-15:]
                    # Seat the pail before opening: opening a loaded rim grasp
                    # in midair can leave one finger hooked under the lip.
                    phases["seat_before_release"] = move(desired-[0,0,.16],q,2.,True)
                    phases["release"] = move(desired-[0,0,.16],q,3.,False)
                    phases["retract"] = move(desired+[0,0,.06],q,2.,False)
                    snapshot("placed_in_bin")
                    released = phases["retract"]["samples"][-15:]
                    checks = dict(held_never_succeeded=not any(s["success"] for s in held),
                        held_contact_fraction=float(np.mean([any(f>.01 for f in s["finger_forces"]) for s in held])),
                        released_success=all(s["success"] for s in released),
                        released_no_finger_contact=all(all(f<.01 for f in s["finger_forces"]) for s in released),
                        max_unintended_force_n=max(p["max_unintended_force_n"] for p in phases.values()))
                    checks["status"] = "passed" if (checks["held_never_succeeded"] and checks["held_contact_fraction"]>=.8
                        and checks["released_success"] and checks["released_no_finger_contact"] and checks["max_unintended_force_n"]<.1) else "failed"
                    if checks["held_contact_fraction"] < .8:
                        checks["reason"] = "Object lost physical finger contact before commanded release; held-control case was not established"
                    (out / "transfer.json").write_text(json.dumps(dict(checks=checks,phases=phases),indent=2))
                    report["transport_and_release"] = checks
                    print("TRANSFER " + json.dumps(checks),flush=True)
                    save()
                break
        else:
            report["grasp"]["status"] = "failed" if args.grasps else "pending"
        save()
    except BaseException as exc:
        report["execution_error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
    finally:
        save()
        if env is not None:
            env.close()


exit_status = 0
try:
    main()
except BaseException:
    exit_status = 1
    traceback.print_exc()
finally:
    import os
    import threading
    # Kit can hang during teardown after a failed environment constructor.
    # All evidence is flushed before this cleanup; bound teardown independently.
    watchdog = threading.Timer(15, lambda: os._exit(exit_status))
    watchdog.daemon = True
    watchdog.start()
    launcher.app.close()
    watchdog.cancel()
raise SystemExit(exit_status)
