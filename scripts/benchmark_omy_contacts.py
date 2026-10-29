#!/usr/bin/env python3
"""Diagnostic replay of native-15 OMY contacts; never opens a leader or XR.

Use a fresh process/output directory per backend/solver configuration. Default
timings include compact recording and task checks, but exclude rendering and XR.
--contacts adds per-substep contact readback and is for accuracy diagnostics only.
Saved states omit contact caches, so replay is not bitwise deterministic.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import time
import traceback
from pathlib import Path

import h5py
import numpy as np


def load_episode(path, episode, state_index=None):
    with h5py.File(path, "r") as file:
        metadata = json.loads(file["data"].attrs["robolab_omy_teleop"])
        if metadata["control_hz"] != 15 or metadata["gripper_encoding"] != "+1=open,-1=closed":
            raise ValueError("Requires native-15 signed OMY commands")
        group = file["data"][episode]
        state_group = group["initial_state" if state_index is None else "states"]
        state = {
            kind: {
                name: {
                    key: value[:] if state_index is None else value[state_index][None]
                    for key, value in body.items()
                }
                for name, body in assets.items()
            }
            for kind, assets in state_group.items()
        }
        actions = group["actions"][:]
    return metadata, state, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--episode", default="demo_1")
    parser.add_argument("--target", required=True, help="Rigid object name, e.g. bowl")
    parser.add_argument("--device", choices=("cpu", "cuda:0", "cuda:1"), required=True)
    parser.add_argument("--iterations", type=int, required=True,
                        help="Diagnostic position cap, 16 through 64 (including intermediate values)")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--physics-threads", type=int, default=2)
    parser.add_argument("--physics-hz", type=int, choices=(120, 240), default=120)
    parser.add_argument("--gpu-partitions", type=int, choices=(1, 2, 4, 8, 16, 32), default=8)
    parser.add_argument("--runtime-cap-test", action="store_true", help="Probe whether live scene-cap USD edits change actual step cost")
    parser.add_argument("--stepping-profile", choices=("ordinary", "held-target"), default="held-target")
    parser.add_argument("--state-index", type=int, help="Restore this post-action row instead of the initial state")
    parser.add_argument("--open", action="store_true", help="Hold arm targets at restored row and command open")
    parser.add_argument("--remove-target", action="store_true", help="Move target 3m away as a causal probe")
    parser.add_argument("--friction", type=float, help="Diagnostic robot/target static and dynamic friction")
    parser.add_argument("--gripper-speed", type=float, default=5.0)
    parser.add_argument("--gripper-stiffness", type=float)
    parser.add_argument("--gripper-damping", type=float)
    parser.add_argument("--gripper-effort", type=float)
    parser.add_argument("--gripper-armature", type=float)
    parser.add_argument("--contact-offset", type=float)
    parser.add_argument("--rest-offset", type=float)
    parser.add_argument("--mimic-frequency", type=float, help="Override every gripper mimic natural frequency in rad/s; zero makes it hard")
    parser.add_argument("--mimic-damping-ratio", type=float)
    parser.add_argument("--finger-collider", choices=("convexHull",), help="Replace finger decomposition with per-mesh hulls in memory")
    parser.add_argument("--finger-collider-scope", choices=("all", "pads", "structure"), default="all")
    parser.add_argument("--target-hull-limit", type=int, choices=(32, 64, 128, 256),
                        help="Diagnostic target convex-decomposition cap, applied before scene creation")
    parser.add_argument("--placement-cache", action="store_true", help="Diagnostic shared placement evaluation at each control boundary")
    parser.add_argument("--placement-cache-ab", action="store_true",
                        help="Alternate baseline/cache across repeats in ABBA BAAB order within one simulator")
    parser.add_argument("--placement-checks", choices=("independent", "shared"), default="independent",
                        help="Production placement-check implementation")
    parser.add_argument("--task-checks", choices=("native", "cpu"), default="native",
                        help="Production task/event evaluation backend")
    parser.add_argument("--production-finger-colliders", choices=("original", "hulls"), default="original",
                        help="Exercise the collection's built-in collider/mass-reference path")
    parser.add_argument("--recompute-arm-inertia", action="store_true", help="Diagnostic collider-derived arm inertia/COM, keeping authored masses")
    parser.add_argument("--mass-properties-from", type=Path,
                        help="Restore robot and target masses, COMs and full inertias from a reference summary to isolate geometry")
    parser.add_argument("--contacts", action="store_true", help="Read true contact separation every physics step (adds overhead)")
    parser.add_argument("--profile-python", action="store_true", help="Save cProfile data for replay loops (adds overhead; not throughput evidence)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1 or args.physics_threads < 0:
        parser.error("steps/repeats must be positive, threads nonnegative")
    if not 16 <= args.iterations <= 64:
        parser.error("iterations must be between 16 and 64")
    if args.open and args.state_index is None:
        parser.error("--open requires --state-index")
    metadata, initial, actions = load_episode(args.source, args.episode, args.state_index)
    if args.open:
        actions = np.repeat(actions[args.state_index][None], args.steps, axis=0)
        actions[:, -1] = 1.0
    elif args.state_index is not None:
        index = args.state_index % len(actions)
        actions = actions[index + 1:]
    actions = actions[:args.steps]
    if not len(actions):
        parser.error("No actions after requested state")
    if args.remove_target:
        initial["rigid_object"][args.target]["root_pose"][:, 0] += 3.0
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import cv2  # noqa: F401 -- precede Isaac Lab
    from isaaclab.app import AppLauncher

    gpu = args.device.split(":")[-1] if args.device.startswith("cuda") else "0"
    launcher = AppLauncher(
        headless=True, device=args.device, enable_cameras=False,
        kit_args=f"--/renderer/activeGpu={gpu} --/app/settings/persistent=false "
                 "--/renderer/multiGpu/enabled=false --/app/runLoops/main/rateLimitEnabled=false "
                 "--/plugins/carb.tasking.plugin/threadCount=8 "
                 "--/plugins/omni.tbb.globalcontrol/maxThreadCount=8 "
                 f"--/persistent/physics/numThreads={args.physics_threads}",
    )
    app = launcher.app
    env = None
    monitor = None
    status = 1
    try:
        import torch
        import omni.physics.tensors as physics_tensors
        from isaaclab.managers import DatasetExportMode
        from pxr import Gf, Usd, UsdPhysics

        import robolab.constants
        from robolab.core.environments.config import parse_env_cfg
        from robolab.core.environments.runtime import create_env
        from robolab.core.omy_teleop.config import configure_performance, register_omy_envs
        from robolab.core.omy_teleop.environment import OmyEnv
        from robolab.core.omy_teleop.options import build_parser
        from robolab.core.omy_teleop.performance import PerformanceMonitor
        from robolab.core.omy_teleop.runner import close_recording, finish_interrupted

        cache_stats = None
        cache_enabled = [True]
        if args.placement_cache or args.placement_cache_ab:
            from scripts.omy_optimization_probes import install_placement_cache
            cache_stats = install_placement_cache(enabled=lambda: cache_enabled[0])
        robolab.constants.RECORD_IMAGE_DATA = True
        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True
        robolab.constants.set_output_dir(str(args.output_dir.resolve()))
        postfix = "OMYContactBenchmark"
        register_omy_envs(metadata["task"], env_postfix=postfix, task_dirs=metadata["task_dirs"],
                         expected_source=metadata.get("task_source"))
        run_args = build_parser().parse_args([
            "--source", "tcp", "--control-hz", "15", "--record", "--save-failures",
            "--recording-profile", "compact",
            "--placement-checks", args.placement_checks,
            "--task-checks", args.task_checks,
            "--finger-colliders", args.production_finger_colliders,
            "--gpu-max-num-partitions", str(args.gpu_partitions),
            "--stepping-profile", args.stepping_profile, "--render-interval", "4",
            "--episode-length-s", str(max(120, args.steps / 15 + 10)),
        ])
        # Intermediate counts are diagnostic-only until qualified; retain the
        # production CLI's narrower choices and apply before PhysX creation.
        run_args.solver_position_iterations = args.iterations
        cfg = parse_env_cfg(metadata["task"] + postfix, device=args.device, num_envs=1, seed=0)
        cfg.episode_length_s = run_args.episode_length_s
        configure_performance(cfg, run_args)
        cfg.sim.dt = 1 / args.physics_hz
        cfg.decimation = args.physics_hz // 15
        cfg.sim.render_interval = args.physics_hz // 30
        gripper = cfg.scene.robot.actuators["gripper"]
        gripper.velocity_limit_sim = args.gripper_speed
        for source, target in (("gripper_stiffness", "stiffness"), ("gripper_damping", "damping"),
                               ("gripper_effort", "effort_limit_sim"), ("gripper_armature", "armature")):
            if getattr(args, source) is not None:
                setattr(gripper, target, getattr(args, source))
        if (args.mimic_frequency is not None or args.mimic_damping_ratio is not None
                or args.finger_collider or args.recompute_arm_inertia):
            original_spawn = cfg.scene.robot.spawn.func

            def spawn_probe_robot(*spawn_args, **spawn_kwargs):
                prim = original_spawn(*spawn_args, **spawn_kwargs)
                for child in Usd.PrimRange(prim):
                    if args.recompute_arm_inertia and child.GetName() in {f"panda_link{i}" for i in range(1, 8)}:
                        mass = UsdPhysics.MassAPI.Apply(child)
                        mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(0))
                        mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(float("-inf")))
                    if args.finger_collider and "_inner_finger/" in str(child.GetPath()) and child.HasAPI(UsdPhysics.CollisionAPI):
                        pad = "fingertips" in child.GetName()
                        if args.finger_collider_scope == "all" or (pad == (args.finger_collider_scope == "pads")):
                            UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr().Set(args.finger_collider)
                    for schema in child.GetAppliedSchemas():
                        if not schema.startswith("PhysxMimicJointAPI:"):
                            continue
                        prefix = "physxMimicJoint:" + schema.split(":", 1)[1] + ":"
                        if args.mimic_frequency is not None:
                            child.GetAttribute(prefix + "naturalFrequency").Set(args.mimic_frequency)
                        if args.mimic_damping_ratio is not None:
                            child.GetAttribute(prefix + "dampingRatio").Set(args.mimic_damping_ratio)
                return prim

            cfg.scene.robot.spawn.func = spawn_probe_robot
        if args.target_hull_limit is not None:
            original_scene_spawn = cfg.scene.scene.spawn.func

            def spawn_probe_scene(*spawn_args, **spawn_kwargs):
                prim = original_scene_spawn(*spawn_args, **spawn_kwargs)
                target_prim = prim.GetChild(args.target)
                if not target_prim:
                    raise RuntimeError(f"Target {args.target} missing from spawned scene")
                changed = 0
                for child in Usd.PrimRange(target_prim):
                    attr = child.GetAttribute("physxConvexDecompositionCollision:maxConvexHulls")
                    if attr and child.HasAPI(UsdPhysics.CollisionAPI):
                        attr.Set(args.target_hull_limit)
                        changed += 1
                if not changed:
                    raise RuntimeError("Target has no authored convex decomposition hull cap")
                return prim

            cfg.scene.scene.spawn.func = spawn_probe_scene
        env, _ = create_env(cfg, env_class=OmyEnv, policy="omy_joint_position", rendering_mode="performance")
        robot, target = env.scene["robot"], env.scene[args.target]
        if args.mass_properties_from is not None:
            reference = json.loads(args.mass_properties_from.read_text())
            if reference["body_names"] != list(robot.body_names) or reference["arguments"]["target"] != args.target:
                raise ValueError("Mass property reference has different body identities")
            for obj, properties in ((robot, reference["body_properties"]),
                                    (target, reference["target_body_properties"])):
                for key in ("masses", "coms", "inertias"):
                    current = getattr(obj.root_physx_view, f"get_{key}")()
                    value = torch.as_tensor(properties[key], dtype=current.dtype, device=current.device).unsqueeze(0)
                    if value.shape != current.shape:
                        raise ValueError(f"Mass property shape mismatch for {key}")
                    getattr(obj.root_physx_view, f"set_{key}")(value, torch.tensor([0], dtype=torch.int32))
        for obj in (robot, target):
            for value, get_name, set_name in (
                (args.contact_offset, "get_contact_offsets", "set_contact_offsets"),
                (args.rest_offset, "get_rest_offsets", "set_rest_offsets"),
            ):
                if value is not None:
                    offsets = getattr(obj.root_physx_view, get_name)()
                    getattr(obj.root_physx_view, set_name)(
                        torch.full_like(offsets, value), torch.tensor([0], dtype=torch.int32))
        recorder = env.recorder_manager
        recorder.set_buffered_recording()
        recorder.cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        recorder.set_hdf5_compression(None)
        recorder.set_flush_interval(0)
        recorder.set_flush_memory_cleanup(False)
        recorder.set_hdf5_file("diagnostics")
        recorder.set_dataset_attrs({"training_ready": False, "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest()})
        if args.friction is not None:
            for obj in (robot, target):
                material = obj.root_physx_view.get_material_properties()
                material[..., :2] = args.friction
                obj.root_physx_view.set_material_properties(material, torch.tensor([0], dtype=torch.int32))
        stage = env.scene.stage
        audit = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0")):
            if not (prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdPhysics.Joint)):
                continue
            attrs = {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                     if any(s in a.GetName().lower() for s in ("offset", "approximation", "stiffness", "damping", "maxforce", "maxjointvelocity", "ccd", "naturalfrequency", "armature", "hull", "shrink", "errorpercentage"))}
            audit.append({"prim": str(prim.GetPath()), "attributes": attrs})
        (args.output_dir / "collider_drive_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        names = list(robot.joint_names)
        finger = names.index("finger_joint")
        linked = [names.index(n) for n in ("right_outer_knuckle_joint", "left_inner_finger_joint",
                  "right_inner_finger_joint", "left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint")]
        summary = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   "source_metadata": metadata, "versions": {n: importlib.metadata.version(n) for n in ("isaaclab", "isaacsim", "torch")},
                   "physics_device": str(env.device), "physics_dt": env.physics_dt, "decimation": cfg.decimation,
                   "capabilities": cfg.omy_capabilities, "joint_names": names, "body_names": robot.body_names,
                   "production_optimizations": {"placement_checks": args.placement_checks,
                                                "task_checks_effective": cfg.omy_task_checks_effective,
                                                "task_checks_fallback_reasons": cfg.omy_task_checks_fallback_reasons,
                                                "finger_colliders": cfg.omy_finger_collider_metadata},
                   "gripper": {"speed": robot.data.joint_vel_limits[0, finger].item(),
                               "stiffness": robot.data.joint_stiffness[0, finger].item(),
                               "damping": robot.data.joint_damping[0, finger].item(),
                               "effort": robot.data.joint_effort_limits[0, finger].item()},
                   "joint_properties": {key: value[0].cpu().tolist() for key, value in (
                       ("stiffness", robot.data.joint_stiffness), ("damping", robot.data.joint_damping),
                       ("effort_limits", robot.data.joint_effort_limits), ("velocity_limits", robot.data.joint_vel_limits))},
                   "body_properties": {"masses": robot.root_physx_view.get_masses()[0].cpu().tolist(),
                                       "coms": robot.root_physx_view.get_coms()[0].cpu().tolist(),
                                       "inertias": robot.root_physx_view.get_inertias()[0].cpu().tolist()},
                   "target_body_properties": {key: getattr(target.root_physx_view, f"get_{key}")()[0].cpu().tolist()
                                              for key in ("masses", "coms", "inertias")},
                   "collision_offsets": {
                       name: {"contact": obj.root_physx_view.get_contact_offsets().cpu().tolist(),
                              "rest": obj.root_physx_view.get_rest_offsets().cpu().tolist()}
                       for name, obj in (("robot", robot), (args.target, target))},
                   "timing_note": "Headless, no XR/rendering. Contacts and Python profiling runs include diagnostic overhead. First 20 commands excluded from steady timings. Cold restores omit contact solver caches.",
                   "runs": []}
        sim_view = contact_view = None
        contacts = []
        original_step = env.sim.step
        if args.contacts:
            sim_view = physics_tensors.create_simulation_view("torch")
            sim_view.set_subspace_roots("/")
            fingers = ["/World/envs/env_0/robot/Gripper/Robotiq_2F_85/" + side + "_inner_finger"
                       for side in ("left", "right")]
            contact_view = sim_view.create_rigid_contact_view(
                fingers, filter_patterns=[[target.root_physx_view.prim_paths[0]]] * len(fingers),
                max_contact_data_count=8192)

            def sample_contacts(*step_args, **step_kwargs):
                result = original_step(*step_args, **step_kwargs)
                force, points, normals, distances, counts, starts = contact_view.get_contact_data(dt=env.physics_dt)
                counts, starts = counts.cpu().numpy().ravel(), starts.cpu().numpy().ravel()
                distances = distances.cpu().numpy().ravel()
                active = [distances[s:s + n] for s, n in zip(starts, counts) if n]
                value = np.concatenate(active) if active else np.array([])
                contacts.append((float(value.min()) if len(value) else np.nan, len(value)))
                return result

            env.sim.step = sample_contacts
            summary["contact_view_sensors"] = contact_view.sensor_count
        monitor = PerformanceMonitor(capacity=max(len(actions), 300))
        monitor.install(env)
        with torch.inference_mode():
            # Initialize lazy buffers/kernels before restoring the source state.
            home = torch.tensor([[*metadata["performance"]["initial_joint_pos_rad"], 1.0]], device=env.device)
            for _ in range(20):
                env.step(home)
                if env.all_terminated:
                    break
            if args.runtime_cap_test:
                cap_attr = stage.GetPrimAtPath(cfg.sim.physics_prim_path).GetAttribute("physxScene:maxPositionIterationCount")
                summary["runtime_cap_test"] = []
                for cap in (16, 32, 64, 32, 16):
                    cap_attr.Set(cap)
                    elapsed = []
                    for _ in range(25):
                        started = time.perf_counter()
                        env.step(home)
                        elapsed.append(1000 * (time.perf_counter() - started))
                    summary["runtime_cap_test"].append({"authored_cap": cap,
                        "mean_ms_last20": float(np.mean(elapsed[-20:])),
                        "p95_ms_last20": float(np.percentile(elapsed[-20:], 95))})
                cap_attr.Set(args.iterations)
                print("RUNTIME_CAP_TEST " + json.dumps(summary["runtime_cap_test"]), flush=True)
            for repeat in range(args.repeats):
                if args.placement_cache_ab:
                    cache_enabled[0] = (False, True, True, False, True, False, False, True)[repeat % 8]
                env.reset(seed=0)
                state = env.scene.get_state(is_relative=True)
                for kind, assets in initial.items():
                    for name, body in assets.items():
                        for key, value in body.items():
                            state[kind][name][key] = torch.as_tensor(value, device=env.device, dtype=state[kind][name][key].dtype)
                env.reset_to(state, env_ids=None, is_relative=True)
                recorder.set_episode_index(repeat, [0])
                recorder.set_episode_seed(0, [0])
                monitor.reset_window()
                contacts.clear()
                q, v, poses, bodies = [], [], [], []
                cache_before = dict(cache_stats) if cache_stats is not None else {}
                commands = torch.as_tensor(actions, device=env.device)
                profiler = None
                if args.profile_python:
                    import cProfile
                    profiler = cProfile.Profile()
                    profiler.enable()
                for i, action in enumerate(commands):
                    monitor.begin_step()
                    env.step(action.unsqueeze(0))
                    monitor.end_step(work_end=time.perf_counter())
                    q.append(robot.data.joint_pos[0].clone())
                    v.append(robot.data.joint_vel[0].clone())
                    poses.append(target.data.root_pose_w[0].clone())
                    bodies.append(robot.data.body_pos_w[0].clone())
                    if env.all_terminated:
                        break
                if profiler is not None:
                    profiler.disable()
                    profiler.dump_stats(str(args.output_dir / f"repeat{repeat}.pstats"))
                all_performance = monitor.summary(env.step_dt)
                retained = list(monitor.samples)[20:]
                monitor.samples.clear()
                monitor.samples.extend(retained)
                performance = monitor.summary(env.step_dt)
                q, v, poses, bodies = (torch.stack(x).cpu().numpy() for x in (q, v, poses, bodies))
                if not all(np.isfinite(x).all() for x in (q, v, poses, bodies)):
                    raise RuntimeError("Nonfinite simulation state")
                error = np.rad2deg(np.abs(q[:, linked] - q[:, finger, None] * np.array([1, -1, 1, -1, -1]))).max(axis=1)
                closed = actions[:len(q), -1] < 0
                contact_array = np.asarray(contacts)
                result = {"repeat": repeat, "commands": len(q), "performance": performance,
                          "performance_all": all_performance, "final_finger_rad": float(q[-1, finger]),
                          "closed_linkage_max_deg": float(error[closed].max()) if closed.any() else None,
                          "closed_linkage_p95_deg": float(np.percentile(error[closed], 95)) if closed.any() else None,
                          "target_z_range_m": [float(poses[:, 2].min()), float(poses[:, 2].max())],
                          "final_target_pose": poses[-1].tolist(),
                          "task_terminated": bool(env.all_terminated),
                          "task_success": bool(env.all_terminated and env.get_env_results()[0]["success"])}
                if cache_stats is not None:
                    result["placement_cache"] = {key: value - cache_before[key] for key, value in cache_stats.items()}
                    result["placement_cache_enabled"] = cache_enabled[0]
                result["task_events"] = {name: copy.deepcopy(term._events) for name, term in recorder._terms.items()
                                         if hasattr(term, "_events")}
                (args.output_dir / f"performance{repeat}.json").write_text(
                    json.dumps(list(monitor.samples), indent=2) + "\n")
                if args.contacts:
                    minima = contact_array[:, 0]
                    valid = minima[np.isfinite(minima)]
                    result["contacts"] = {"substeps": len(minima), "contact_substeps": len(valid),
                        "max_penetration_m": float(max(0, -valid.min())) if len(valid) else None,
                        "p95_penetration_m": float(np.percentile(np.maximum(0, -valid), 95)) if len(valid) else None}
                np.savez_compressed(args.output_dir / f"repeat{repeat}.npz", joint_position=q,
                    joint_velocity=v, object_pose=poses, body_position=bodies, actions=actions[:len(q)],
                    linkage_error_deg=error, contacts=contact_array,
                    step_work_ms=[s["work"] * 1000 for s in retained])
                summary["runs"].append(result)
                (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print("CONTACT_RESULT " + json.dumps(result), flush=True)
                finish_interrupted(env, True)
        if monitor:
            monitor.close()
        env.sim.step = original_step
        close_recording(env)
        status = 0
    except BaseException:
        (args.output_dir / "failure.json").write_text(json.dumps({
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "error": traceback.format_exc(),
        }, indent=2) + "\n")
        traceback.print_exc()
        raise
    finally:
        if monitor:
            monitor.close()
        if env is not None:
            env.close()
        app.app.post_quit(status)
        app.close()


if __name__ == "__main__":
    main()
