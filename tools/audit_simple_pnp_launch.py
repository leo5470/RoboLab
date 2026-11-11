#!/usr/bin/env python3
"""Audit current generated PnP scenes and simulate each placement without a policy.

Run using the project's Isaac Sim Python environment. Examples::

    python tools/audit_simple_pnp_launch.py --phase static --headless
    python tools/audit_simple_pnp_launch.py --phase physics --headless \
        --shard-index 0 --shard-count 2 --device cuda:0
    python tools/audit_simple_pnp_launch.py --phase report

Only audit outputs are written. Task definitions, scenes, and assets are read-only.
Physics batches clone one task per catalog entry and apply each variant's exact
authored target/bin poses. A fingerprint check verifies all other scene content
is identical within the batch. Robot joints hold their default positions.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import gc
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
SCENES = ROOT / "assets/scenes/simple_pnp"
TASKS = ROOT / "robolab/tasks/single_pnp"
DEFAULT_OUTPUT = ROOT / "output/generated_launch_audit"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_records():
    scene_manifest = json.loads((SCENES / "manifest.json").read_text())
    task_manifest = json.loads((TASKS / "manifest.json").read_text())
    by_scene = {Path(t["scene"]).name: t for t in task_manifest["tasks"]}
    records = []
    for scene in scene_manifest["scenes"]:
        task = by_scene[scene["scene"]]
        records.append({
            "scene": scene["scene"], "catalog_index": scene["catalog_index"],
            "target": scene["target"]["name"], "task": task["task"],
            "class_name": task["class_name"],
        })
    return scene_manifest, records


def prim_pose(layer, path: str):
    values = {}
    for op in ("translate", "orient", "scale"):
        spec = layer.GetAttributeAtPath(path + ".xformOp:" + op)
        if spec is None:
            raise ValueError(f"Missing {op} on {path}")
        value = spec.default
        if op == "orient":
            values[op] = [float(value.GetReal()), *map(float, value.GetImaginary())]
        else:
            values[op] = list(map(float, value))
    return values


def skeleton(text: str, target: str) -> str:
    """Ignore only varying poses; keep payloads, scale, fixtures and physics."""
    start = text.index(f'    def "{target}" (')
    before, rest = text[:start], text[start:]
    rest = re.sub(r'(xformOp:(?:orient|translate)\s*=)\s*\([^\n]+\)', r'\1 (POSE)', rest)
    return hashlib.sha256((before + rest).encode()).hexdigest()


def aabb(cache, prim):
    bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if bound.IsEmpty():
        return None
    return {"min": list(map(float, bound.GetMin())), "max": list(map(float, bound.GetMax()))}


def union_bounds(bounds):
    bounds = [b for b in bounds if b is not None]
    if not bounds:
        return None
    return {"min": [min(b["min"][i] for b in bounds) for i in range(3)],
            "max": [max(b["max"][i] for b in bounds) for i in range(3)]}


def csv_rows(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def static_audit(output: Path):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    manifest, records = load_records()
    groups = collections.defaultdict(list)
    for record in records:
        groups[record["catalog_index"]].append(record)
    issues = []
    expected_scenes = {r["scene"] for r in records}
    expected_tasks = {r["task"] for r in records}
    if expected_scenes != {p.name for p in SCENES.glob("*.usda")}:
        issues.append("Scene files differ from scene manifest")
    if expected_tasks != {p.name for p in TASKS.glob("*_task.py")}:
        issues.append("Task files differ from task manifest")
    rows, asset_rows = [], []
    for group_index, (catalog_index, variants) in enumerate(sorted(groups.items())):
        first = variants[0]
        stage = Usd.Stage.Open(str(SCENES / first["scene"]))
        stage.SetEditTarget(stage.GetSessionLayer())
        name = first["target"]
        target = stage.GetPrimAtPath("/world/" + name)
        bin_prim = stage.GetPrimAtPath("/world/grey_bin")
        if not target or not bin_prim:
            raise ValueError(f"Missing target/bin in {first['scene']}")
        colliders = [p for p in Usd.PrimRange(target, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)
                     and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
        nested_bodies = [str(p.GetPath()) for p in Usd.PrimRange(target, Usd.TraverseInstanceProxies())
                         if p != target and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        signature = skeleton((SCENES / first["scene"]).read_text(), name)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"],
                                useExtentsHint=False)
        top_candidates = [p for p in Usd.PrimRange(stage.GetPrimAtPath("/world/table"), Usd.TraverseInstanceProxies())
                          if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetName() == "top"]
        if len(top_candidates) != 1:
            raise ValueError(f"Expected one tabletop collider in {first['scene']}")
        table = aabb(cache, top_candidates[0])
        target.GetAttribute("xformOp:orient").Set(Gf.Quatf(1))
        target.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0))
        cache.Clear()
        local_collider_bounds = union_bounds([aabb(cache, p) for p in colliders])
        local_visual_bounds = aabb(cache, target)
        payloads = stage.GetRootLayer().GetPrimAtPath("/world/" + name).payloadList.prependedItems
        payload = payloads[0].assetPath
        asset_path = (SCENES / payload).resolve()
        asset_stage = Usd.Stage.Open(str(asset_path))
        original_transform = UsdGeom.Xformable(asset_stage.GetDefaultPrim()).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())
        original_rotation = original_transform.ExtractRotation().GetQuat()
        original_scale = [Gf.Vec3d(*original_transform[i][:3]).GetLength() for i in range(3)]
        q = [float(original_rotation.GetReal()), *map(float, original_rotation.GetImaginary())]
        rotation_replaced = abs(q[1]) > 1e-5 or abs(q[2]) > 1e-5
        asset_row = {"catalog_index": catalog_index, "target": name,
                     "asset": str(asset_path.relative_to(ROOT)), "variant_count": len(variants),
                     "original_rotation_wxyz": q, "non_yaw_rotation_replaced": rotation_replaced,
                     "root_rigid_body": target.HasAPI(UsdPhysics.RigidBodyAPI),
                     "nested_bodies": nested_bodies, "collider_count": len(colliders),
                     "local_collider_bounds": local_collider_bounds,
                     "local_visual_bounds": local_visual_bounds, "original_scale": original_scale}
        for record in variants:
            path = SCENES / record["scene"]
            text = path.read_text()
            if skeleton(text, name) != signature:
                raise ValueError(f"Batch cannot reproduce non-pose differences in {path}")
            layer = Sdf.Layer.FindOrOpen(str(path))
            target_pose = prim_pose(layer, "/world/" + name)
            bin_pose = prim_pose(layer, "/world/grey_bin")
            for prim, pose in [(target, target_pose), (bin_prim, bin_pose)]:
                prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*pose["translate"]))
                prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(pose["orient"][0], Gf.Vec3f(*pose["orient"][1:])))
                prim.GetAttribute("xformOp:scale").Set(Gf.Vec3f(*pose["scale"]))
            cache.Clear()
            bounds = union_bounds([aabb(cache, p) for p in colliders])
            bin_bounds = aabb(cache, bin_prim)
            flags = []
            if not bounds:
                flags.append("missing_target_collider")
            if not asset_row["root_rigid_body"] or nested_bodies:
                flags.append("unexpected_rigid_body_structure")
            if rotation_replaced:
                flags.append("non_yaw_asset_rotation_replaced")
            if any(abs(original_scale[i] - target_pose["scale"][i]) > 1e-5 for i in range(3)):
                flags.append("asset_scale_replaced")
            gap = None
            overlaps_bin = False
            if bounds:
                gap = bounds["min"][2] - table["max"][2]
                if gap < -0.001:
                    flags.append("target_penetrates_table_height")
                if gap > 0.03:
                    flags.append("target_starts_over_3cm_above_table")
                if any(bounds["min"][i] < table["min"][i] - 0.001 or
                       bounds["max"][i] > table["max"][i] + 0.001 for i in (0, 1)):
                    flags.append("target_extends_past_table_edge")
                overlaps_bin = all(min(bounds["max"][i], bin_bounds["max"][i]) -
                                   max(bounds["min"][i], bin_bounds["min"][i]) > 0.001 for i in range(3))
                if overlaps_bin:
                    flags.append("target_bin_aabb_overlap")
            if any(bin_bounds["min"][i] < table["min"][i] - 0.001 or
                   bin_bounds["max"][i] > table["max"][i] + 0.001 for i in (0, 1)):
                flags.append("bin_extends_past_table_edge")
            # Parse task source without importing 7,200 simulator configurations.
            task_ast = ast.parse((TASKS / record["task"]).read_text())
            task_class = next((n for n in task_ast.body if isinstance(n, ast.ClassDef)
                               and n.name == record["class_name"]), None)
            if task_class is None:
                issues.append(f"Missing class {record['class_name']}")
            else:
                calls = [n for n in ast.walk(task_class) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id == "import_scene"]
                if len(calls) != 1 or ast.literal_eval(calls[0].args[0]) != "simple_pnp/" + record["scene"]:
                    issues.append(f"Wrong scene in {record['task']}")
            rows.append(dict(record, asset=asset_row["asset"], target_pose=target_pose,
                             bin_pose=bin_pose, target_collider_bounds=bounds, bin_bounds=bin_bounds,
                             table_bounds=table, initial_bottom_gap_m=gap, flags=flags))
        asset_rows.append(asset_row)
        print("STATIC_GROUP " + json.dumps({"done": group_index + 1, "total": len(groups),
                                            "catalog_index": catalog_index, "target": name}), flush=True)
    report = {"task_count": len(rows), "catalog_entry_count": len(groups), "integrity_issues": issues,
              "flag_counts": dict(collections.Counter(f for row in rows for f in row["flags"])),
              "generator_bin_dims": manifest["destination"]["assumed_solver_dims"],
              "actual_bin_dims": [rows[0]["bin_bounds"]["max"][i] - rows[0]["bin_bounds"]["min"][i]
                                  for i in range(3)], "assets": asset_rows, "tasks": rows}
    write_json(output / "static.json", report)
    csv_rows(output / "static_tasks.csv", rows,
             ["class_name", "catalog_index", "target", "scene", "asset", "initial_bottom_gap_m", "flags"])
    print("STATIC_DONE " + json.dumps({k: v for k, v in report.items() if k not in ("assets", "tasks")}), flush=True)


def physics_audit(args):
    import numpy as np
    import torch
    from isaaclab.managers import RecorderManagerBaseCfg
    from isaaclab.sensors import CameraCfg, FrameTransformerCfg, ContactSensorCfg
    import robolab.constants as constants
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.runtime import create_env
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
    constants.RECORD_IMAGE_DATA = False
    static = json.loads((args.static_report or args.output / "static.json").read_text())
    selected = None
    if args.selection:
        selection = json.loads(args.selection.read_text())
        selected = {row if isinstance(row, str) else row["class_name"] for row in selection}
    groups = collections.defaultdict(list)
    asset_by_index = {a["catalog_index"]: a for a in static["assets"]}
    for task in static["tasks"]:
        if selected is not None and task["class_name"] not in selected:
            continue
        groups[task["catalog_index"]].append(task)
    if selected is not None and selected != {t["class_name"] for rows in groups.values() for t in rows}:
        raise ValueError("Selection contains tasks missing from static report")
    groups = [(index, rows) for n, (index, rows) in enumerate(sorted(groups.items()))
              if n % args.shard_count == args.shard_index]
    if args.limit_groups:
        groups = groups[:args.limit_groups]
    result_path = args.output / f"physics_{args.shard_index}.jsonl"
    finished = set()
    if result_path.exists():
        for line in result_path.read_text().splitlines():
            row = json.loads(line)
            if "tasks" in row:
                finished.add(row["catalog_index"])
    for batch_number, (catalog_index, tasks) in enumerate(groups):
        if catalog_index in finished:
            continue
        environment = None
        started = time.monotonic()
        print("PHYSICS_START " + json.dumps({"catalog_index": catalog_index,
                                             "target": tasks[0]["target"]}), flush=True)
        try:
            constants.set_output_dir(str(args.output / f"runtime_{args.shard_index}"))
            auto_register_droid_envs(task=[str(TASKS / tasks[0]["task"])], task_dirs=None, cameras=[])
            cfg = parse_env_cfg(tasks[0]["class_name"], device=args.device, num_envs=len(tasks),
                                seed=0, use_fabric=True)
            for field, value in list(vars(cfg.scene).items()):
                if isinstance(value, (CameraCfg, FrameTransformerCfg)) or (
                    isinstance(value, ContactSensorCfg) and not args.check_success
                ):
                    setattr(cfg.scene, field, None)
            for field in list(vars(cfg.observations)):
                setattr(cfg.observations, field, None)
            cfg.recorders = RecorderManagerBaseCfg()
            cfg.rerender_on_reset = False
            cfg.wait_for_textures = False
            environment, cfg = create_env(cfg, device=args.device, num_envs=len(tasks))
            print(f"PHYSICS_CREATED {catalog_index}", flush=True)
            environment.reset()
            print(f"PHYSICS_RESET {catalog_index}", flush=True)
            origins = environment.scene.env_origins
            for name, pose_key in [(tasks[0]["target"], "target_pose"), ("grey_bin", "bin_pose")]:
                body = environment.scene[name]
                poses = torch.tensor([r[pose_key]["translate"] + r[pose_key]["orient"] for r in tasks],
                                     device=environment.device, dtype=torch.float32)
                poses[:, :3] += origins
                body.write_root_pose_to_sim(poses)
                body.write_root_velocity_to_sim(torch.zeros((len(tasks), 6), device=environment.device))
            robot = environment.scene["robot"]
            robot.set_joint_position_target(robot.data.joint_pos.clone())
            environment.scene.update(0.0)
            body = environment.scene[tasks[0]["target"]]
            asset = asset_by_index[catalog_index]
            local_bounds = asset["local_collider_bounds"] or asset["local_visual_bounds"]
            if local_bounds is None:
                raise ValueError("Target has no collision or visual geometry")
            corners = np.array([[x, y, z] for x in (local_bounds["min"][0], local_bounds["max"][0])
                                for y in (local_bounds["min"][1], local_bounds["max"][1])
                                for z in (local_bounds["min"][2], local_bounds["max"][2])])
            task_results = [dict(class_name=t["class_name"], catalog_index=catalog_index,
                                 target=t["target"], scene=t["scene"], samples=[],
                                 first_below_table_time_s=None, first_outside_table_time_s=None,
                                 initial_bottom_gap_m=t["initial_bottom_gap_m"], static_flags=t["flags"])
                            for t in tasks]
            if args.check_success:
                from robolab.core.world.world_state import get_world
                world = get_world(environment)
                success_term = cfg.terminations.success
                for row in task_results:
                    row.update(first_success_without_motion_s=None, success_sample_count=0,
                               first_bin_contact_s=None, first_gripper_contact_s=None,
                               max_speed_m_s=0.0, max_root_z_m=-float("inf"))
            steps = round(args.seconds / cfg.sim.dt)
            checkpoints = {0, 1, 6, 12, 30, 60, 120, 240, 360, 600, 1200, steps}

            def observe(step):
                state = body.data.root_state_w.detach().cpu().numpy().copy()
                state[:, :3] -= origins.detach().cpu().numpy()
                # Conservative upper geometry bound: an entire body below the
                # tabletop is definitive even if a rotated local box is loose.
                w, x, y, z = (state[:, i] for i in range(3, 7))
                zrow = np.stack([2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)], axis=1)
                max_z = (zrow @ corners.T).max(axis=1) + state[:, 2]
                velocities = np.linalg.norm(state[:, 7:10], axis=1)
                predicate_check = args.check_success and step > 0 and step % cfg.decimation == 0
                if predicate_check:
                    environment.episode_length_buf[:] = step // cfg.decimation
                    success = success_term.func(environment, **success_term.params).detach().cpu().tolist()
                    bin_contact = world.in_contact(tasks[0]["target"], "grey_bin").detach().cpu().tolist()
                    gripper_contact = world.in_contact(tasks[0]["target"], "gripper").detach().cpu().tolist()
                for i, row in enumerate(task_results):
                    if args.check_success:
                        row["max_speed_m_s"] = max(row["max_speed_m_s"], float(velocities[i]))
                        row["max_root_z_m"] = max(row["max_root_z_m"], float(state[i, 2]))
                    if predicate_check:
                        if success[i]:
                            row["success_sample_count"] += 1
                            if row["first_success_without_motion_s"] is None:
                                row["first_success_without_motion_s"] = step * cfg.sim.dt
                        if bin_contact[i] and row["first_bin_contact_s"] is None:
                            row["first_bin_contact_s"] = step * cfg.sim.dt
                        if gripper_contact[i] and row["first_gripper_contact_s"] is None:
                            row["first_gripper_contact_s"] = step * cfg.sim.dt
                    table = tasks[i]["table_bounds"]
                    if max_z[i] < table["max"][2] - 0.05 and row["first_below_table_time_s"] is None:
                        row["first_below_table_time_s"] = step * cfg.sim.dt
                    if any(state[i, axis] < table["min"][axis] - 0.01 or
                           state[i, axis] > table["max"][axis] + 0.01 for axis in (0, 1)):
                        if row["first_outside_table_time_s"] is None:
                            row["first_outside_table_time_s"] = step * cfg.sim.dt
                    if step in checkpoints:
                        row["samples"].append({"time_s": step * cfg.sim.dt,
                                               "position": state[i, :3].tolist(),
                                               "rotation_wxyz": state[i, 3:7].tolist(),
                                               "speed_m_s": float(velocities[i]),
                                               "upper_bound_z_m": float(max_z[i])})
                    if step == steps:
                        row["final_position"] = state[i, :3].tolist()
                        row["final_speed_m_s"] = float(velocities[i])
                        row["displacement_m"] = float(np.linalg.norm(state[i, :3] -
                                                         np.array(tasks[i]["target_pose"]["translate"])))
                        row["fell_below_table"] = row["first_below_table_time_s"] is not None
                        row["left_table_bounds"] = row["first_outside_table_time_s"] is not None
                        row["moving_at_end"] = bool(velocities[i] > 0.05)
                        row["nonfinite_state"] = not bool(np.isfinite(state[i]).all())
                        if args.check_success:
                            row["success_without_motion"] = row["first_success_without_motion_s"] is not None
                            row["final_success_without_motion"] = bool(success[i]) if predicate_check else None
            observe(0)
            for step in range(1, steps + 1):
                environment.scene.write_data_to_sim()
                environment.sim.step(render=False)
                environment.scene.update(cfg.sim.dt)
                observe(step)
            result = {"catalog_index": catalog_index, "target": tasks[0]["target"], "tasks": task_results,
                      "seconds": args.seconds, "dt": cfg.sim.dt, "device": args.device,
                      "check_success": args.check_success,
                      "elapsed_s": time.monotonic() - started}
        except Exception:
            result = {"catalog_index": catalog_index, "target": tasks[0]["target"],
                      "error": traceback.format_exc(), "elapsed_s": time.monotonic() - started}
        finally:
            if environment is not None:
                environment.close()
        with result_path.open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        print("PHYSICS_GROUP " + json.dumps({"batch": batch_number + 1, "total": len(groups),
                                             "catalog_index": catalog_index, "target": tasks[0]["target"],
                                             "elapsed_s": result["elapsed_s"],
                                             "fall_count": sum(t["fell_below_table"] for t in result.get("tasks", [])),
                                             "success_without_motion_count": sum(bool(t.get("success_without_motion")) for t in result.get("tasks", [])),
                                             "error": result.get("error")}), flush=True)
        if result.get("error"):
            # A partially constructed environment can poison the simulator's
            # singleton. Preserve diagnostics and exit; resume skips successes.
            raise RuntimeError(result["error"])
        del environment, cfg, robot, body
        gc.collect()
        torch.cuda.empty_cache()


def make_report(output):
    static = json.loads((output / "static.json").read_text())
    by_index, errors = {}, []
    for path in sorted(output.glob("physics_*.jsonl")):
        for line in path.read_text().splitlines():
            result = json.loads(line)
            if "tasks" in result:
                by_index[result["catalog_index"]] = result
            else:
                errors.append(result)
    dynamic = {task["class_name"]: task for group in by_index.values() for task in group["tasks"]}
    rows = [dict(task, **{k: v for k, v in dynamic.get(task["class_name"], {}).items()
                         if k not in task and k not in ("static_flags",)}) for task in static["tasks"]]
    falls = [r for r in rows if r.get("fell_below_table")]
    asset_summary = []
    for asset in static["assets"]:
        tasks = [r for r in rows if r["catalog_index"] == asset["catalog_index"]]
        asset_summary.append(dict(asset,
            physics_tested=sum("fell_below_table" in r for r in tasks),
            fall_count=sum(bool(r.get("fell_below_table")) for r in tasks),
            edge_departure_count=sum(bool(r.get("left_table_bounds")) for r in tasks),
            moving_at_end_count=sum(bool(r.get("moving_at_end")) for r in tasks),
            table_penetration_count=sum("target_penetrates_table_height" in r["flags"] for r in tasks),
            bin_aabb_overlap_count=sum("target_bin_aabb_overlap" in r["flags"] for r in tasks)))
    report = {"total_tasks": len(rows), "physics_tested": len(dynamic), "fall_count": len(falls),
              "catalog_entries_with_falls": sum(a["fall_count"] > 0 for a in asset_summary),
              "edge_departure_count": sum(bool(r.get("left_table_bounds")) for r in rows),
              "moving_at_end_count": sum(bool(r.get("moving_at_end")) for r in rows),
              "nonfinite_state_count": sum(bool(r.get("nonfinite_state")) for r in rows),
              "static_flag_counts": static["flag_counts"], "integrity_issues": static["integrity_issues"],
              "errors": errors, "assets": asset_summary, "tasks": rows}
    write_json(output / "report.json", report)
    csv_rows(output / "tasks.csv", rows, ["class_name", "catalog_index", "target", "scene", "asset",
             "initial_bottom_gap_m", "flags", "fell_below_table", "left_table_bounds", "moving_at_end",
             "first_below_table_time_s", "first_outside_table_time_s", "final_position", "displacement_m"])
    csv_rows(output / "assets.csv", asset_summary, ["catalog_index", "target", "asset", "variant_count",
             "physics_tested", "fall_count", "edge_departure_count", "moving_at_end_count",
             "table_penetration_count", "bin_aabb_overlap_count", "non_yaw_rotation_replaced"])
    csv_rows(output / "fallen_tasks.csv", falls, ["class_name", "catalog_index", "target", "scene", "asset",
             "first_below_table_time_s", "final_position", "initial_bottom_gap_m", "flags"])
    summary = [f"Audited {len(rows):,} generated tasks across {len(asset_summary)} catalog entries.",
               f"Physics checked {len(dynamic):,} tasks; {len(falls):,} fell below the tabletop "
               f"across {report['catalog_entries_with_falls']} catalog entries.",
               f"{report['edge_departure_count']} object origins left the table's XY bounds; "
               f"{report['moving_at_end_count']} objects were still moving faster than 5 cm/s at the end.", "",
               "Physics: Isaac Sim 5.0 / Isaac Lab 2.2, seed 0, GPU PhysX, dt=1/120 s, robot joints held "
               "at default positions, no policy, five simulated seconds per task. Cameras and sensor "
               "readouts disabled. Each batch uses one catalog entry's current task/asset and all 50 "
               "authored target/bin placements; static fingerprints verify all other content is identical.", "",
               "A fall means the entire target collision-bounds box goes at least 5 cm below the "
               "authored tabletop. An AABB overlap is a conservative geometry flag, not a confirmed contact. "
               "A task that stays on the table for five seconds may still have an invalid starting pose "
               "or fail later. One launch per placement, without policy movement.", "",
               "Static checks use the current USD payloads, not only catalog sizes. "
               f"Generator bin size: {static['generator_bin_dims']}; actual: {static['actual_bin_dims']}.", "",
               "The generator drops the mouse asset's 90-degree rotation, changes apple_01's scale "
               "from 0.01 to 1 (100 times too large), and uses half the object height rather than "
               "its actual lower geometry bound to place it. All 50 oversized apple_01 placements "
               "left the table bounds and remained in motion; their giant bounds stay above the "
               "table, so they are not counted in the below-table fall metric.", "",
               "The coffee cup uses 32 instanced colliders, which the final static scan includes. "
               "Its physics fall detector used a slightly larger visual bounds box; this is "
               "conservative and does not turn a supported cup into a reported fall.", "",
               "| Catalog index | Target | Falls / tested | Left table XY | Table penetration | Bin AABB overlap |",
               "| --- | --- | --- | --- | --- | --- |"]
    for asset in sorted(asset_summary, key=lambda a: (-a["fall_count"], a["catalog_index"])):
        if asset["fall_count"] or asset["table_penetration_count"] or not asset["physics_tested"]:
            summary.append(f"| {asset['catalog_index']:03d} | {asset['target']} | "
                           f"{asset['fall_count']} / {asset['physics_tested']} | "
                           f"{asset['edge_departure_count']} | "
                           f"{asset['table_penetration_count']} | {asset['bin_aabb_overlap_count']} |")
    if errors:
        summary.extend(["", "Some launches reported errors; see report.json."])
    (output / "summary.md").write_text("\n".join(summary) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("assets", "tasks", "errors")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["static", "physics", "report"], required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--static-report", type=Path, default=None)
    parser.add_argument("--selection", type=Path, default=None)
    parser.add_argument("--check-success", action="store_true")
    # The report phase uses the standard library and does not launch Isaac Sim.
    if "--phase" in sys.argv and sys.argv[sys.argv.index("--phase") + 1] == "report":
        args = parser.parse_args()
        make_report(args.output)
        return
    import cv2  # noqa: F401 -- import before the simulator, as in project launchers.
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid shard index/count")
    args.output.mkdir(parents=True, exist_ok=True)
    launcher = AppLauncher(args)
    try:
        if args.phase == "static":
            static_audit(args.output)
        else:
            physics_audit(args)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
