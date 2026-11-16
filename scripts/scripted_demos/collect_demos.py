"""Phases 5-6: collect successful scripted episodes per object family into a PPS HDF5 file, then check it with PPS's
own reader (``scripts/scripted_demos/validate_pps.sh``: ``prepare_jointpos.py --validate-only``).

Objects are collected one after another in one process (one Kit start). Episodes are spread evenly over ``--num-layouts`` training layouts, taken in order from the
layouts that are not evaluation layouts ``00 05 … 45``. A layout where the object cannot be picked at all (no feasible
grasp on its first attempts, or no success in ``--give-up-after`` attempts) is skipped and the next one is used, as
PPS's RoboCasa demonstrations screened layouts by the scripted policy succeeding. Attempts per layout continue from
the last recorded seed, so the run resumes. Only successful episodes are written (success = the env's ``success`` termination). Up to ``--videos`` successes,
one per layout, are also rendered to MP4 (exterior camera, wrist camera, viewport).

    SIM_GPU=0 scripts/scripted_demos/run.sh scripts/scripted_demos/collect_demos.py --objects 301 294 [--episodes 50]
    scripts/scripted_demos/run.sh scripts/scripted_demos/collect_demos.py --objects 303 --qualified --num-layouts 14

Writes, under ``<scripted_demos>/<family>/``: ``pps.hdf5``, ``layouts.json``, ``attempts.jsonl``, ``videos/*.mp4``,
``manifest.json`` and ``summary.json``.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import subprocess

import h5py
import numpy as np

from robolab.core.scripted_demos.grasps import objects
from robolab.core.scripted_demos import collect, sim
from robolab.core.scripted_demos.writer import Writer
from robolab.core.scripted_demos.records import file_sha256, git_revision, read_jsonl


def layout_verdict(rows: list[dict], have: int, args) -> str | None:
    """Why a layout should be skipped, or None."""
    if have:
        return None
    if len(rows) >= 3 and all(r["failure"] == "no_feasible_grasp" for r in rows[:3]):
        return "no feasible grasp after settling"
    if len(rows) >= args.give_up_after:
        return f"no success in {len(rows)} attempts"
    return None


def episodes_in(path, task: str) -> int:
    if not path.exists():
        return 0
    with h5py.File(path, "r") as f:
        return sum(1 for n in f.get("data", {}) if f["data"][n].attrs.get("layout_task") == task)


def summarize(out_dir, obj, layouts) -> dict:
    attempts = read_jsonl(out_dir / "attempts.jsonl")
    with h5py.File(out_dir / "pps.hdf5", "r") as f:
        eps = [f["data"][n] for n in sorted(f["data"])]
        lengths = [int(g.attrs["num_samples"]) for g in eps]
        joints = np.concatenate([g["joint_position"][:] for g in eps]) if eps else np.zeros((0, 7))
        actions = np.concatenate([g["actions"][:] for g in eps]) if eps else np.zeros((0, 8))
        grasp_ids = [int(g.attrs["grasp_id"]) for g in eps]
        approach = np.array([json.loads(g.attrs["plan"])["approach_axis_world"] for g in eps]) if eps else np.zeros((0, 3))
        per_layout = collections.Counter(g.attrs["layout_task"] for g in eps)
    failures = collections.Counter(a["failure"] for a in attempts if not a["success"])
    return {
        "object": obj.key, "family": obj.family, "layouts": layouts,
        "attempts": len(attempts), "successes": sum(a["success"] for a in attempts),
        "success_rate": (sum(a["success"] for a in attempts) / len(attempts)) if attempts else None,
        "episodes_written": len(lengths), "episodes_per_layout": dict(sorted(per_layout.items())),
        "failure_taxonomy": dict(failures),
        "step_length": {"min": min(lengths, default=None), "max": max(lengths, default=None),
                        "mean": float(np.mean(lengths)) if lengths else None,
                        "histogram": np.histogram(lengths, bins=[0, 100, 120, 130, 140, 150, 160, 180, 220, 400])[0].tolist()
                        if lengths else []},
        "ranges": {"joint_position_min": joints.min(0).round(4).tolist() if len(joints) else None,
                   "joint_position_max": joints.max(0).round(4).tolist() if len(joints) else None,
                   "panda_joint3_range": [float(joints[:, 2].min()), float(joints[:, 2].max())] if len(joints) else None,
                   "actions_min": actions.min(0).round(4).tolist() if len(actions) else None,
                   "actions_max": actions.max(0).round(4).tolist() if len(actions) else None},
        "grasp_diversity": {"distinct_grasp_ids": len(set(grasp_ids)),
                            "approach_axis_mean": approach.mean(0).round(3).tolist() if len(approach) else None,
                            "approach_axis_spread": float(np.linalg.norm(approach - approach.mean(0), axis=1).mean())
                            if len(approach) else None,
                            "approach_z_histogram": np.histogram(approach[:, 2], bins=[-1.01, -0.7, -0.3, 0.3, 1.01])[0].tolist()
                            if len(approach) else []},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objects", nargs="+", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--layouts", nargs="*", default=None, help="candidate layouts, in order (default: all non-eval)")
    parser.add_argument("--num-layouts", type=int, default=10)
    parser.add_argument("--max-attempts-per-layout", type=int, default=40)
    parser.add_argument("--give-up-after", type=int, default=12, help="skip a layout with no success in this many attempts")
    parser.add_argument("--videos", type=int, default=5)
    parser.add_argument("--qualified", action="store_true", help="draw only screened grasps (D9)")
    parser.add_argument("--output", default=None, help="one object only; default: <scripted_demos>/<family>")
    args = parser.parse_args()
    if args.output and len(args.objects) > 1:
        parser.error("--output takes one object")
    for key in args.objects:
        obj = objects.get(key)
        out_dir = objects.scripted_demos_root() / obj.family if args.output is None else pathlib.Path(args.output)
        collect_object(obj, out_dir, args)
        if (out_dir / "pps.hdf5").exists():
            if not os.environ.get("PPS_REPO"):
                print(f"{obj.key}: PPS validate-only skipped (set PPS_REPO, and PPS_PYTHON if needed)", flush=True)
                continue
            validate = pathlib.Path(__file__).with_name("validate_pps.sh")
            with open(out_dir / "pps_validation.log", "w") as log:
                code = subprocess.run([str(validate), str(out_dir / "pps.hdf5"), obj.family], stdout=log,
                                      stderr=subprocess.STDOUT).returncode
            last = (out_dir / "pps_validation.log").read_text().strip().splitlines()[-1:]
            print(f"{obj.key}: validate-only {'passed' if code == 0 else 'FAILED'} ({' '.join(last)})", flush=True)


def collect_object(obj: objects.BatchObject, out_dir: pathlib.Path, args) -> None:
    candidates = args.layouts or objects.training_layouts(obj, count=100)
    if set(candidates) & set(objects.EVAL_LAYOUTS):
        raise SystemExit(f"training layouts overlap the evaluation layouts: {sorted(set(candidates) & set(objects.EVAL_LAYOUTS))}")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(args.num_layouts, len(candidates))
    per_layout = [args.episodes // n + (i < args.episodes % n) for i in range(n)]
    attempts_path = out_dir / "attempts.jsonl"
    used: list[str] = []
    skipped: dict[str, str] = {}

    def write_layouts():
        (out_dir / "layouts.json").write_text(json.dumps({
            "object": obj.key, "family": obj.family, "layouts": used, "tasks": [obj.task(l) for l in used],
            "skipped": skipped, "candidates": candidates, "excluded_eval_layouts": list(objects.EVAL_LAYOUTS)}, indent=1))
    root = objects.grasp_data_root()
    lib_path = root / "library" / f"{obj.key}.npz"

    writer = None
    videos_dir = out_dir / "videos"
    for layout in candidates:
        if len(used) >= n:
            break
        target = per_layout[len(used)]
        task = obj.task(layout)
        rows = [r for r in read_jsonl(attempts_path) if r["task"] == task]
        have = episodes_in(out_dir / "pps.hdf5", task)
        verdict = layout_verdict(rows, have, args)
        if verdict:
            skipped[layout] = verdict
            write_layouts()
            continue
        if have >= target or len(rows) >= args.max_attempts_per_layout:
            used.append(layout)
            write_layouts()
            continue
        session = collect.Session(obj, layout, render=True, robolab_output=out_dir / "robolab_output",
                                  library_path=lib_path)
        if writer is None:
            writer = Writer(out_dir / "pps.hdf5", task=obj.family, prompt=session.instruction)
        allowed, select_params = None, None
        if args.qualified:
            found = collect.qualified_grasps(task)
            allowed, select_params = found if found else (None, None)  # reuse the screening configuration
            if not allowed:
                skipped[layout] = "no qualified grasps"
                write_layouts()
                session.close()
                continue
        seed = max((r["seed"] for r in rows), default=-1) + 1
        while have < target and len(rows) < args.max_attempts_per_layout:
            # At most one video per layout, so the videos show different layouts.
            done_videos = [p for p in videos_dir.glob("*.mp4") if not p.name.endswith(".partial.mp4")] \
                if videos_dir.exists() else []
            layout_has_video = any(p.name.startswith(f"{task}_") for p in done_videos)
            video = videos_dir / f"{task}_s{seed:03d}" if len(done_videos) < args.videos and not layout_has_video else None
            result, buffer = session.run(seed, record=True, allowed=allowed, video=video, select_params=select_params,
                                         motion_params=select_params.motion if select_params else None)
            row = result.to_json() | {"recorded_rows": len(buffer) if buffer else 0}
            if result.success:
                arrays = buffer.arrays()
                plan = dict(result.plan)
                plan["approach_axis_world"] = np.asarray(plan["waypoints"]["grasp"])[:3, 0].round(5).tolist()
                name = writer.append(arrays, {
                    "layout_task": task, "source_family": task, "seed": seed, "grasp_id": plan["grasp_id"],
                    "plan": json.dumps(plan), "selection_rule": "qualified" if args.qualified else "top-k cost draw",
                    "executor": json.dumps(result.executor)})
                row["episode"] = name
                have += 1
            rows.append(row)
            collect.append_jsonl(attempts_path, row)
            print(f"{task} seed {seed}: {'SUCCESS' if result.success else result.failure} steps={result.steps} "
                  f"[{have}/{target}] ({result.wall_s:.0f}s)", flush=True)
            seed += 1
            verdict = layout_verdict(rows, have, args)
            if verdict:
                skipped[layout] = verdict
                break
        session.close()
        if layout not in skipped:
            used.append(layout)
        write_layouts()

    manifest = {
        "object": obj.key, "family": obj.family, "file": "pps.hdf5",
        "selection_rule": "successful scripted episodes only (env success termination)",
        "draw": "qualified grasps (Phase 4b screening)" if args.qualified else "uniform over the top-k (k=8) cost-ranked feasible grasps",
        "registration": "stock auto_register_droid_envs(task_dirs=['single_pnp']), WRIST_LEFT cameras, 15 Hz",
        "conversion": "policies/pi0_family/client.py Pi0DroidJointposClient._extract_observation/_pack_request",
        "constant_columns": "none: panda_joint3 is commanded freely (7-DoF IK)",
        "library": str(lib_path), "library_sha256": file_sha256(lib_path),
        "library_manifest_sha256": file_sha256(root / "library" / "manifest.json"),
        "code_revision": {"RoboLab": git_revision(objects.ROBOLAB_DIR)},
        "layouts": used, "skipped_layouts": skipped, "episodes_per_layout_target": dict(zip(used, per_layout)),
        "seeds": "per layout, consecutive from 0; numpy SeedSequence([catalog index, layout, seed])",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if (out_dir / "pps.hdf5").exists():
        summary = summarize(out_dir, obj, used)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1), flush=True)


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
