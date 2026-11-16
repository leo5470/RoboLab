"""Simulator trials that record nothing, one subcommand per plan phase (rendering off unless asked):

* ``transfer`` (Phase 3): do GraspDataGen's floating-gripper grasps hold when the DROID arm executes them? Per object,
  on one training layout, the first ``--count`` grasps in selector order that pass the filters, numpy IK and the
  command replay. Each: pregrasp, grasp, close, lift 10 cm, hold 1 s. Lifted = object rose > 5 cm with both fingers
  in contact >= 80% of the hold. Output ``<out>/transfer/<idx>_<name>.jsonl`` + ``summary.json``.
* ``pnp`` (Phase 4, and the 4b gate with ``--qualified``): full pick-and-place, ``--seeds`` per training layout.
  Gate: success >= 30% and, from 5 failures on, no failure class above half of them.
  Output ``<out>/trials[_qualified][_<tag>]/<idx>_<name>.jsonl`` + ``summary.json``.
* ``screen`` (Phase 4b, D9): per layout and seed, ``--k`` grasps drawn at random from those passing the table/bin
  filters, numpy IK and the command replay (no corridor filter, no cost ranking), each run end to end; the ones that
  succeed are the layout's qualified grasps, which ``pnp --qualified`` and ``collect_demos.py --qualified`` draw from.
  ``--grasp-class handle`` screens handle grasps only. Output ``<out>/qualified/<task>.json``.

All three resume from their outputs.

    scripts/scripted_demos/run.sh scripts/scripted_demos/trials.py transfer [--objects 303 ...] [--count 50]
    scripts/scripted_demos/run.sh scripts/scripted_demos/trials.py pnp [--objects ...] [--seeds 3] [--qualified]
    scripts/scripted_demos/run.sh scripts/scripted_demos/trials.py screen --objects 303 [--grasp-class handle] [--k 16]
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import pathlib

import numpy as np

from robolab.core.scripted_demos.grasps import frames, library, objects
from robolab.core.scripted_demos import collect, motion, select, sim
from robolab.core.scripted_demos.records import read_jsonl


def pnp_gate(rows: list[dict], threshold: float = 0.3, min_failures_for_share: int = 5) -> dict:
    """Plan Phase 4 gate: success >= 30% and no failure class above half of all failures. The share rule is applied
    only from ``min_failures_for_share`` failures on (with one failure, its class is trivially 100%)."""
    n = len(rows)
    wins = sum(r["success"] for r in rows)
    failures = collections.Counter(r["failure"] for r in rows if not r["success"])
    worst = max(failures.values()) if failures else 0
    rate = wins / n if n else 0.0
    share_ok = (n - wins) < min_failures_for_share or worst <= (n - wins) / 2
    return {"attempts": n, "successes": wins, "success_rate": rate, "failures": dict(failures),
            "max_failure_share": worst / max(1, n - wins), "passed": bool(n and rate >= threshold and share_ok)}


def transfer_summary(rows: list[dict]) -> dict:
    out = {"trials": len(rows), "lifted": sum(r["success"] for r in rows)}
    out["transfer_rate"] = out["lifted"] / len(rows) if rows else None
    for key in ("passed_1g", "approach"):
        groups = collections.defaultdict(list)
        for r in rows:
            groups[str(r["extra"][key])].append(r["success"])
        out[f"by_{key}"] = {k: {"n": len(v), "rate": float(np.mean(v))} for k, v in sorted(groups.items())}
    out["failures"] = dict(collections.Counter(r["failure"] for r in rows if not r["success"]))
    lifts = [r["extra"].get("object_lift_m") for r in rows if r["extra"].get("object_lift_m") is not None]
    out["median_object_lift_m"] = float(np.median(lifts)) if lifts else None
    return out


def write_summary(path: pathlib.Path, key: str, value: dict) -> None:
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1))
    print(key, json.dumps(value), flush=True)


def run_transfer(args, out: pathlib.Path) -> None:
    out_dir = out / "transfer"
    for key in args.objects:
        obj = objects.get(key)
        layout = (args.layouts or objects.training_layouts(obj))[0]
        path = out_dir / f"{obj.key}.jsonl"
        done = {r["extra"]["grasp_id"] for r in read_jsonl(path)}
        session = collect.Session(obj, layout, render=args.render, robolab_output=out / "robolab_output" / "transfer")
        session.reset(0)
        session.settle(None, None, 0)
        state = sim.read_scene(session.world, session.ik, obj)
        found = select.evaluate(session.library, state, session.chain, np.random.default_rng(0),
                                select.Params(lift_only=0.10, backoff=(0.07, 0.07)))
        plans = found.plans[: args.count]
        print(f"{obj.key} {session.task}: {found.report['counts']} -> testing {len(plans)}", flush=True)
        for i, plan in enumerate(plans):
            if plan.grasp_id in done:
                continue
            result, _ = session.run(i, mode="transfer", plan=plan)
            approach_z = float((state.T_world_obj @ session.library["T_obj_grasp"][plan.grasp_id])[2, frames.APPROACH_AXIS])
            result.extra |= {
                "grasp_id": plan.grasp_id,
                "passed_1g": bool(session.library["passed_1g"][plan.grasp_id]) if "passed_1g" in session.library else None,
                "twin": int(session.library["twin"][plan.grasp_id]),
                "approach": "top" if approach_z < -0.7 else ("side" if abs(approach_z) <= 0.3 else "oblique"),
                "approach_z": approach_z,
            }
            result.selection = found.report
            collect.append_jsonl(path, result.to_json())
            print(f"  [{i + 1}/{len(plans)}] grasp {plan.grasp_id} {result.extra['approach']}: "
                  f"{'LIFTED' if result.success else result.failure} lift={result.extra.get('object_lift_m', 0):.3f} "
                  f"({result.wall_s:.0f}s)", flush=True)
        session.close()
        write_summary(out_dir / "summary.json", obj.key,
                      {"task": session.task, "selection_counts": found.report["counts"],
                       **transfer_summary(read_jsonl(path))})


def run_pnp(args, out: pathlib.Path) -> None:
    motion_params = motion.MotionParams(segment_scale={k: float(v) for k, v in (x.split("=") for x in args.segment_scale)})
    select_params = select.Params(motion=motion_params)
    if args.yaws is not None:
        select_params.preplace_yaws = tuple(args.yaws)
    out_dir = out / (("trials_qualified" if args.qualified else "trials") + (f"_{args.tag}" if args.tag else ""))
    for key in args.objects:
        obj = objects.get(key)
        path = out_dir / f"{obj.key}.jsonl"
        done = {(r["layout"], r["seed"]) for r in read_jsonl(path)}
        videos = sum(1 for r in read_jsonl(path) if r["success"])
        for layout in args.layouts or objects.training_layouts(obj):
            seeds = args.seed_list if args.seed_list is not None else range(args.seeds)
            todo = [s for s in seeds if (layout, s) not in done]
            if not todo:
                continue
            render = args.render or args.videos > 0 or args.failure_videos
            session = collect.Session(obj, layout, render=render, robolab_output=out / "robolab_output" / "trials")
            allowed, select_params_task, motion_params_task = None, select_params, motion_params
            if args.qualified:
                found = collect.qualified_grasps(session.task)
                if found:
                    allowed, select_params_task = found  # the configuration the grasps were screened with
                    motion_params_task = select_params_task.motion
            for seed in todo:
                if args.qualified and not allowed:
                    result = collect.Result(session.task, layout, seed, False, "no_feasible_grasp", 0, None,
                                            {"qualified": 0}, None, 0.0)
                else:
                    keep = videos < args.videos or args.failure_videos
                    video = out_dir / "videos" / f"{session.task}_s{seed}" if keep else None
                    result, _ = session.run(seed, allowed=allowed, video=video, select_params=select_params_task,
                                            motion_params=motion_params_task, keep_failed_video=args.failure_videos)
                    videos += int(result.success and video is not None)
                collect.append_jsonl(path, result.to_json())
                print(f"{session.task} seed {seed}: {'SUCCESS' if result.success else result.failure} "
                      f"steps={result.steps} grasp={result.plan and result.plan['grasp_id']} ({result.wall_s:.0f}s)",
                      flush=True)
            session.close()
        write_summary(out_dir / "summary.json", obj.key, pnp_gate(read_jsonl(path)))


def run_screen(args, out: pathlib.Path) -> None:
    # Only the IK and table filters (no corridor, no cost ranking), plus the command replay. Saved with the results:
    # pnp --qualified and collect_demos.py --qualified reselect the qualified grasps with exactly this configuration.
    params = select.Params(use_corridor=False, max_candidates_for_ik=4096)
    config = params.to_json()
    for key in args.objects:
        obj = objects.get(key)
        for layout in args.layouts or objects.training_layouts(obj):
            task = obj.task(layout)
            path = out / "qualified" / f"{task}.json"
            data = json.loads(path.read_text()) if path.exists() else None
            if data is not None and data.get("params") != config:
                # Screened under another configuration (or before it was saved): out of date, start over.
                stale = out / "qualified" / "stale" / f"{task}.{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
                stale.parent.mkdir(parents=True, exist_ok=True)
                path.replace(stale)
                print(f"{task}: screening was out of date; moved to {stale}", flush=True)
                data = None
            if data is None:
                data = {"task": task, "k": args.k, "params": config, "trials": []}
            tried = {(r["seed"], r["grasp_id"]) for r in data["trials"]}
            session = collect.Session(obj, layout, render=False, robolab_output=out / "robolab_output" / "screen")
            allowed = None
            if args.grasp_class:
                kind = library.classify(session.library["T_obj_grasp"], objects.mesh_vertices(obj), obj.index)
                allowed = np.nonzero(kind == args.grasp_class)[0]
                data["grasp_class"] = args.grasp_class
            for seed in range(args.seeds):
                rng = np.random.default_rng(np.random.SeedSequence([obj.index, int(layout), 10_000 + seed]))
                session.reset(seed)
                session.settle(None, None, seed)
                found = select.evaluate(session.library, sim.read_scene(session.world, session.ik, obj), session.chain,
                                        rng, params, allowed)
                data.setdefault("selection", {})[str(seed)] = found.report.get("counts")
                pool = found.plans
                picks = [pool[i] for i in rng.choice(len(pool), size=min(args.k, len(pool)), replace=False)] if pool else []
                for plan in picks:
                    if (seed, plan.grasp_id) in tried:
                        continue
                    result, _ = session.run(seed, plan=plan, select_params=params, motion_params=params.motion)
                    data["trials"].append({"seed": seed, "grasp_id": plan.grasp_id, "success": result.success,
                                           "failure": result.failure, "steps": result.steps})
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(data, indent=1))
                    print(f"{task} seed {seed} grasp {plan.grasp_id}: {'OK' if result.success else result.failure}",
                          flush=True)
            session.close()
            data["qualified"] = sorted({r["grasp_id"] for r in data["trials"] if r["success"]})
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=1))
            print(f"{task}: {len(data['qualified'])} qualified of {len(data['trials'])} tried", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--objects", nargs="*", default=[str(o.index) for o in objects.BATCH])
    common.add_argument("--layouts", nargs="*", default=None, help="default: the object's training layouts")
    common.add_argument("--render", action="store_true")
    common.add_argument("--out-dir", type=pathlib.Path, default=None, help="default: <grasp_data>")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transfer", parents=[common], help="Phase 3 lift test")
    t.add_argument("--count", type=int, default=50)

    p = sub.add_parser("pnp", parents=[common], help="Phase 4 / 4b pick-and-place trials")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed-list", type=int, nargs="*", default=None, help="run these seeds instead of 0..seeds-1")
    p.add_argument("--qualified", action="store_true", help="draw only screened grasps (Phase 4b gate)")
    p.add_argument("--videos", type=int, default=0, help="videos of the first N successes (renders)")
    p.add_argument("--failure-videos", action="store_true", help="also keep videos of failed attempts")
    p.add_argument("--tag", default="", help="suffix for the output folder (parameter experiments)")
    p.add_argument("--segment-scale", nargs="*", default=[], metavar="SEGMENT=SCALE",
                   help="lengthen segments, e.g. preplace=1.6 lift=1.5")
    p.add_argument("--yaws", type=float, nargs="*", default=None, help="preplace yaw candidates (rad)")

    s = sub.add_parser("screen", parents=[common], help="Phase 4b per-layout grasp screening")
    s.add_argument("--k", type=int, default=16)
    s.add_argument("--seeds", type=int, default=1)
    s.add_argument("--grasp-class", choices=["handle", "rim", "body"], default=None)

    args = parser.parse_args()
    out = args.out_dir or objects.grasp_data_root()
    {"transfer": run_transfer, "pnp": run_pnp, "screen": run_screen}[args.command](args, out)


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
