"""Scripted pick-and-place collected in the OMY teleoperation environment, as OMY recordings.

    scripts/scripted_demos/run.sh scripts/scripted_demos/collect_omy.py --object 230 --layouts 01 02 \\
        --episodes-per-layout 2 --output <session dir> [--hold-joint3] [--qualified]
    # explicit targets (robolab.core.omy_teleop.coverage_profile targets), built in one v1 layout's scene:
    scripts/scripted_demos/run.sh scripts/scripted_demos/collect_omy.py --object 230 --layouts 01 \\
        --targets targets.json [--target-ids t0_0000 ...] --episodes-per-target 1 --output <session dir>

Writes an OMY session: ``<output>/<layout task>/part_000/{omy_demos.hdf5, env_cfg.json}`` per layout, every attempt
recorded (failures included, as OMY collection keeps them), plus ``attempts.jsonl``, ``summary.json`` and
``manifest.json`` at the session root. See ``robolab/core/scripted_demos/omy_collect.py`` and
docs/scripted_demos.md, "In the OMY environment".
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from robolab.core.scripted_demos import collect, sim
from robolab.core.scripted_demos.grasps import objects
from robolab.core.scripted_demos.records import git_revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object", required=True, help="catalog index or name, e.g. 230")
    parser.add_argument("--layouts", nargs="+", required=True, help="v1 layout ids, e.g. 01 02")
    parser.add_argument("--episodes-per-layout", type=int, default=1, help="successful episodes per layout")
    parser.add_argument("--max-attempts-per-layout", type=int, default=10)
    parser.add_argument("--targets", type=pathlib.Path,
                        help="target manifest: one attempt sequence per target, in the scene of the single --layouts")
    parser.add_argument("--target-ids", nargs="*", default=None, help="these targets only (default: all)")
    parser.add_argument("--episodes-per-target", type=int, default=1)
    parser.add_argument("--max-attempts-per-target", type=int, default=3)
    parser.add_argument("--hold-joint3", action="store_true", help="plan and execute with panda_joint3 held at home")
    parser.add_argument("--qualified", action="store_true", help="draw only screened grasps")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="new session directory")
    args = parser.parse_args()
    obj = objects.get(args.object)
    targets = None
    if args.targets is not None:
        if len(args.layouts) != 1:
            raise SystemExit("--targets needs exactly one --layouts (the scene the targets are built in)")
        manifest = json.loads(args.targets.read_text())
        targets = [row | {"manifest": str(args.targets.resolve())} for row in manifest["targets"]
                   if args.target_ids is None or row["id"] in args.target_ids]
        if not targets:
            raise SystemExit("no targets selected")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"output must be new or empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    locked = ("panda_joint3",) if args.hold_joint3 else ()

    from robolab.core.scripted_demos.omy_collect import OmySession, default_select_params

    rows, status = [], 0
    try:
        for layout in args.layouts:
            session = OmySession(obj, layout, output=args.output / obj.task(layout) / "part_000",
                                 locked_joints=locked)
            params, allowed = default_select_params(obj, layout, args.qualified)
            plan = ([(target, args.episodes_per_target, args.max_attempts_per_target) for target in targets]
                    if targets is not None else [(None, args.episodes_per_layout, args.max_attempts_per_layout)])
            try:
                for target, wanted, budget in plan:
                    successes = 0
                    for seed in range(budget):
                        if successes >= wanted:
                            break
                        t0 = time.time()
                        result, _ = session.run(seed, select_params=params, allowed=allowed, target=target)
                        row = result.to_json() | {"part": str(session.output.relative_to(args.output)),
                                                  "target": target["id"] if target else None}
                        rows.append(row)
                        collect.append_jsonl(args.output / "attempts.jsonl", row)
                        successes += bool(result.success)
                        label = f"{session.task}{' ' + target['id'] if target else ''}"
                        print(f"{label} seed {seed}: {'SUCCESS' if result.success else result.failure} "
                              f"steps={result.steps} ({time.time() - t0:.0f}s) [{successes}/{wanted}]", flush=True)
            finally:
                session.close()
        summary = {"object": obj.key, "layouts": args.layouts,
                   "targets": None if targets is None else {"manifest": str(args.targets.resolve()),
                                                            "count": len(targets),
                                                            "succeeded": len({r["target"] for r in rows if r["success"]})},
                   "attempts": len(rows),
                   "successes": sum(r["success"] for r in rows),
                   "failures": {f: sum(r["failure"] == f for r in rows) for f in {r["failure"] for r in rows} if f}}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
        (args.output / "manifest.json").write_text(json.dumps({
            "kind": "scripted_omy_session", "object": obj.key, "family": obj.family, "layouts": args.layouts,
            "targets": str(args.targets.resolve()) if targets is not None else None,
            "episodes_per_layout": args.episodes_per_layout, "locked_joints": list(locked), "qualified": args.qualified,
            "code_revision": {"RoboLab": git_revision(objects.ROBOLAB_DIR)}}, indent=1))
        print(json.dumps(summary), flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        status = 1
    finally:
        sim.close_app(status)


if __name__ == "__main__":
    main()
