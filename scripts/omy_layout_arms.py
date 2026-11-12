#!/usr/bin/env python3
"""Build the matched L/T pail datasets of the layout-coverage experiment; resumable.

Stages, each skipped when its frozen output exists:

1. qualify + local: the single-source pipeline with the generator profile,
   candidate factor 3 (60/30/30 near/moderate/wide candidates), into
   ROOT/generated, sharing ROOT/state.
2. targets: sample explicit targets (target_sampler, v1 construction), export
   the evaluation layouts, and rank qualified sources per target along the
   repaired path (path_screen --rank-profile); all frozen in ROOT/state/SESSION.
3. explicit: one job per target with its best feasible source, falling back
   to the next one, until enough distinct targets are accepted.
4. select: choose each arm's exact stratum counts (layout_arms.ARMS), drop
   candidates starting at an evaluation layout, publish ROOT/pools/{L,T}/SESSION.

See docs/omy_mimicgen_diversity_plan_20261108.md. GPUs come from
CUDA_VISIBLE_DEVICES (host GPU 0 when unset).
"""

import argparse
import fcntl
import glob
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from robolab.core.omy_teleop import layout_arms  # noqa: E402
from robolab.core.omy_teleop.final_dataset import export, read_json  # noqa: E402
from robolab.core.omy_teleop.gpu_pool import resolve_gpu_pool  # noqa: E402
from robolab.core.omy_teleop.pipeline import (LIMITS, Pipeline, atomic_json, implementation, source_key,  # noqa: E402
                                              verify_lineage, verify_pool)

USD_PYTHON = REPO / "tools/simple_pnp_v2/usd_python.sh"
STAGES = ("qualify", "local", "targets", "explicit", "select")


def usd_module(module, *args):
    subprocess.run([str(USD_PYTHON), "-m", module, *map(str, args)], cwd=REPO, check=True)


def freeze_targets(state, work, args):
    """Sample targets, export evaluation layouts and rank sources; each written once."""
    targets, layouts, ranking = state / "targets.json", state / "evaluation_layouts.json", state / "ranking.json"
    scenes = sorted({str(Path(s["input"]) / "env_cfg.json") for s in work.sources})
    v1 = sorted({str(REPO / "assets/scenes/simple_pnp" / Path(read_json(c)["scene"]["scene"]["spawn"]["usd_path"]).name)
                 for c in scenes})
    if not targets.exists():
        usd_module("robolab.core.omy_teleop.target_sampler", "--profile", args.target_profile, "--count",
                   args.target_count, "--seed", args.target_seed, "--output", targets, "--scene", *v1)
    if not layouts.exists():
        evaluation = sorted(glob.glob(str(REPO / args.evaluation_scenes)))
        if not evaluation:
            raise SystemExit(f"No evaluation scenes match {args.evaluation_scenes}")
        usd_module("robolab.core.omy_teleop.target_sampler", "--profile", args.target_profile, "--layouts-only",
                   "--output", layouts, "--scene", *evaluation)
    if not ranking.exists():
        qualified = state / "qualified_sources.json"
        atomic_json(qualified, {"sources": [s for s in work.sources if source_key(s) in work.qualified]})
        usd_module("robolab.core.omy_teleop.path_screen", "--manifest", targets, "--sources", qualified,
                   "--rank-profile", args.generator_profile, "--workers", args.screen_workers, "--output", ranking)
    return targets, read_json(layouts), read_json(ranking)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", default="20261024_134851_plasticpail_a01")
    parser.add_argument("--root", type=Path, default=Path("output/omy_layout_arms"))
    parser.add_argument("--collected-root", type=Path, default=Path("output/omy_teleop"))
    parser.add_argument("--generator-profile", default="p4")
    parser.add_argument("--target-profile", default="pail_v2_deployment_strata")
    parser.add_argument("--target-count", type=int, default=150, help="Explicit targets to sample and rank")
    parser.add_argument("--target-seed", type=int, default=20261112)
    parser.add_argument("--evaluation-scenes", default="assets/scenes/simple_pnp_v2/pnp_242_plasticpail_a01_*.usda")
    parser.add_argument("--screen-workers", type=int, default=16)
    parser.add_argument("--max-attempts-per-source", type=int, default=150)
    parser.add_argument("--stop-after", choices=STAGES)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    roots = {"collected": args.collected_root.resolve(), "generated": root / "generated", "pools": root / "pools/local",
             "training": root / "training", "cache": root / "cache", "pps": root / "pps", "state": root / "state",
             "summary": root / "summary.json"}
    roots["state"].mkdir(parents=True, exist_ok=True)
    lock = (roots["state"] / "layout_arms.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error(f"Another run holds {roots['state'] / 'layout_arms.lock'}")
    gpus = [gpu["host_index"] for gpu in resolve_gpu_pool()]
    print(f"GPU pool (host indices): {gpus}", flush=True)

    # 1-2: qualification and local candidates (L's 20/10/10 at factor 3; T's local share is a subset).
    stop = "qualify" if args.stop_after == "qualify" else "generate"
    pipeline = Pipeline(roots, gpus, count=50, collected_count=10, candidate_factor=3,
                        max_attempts_per_source=args.max_attempts_per_source, stop_after=stop,
                        generator_profile=args.generator_profile)
    if pipeline.run([args.session]) != 0:
        raise SystemExit("Qualification or local generation did not complete")
    work = pipeline.work[0]
    print(f"qualified {len(work.qualified)}/{len(work.sources)}; local shortfall {work.tier_shortfall}", flush=True)
    if args.stop_after in ("qualify", "local"):
        return 0

    state = roots["state"] / args.session
    targets, layouts, ranking = freeze_targets(state, work, args)
    print(f"targets: {ranking['counts']}", flush=True)
    if args.stop_after == "targets":
        return 0

    explicit = layout_arms.ExplicitTargets(work, targets, ranking, gpus, args.generator_profile,
                                           need=layout_arms.explicit_need())
    report = explicit.run()
    atomic_json(state / "explicit_report.json", report)
    print(f"explicit: {report}", flush=True)
    if report["accepted_targets"] < layout_arms.explicit_need():
        print("explicit targets below the planned surplus; selection proceeds if each arm can still be filled",
              flush=True)
    if args.stop_after == "explicit":
        return 0

    rows, rejected, tiers = layout_arms.candidate_rows(roots, work)
    names = layouts["names"]
    selected, selection = layout_arms.select_arms(rows, tiers, work.qualified, layouts["targets"], names)
    for arm, chosen in selected.items():
        output = root / "pools" / arm / args.session
        if output.exists():
            print(f"{arm}: {output} already published", flush=True)
            continue
        manifest = {"schema_version": 1, "status": "complete", "session": args.session, "family": work.family,
                    "arm": arm, "selected_count": 50, "quotas": layout_arms.ARMS[arm], "counts_by_stratum": chosen["counts"],
                    "counts_by_kind": {"collected": chosen["counts"].get("collected", 0),
                                       "generated": 50 - chosen["counts"].get("collected", 0)},
                    "generated_per_parent": chosen["parents"], "normalization": chosen["normalization"],
                    "thresholds": LIMITS, "generation_mode": "single_source",
                    "generator_profile": args.generator_profile, "implementation": implementation(args.generator_profile),
                    "qualified_sources": [{"input": k[0], "episode": k[1]} for k in sorted(work.qualified)],
                    "targets": {"manifest": str(targets), "ranking": str(state / "ranking.json"),
                                "evaluation_layouts": str(state / "evaluation_layouts.json")},
                    "selection": selection, "rejected": rejected,
                    "method": "exact stratum quotas; farthest-first within each stratum against all selected; "
                              "parents balanced per stratum; one demo per explicit target; starts within "
                              f"{layout_arms.COINCIDENT_M} m joint-XY of an evaluation layout dropped"}
        export(output, chosen["rows"], manifest)
        verify_lineage(output, verify_pool(output, 50), work.qualified)
        print(f"{arm}: published {output} {chosen['counts']}", flush=True)
    print(f"selection: {json.dumps({k: v for k, v in selection.items() if k != 'dropped'})}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
