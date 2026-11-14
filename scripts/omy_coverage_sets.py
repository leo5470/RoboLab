#!/usr/bin/env python3
"""Build nested 50 / 100 / 200-demonstration coverage sets for one collected session; resumable.

The recipe is the T arm of scripts/omy_layout_arms.py scaled up (see
robolab/core/omy_teleop/coverage_sets.py): the humans, local perturbations and
explicit targets across the object's Simple-PnP v2 range, each larger set
containing the smaller ones. Stages, each skipped when its frozen output exists:

1. qualify + local: the single-source pipeline with the generator profile
   until 1.5x the largest set's local share exists (2:1:1 near/moderate/wide).
2. targets: export the object's evaluation layouts, derive its target profile
   from them (coverage_profile), sample explicit targets in v1 construction
   and rank qualified sources per target along the repaired path.
3. explicit: one job per target until 1.5x the largest set's explicit share
   of distinct layouts is accepted.
4. select: choose the nested sets (frozen in state), publish
   ROOT/pools/T<size>/SESSION and the disjoint render chunks.
5. export: render each chunk and export its PPS file, then delete the chunk's
   training file and cache; assemble ROOT/pps/T<size>/SESSION from the chunks.

--base-pool/--base-pps name an already published and exported smaller set
(the pail's T arm) that every new set must contain. GPUs come from
CUDA_VISIBLE_DEVICES (host GPU 0 when unset).
"""

import argparse
import fcntl
import glob
import json
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from robolab.core.omy_teleop import coverage_sets, layout_arms  # noqa: E402
from robolab.core.omy_teleop.final_dataset import export, read_json  # noqa: E402
from robolab.core.omy_teleop.gpu_pool import resolve_gpu_pool  # noqa: E402
from robolab.core.omy_teleop.pipeline import (LIMITS, Pipeline, atomic_json, implementation, source_key,  # noqa: E402
                                              verify_lineage, verify_pool)

USD_PYTHON = REPO / "tools/simple_pnp_v2/usd_python.sh"
STAGES = ("generate", "targets", "explicit", "select", "export")


def usd_module(module, *args):
    subprocess.run([str(USD_PYTHON), "-m", module, *map(str, args)], cwd=REPO, check=True)


def freeze_targets(state, work, args, object_name):
    """Evaluation layouts, target profile, targets and source ranking; each written once."""
    layouts, profile = state / "evaluation_layouts.json", state / "profile.json"
    targets, ranking = state / "targets.json", state / "ranking.json"
    configs = sorted({str(Path(s["input"]) / "env_cfg.json") for s in work.sources})
    v1 = sorted({str(REPO / "assets/scenes/simple_pnp" / Path(read_json(c)["scene"]["scene"]["spawn"]["usd_path"]).name)
                 for c in configs})
    if not layouts.exists():
        scenes = sorted(glob.glob(str(REPO / args.evaluation_scenes.format(object=object_name))))
        if not scenes:
            raise SystemExit(f"No evaluation scenes match {args.evaluation_scenes.format(object=object_name)}")
        usd_module("robolab.core.omy_teleop.coverage_profile", "layouts", "--object", object_name,
                   "--output", layouts, "--scene", *scenes)
    if not targets.exists():
        if not profile.exists():
            usd_module("robolab.core.omy_teleop.coverage_profile", "profile", "--object", object_name,
                       "--layouts", layouts, "--output", profile, "--scene", *v1)
        usd_module("robolab.core.omy_teleop.coverage_profile", "targets", "--profile-file", profile,
                   "--count", args.target_count, "--seed", args.target_seed, "--output", targets, "--scene", *v1)
    if not ranking.exists():
        qualified = state / "qualified_sources.json"
        atomic_json(qualified, {"sources": [s for s in work.sources if source_key(s) in work.qualified]})
        usd_module("robolab.core.omy_teleop.path_screen", "--manifest", targets, "--sources", qualified,
                   "--rank-profile", args.generator_profile, "--workers", args.screen_workers, "--output", ranking)
    return targets, read_json(layouts), read_json(ranking)


def wait_for_space(path, need_gb, log=print):
    while shutil.disk_usage(path).free < need_gb * 2 ** 30:
        log(f"waiting: under {need_gb} GB free on {path}", flush=True)
        time.sleep(300)


def export_chunk(name, session, roots, root, gpu, args):
    """Render one chunk pool, export and verify its PPS file, then drop the chunk's training file and cache."""
    from robolab.core.omy_postprocess import pps_hdf5
    from robolab.core.omy_postprocess.automatic_dataset import process_one, verify_final
    from robolab.core.omy_postprocess.io import sha256_file
    from robolab.core.omy_teleop.pipeline import freeze_training_plan, training_options

    chunk = {**roots, **{k: root / "chunks" / k / name for k in ("pools", "training", "cache", "pps")}}
    pool, pps = chunk["pools"] / session, chunk["pps"] / session
    marker = pps / "chunk_done.json"
    if marker.exists():
        if sha256_file(pps / pps_hdf5.FILE_NAME) != read_json(marker)["sha256"]:
            raise ValueError(f"{name}: PPS chunk changed after it was verified")
        return
    manifest = verify_pool(pool, len(read_json(pool / "dataset.json")["episodes"]))
    count, humans = manifest["selected_count"], manifest["counts_by_kind"].get("collected", 0)
    wait_for_space(root, args.min_free_gb)
    freeze_training_plan(manifest, chunk, count, humans)
    training = chunk["training"] / session
    if not training.exists():
        process_one(session, chunk["collected"], chunk["generated"], chunk["training"], chunk["cache"],
                    count=count, collected_count=humans, device=f"cuda:{gpu}")
    final = verify_final(training, training_options(session, chunk, count, humans))
    if {r["fingerprint"] for r in final["episodes"]} != {r["fingerprint"] for r in manifest["episodes"]}:
        raise ValueError(f"{name}: training file does not match the chunk pool")
    if not pps.exists():
        pps_hdf5.export_session(pool, training, chunk["pps"])
    exported = pps_hdf5.verify_session(pps, pool, training)
    atomic_json(marker, {"sha256": exported["sha256"], "episodes": count, "verified_unix_s": time.time()})
    print(f"{name}: {count} episodes rendered and exported ({pps})", flush=True)
    if not args.keep_renders:
        for folder in (chunk["training"], chunk["cache"]):
            shutil.rmtree(folder, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True)
    parser.add_argument("--root", type=Path, help="Default: output/omy_coverage_sets/SESSION")
    parser.add_argument("--collected-root", type=Path, default=Path("output/omy_teleop"))
    parser.add_argument("--sizes", default="50,100,200")
    parser.add_argument("--base-pool", type=Path, help="Published smaller set every new set must contain")
    parser.add_argument("--base-pps", type=Path, help="PPS export of --base-pool")
    parser.add_argument("--preferred-humans-pool", type=Path,
                        help="Pool whose humans the sets keep (default: output/omy_final_pool50/SESSION if present)")
    parser.add_argument("--generator-profile", default="p4")
    parser.add_argument("--target-count", type=int, default=300, help="Explicit targets to sample and rank")
    parser.add_argument("--target-seed", type=int, default=20261113)
    parser.add_argument("--evaluation-scenes", default="assets/scenes/simple_pnp_v2/pnp_*_{object}_[0-9][0-9].usda")
    parser.add_argument("--screen-workers", type=int, default=16)
    parser.add_argument("--max-attempts-per-source", type=int, default=300)
    parser.add_argument("--min-free-gb", type=float, default=70, help="Free disk required before rendering a chunk")
    parser.add_argument("--keep-renders", action="store_true", help="Keep each chunk's training file and cache")
    parser.add_argument("--stop-after", choices=STAGES)
    args = parser.parse_args(argv)
    sizes = sorted(int(s) for s in args.sizes.split(","))
    if (args.base_pool is None) != (args.base_pps is None):
        parser.error("--base-pool and --base-pps go together")
    session = args.session
    root = (args.root or Path("output/omy_coverage_sets") / session).resolve()
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
    print(f"{session}: sizes {sizes}; GPU pool (host indices) {gpus}", flush=True)
    state = roots["state"] / session
    state.mkdir(parents=True, exist_ok=True)
    local_total, explicit_need = coverage_sets.candidate_needs(sizes[-1])

    # 1: qualification and local candidates.
    pipeline = Pipeline(roots, gpus, count=10 + local_total, collected_count=None, candidate_factor=1.0,
                        max_attempts_per_source=args.max_attempts_per_source, stop_after="generate",
                        generator_profile=args.generator_profile)
    code = pipeline.run([session])
    work = pipeline.work[0]
    print(f"generate: exit {code}; qualified {len(work.qualified)}/{len(work.sources or [])}; "
          f"local shortfall {work.tier_shortfall}; error {work.error}", flush=True)
    if not work.qualified:
        raise SystemExit("No source qualified under the generator profile")
    if args.stop_after == "generate":
        return 0

    # 2-3: explicit targets.
    targets, layouts, ranking = freeze_targets(state, work, args, work.family)
    print(f"targets: {ranking['counts']}", flush=True)
    if args.stop_after == "targets":
        return 0
    explicit = layout_arms.ExplicitTargets(work, targets, ranking, gpus, args.generator_profile, need=explicit_need)
    report = explicit.run()
    atomic_json(state / "explicit_report_coverage_sets.json", report)
    print(f"explicit: {report}", flush=True)
    if args.stop_after == "explicit":
        return 0

    # 4: nested selection, frozen once made.
    rows, rejected, tiers = layout_arms.candidate_rows(roots, work)
    by_fingerprint = {r["fingerprint"]: r for r in rows}
    names = layouts["names"]
    base = [r["fingerprint"] for r in read_json(args.base_pool / "dataset.json")["episodes"]] if args.base_pool else []
    frozen = state / "coverage_sets_selection.json"
    if not frozen.exists():
        preferred_pool = args.preferred_humans_pool or REPO / "output/omy_final_pool50" / session
        preferred = ([r["fingerprint"] for r in read_json(preferred_pool / "dataset.json")["episodes"]
                      if r["kind"] == "collected"] if (preferred_pool / "dataset.json").exists() else [])
        selected, selection = coverage_sets.select_nested(rows, tiers, work.qualified, layouts["targets"], names, sizes,
                                                          base=base, preferred_humans=preferred)
        filled = {size: value for size, value in selected.items() if "rows" in value}
        chunks = coverage_sets.render_chunks({size: value["rows"] for size, value in filled.items()}, base)
        atomic_json(frozen, {
            "schema_version": 1, "session": session, "sizes": sizes, "base_pool": str(args.base_pool or ""),
            "selection": selection,
            "sets": {str(size): {"fingerprints": [r["fingerprint"] for r in value["rows"]], "quotas": value["quotas"],
                                 "counts": value["counts"], "parents": value["parents"],
                                 "normalization": value["normalization"]} for size, value in filled.items()},
            "chunks": [[r["fingerprint"] for r in chunk] for chunk in chunks]})
    plan = read_json(frozen)
    print(f"selection: {json.dumps({k: v for k, v in plan['selection'].items() if k != 'dropped'})}", flush=True)
    common = {"schema_version": 1, "status": "complete", "session": session, "family": work.family,
              "thresholds": LIMITS, "generation_mode": "single_source", "generator_profile": args.generator_profile,
              "implementation": implementation(args.generator_profile),
              "qualified_sources": [{"input": k[0], "episode": k[1]} for k in sorted(work.qualified)],
              "targets": {"manifest": str(targets), "ranking": str(state / "ranking.json"),
                          "evaluation_layouts": str(state / "evaluation_layouts.json")}}
    set_pools = {}
    for size, chosen in sorted(plan["sets"].items(), key=lambda item: int(item[0])):
        output = root / "pools" / f"T{size}" / session
        set_pools[int(size)] = output
        if output.exists():
            continue
        picked = [by_fingerprint[f] for f in chosen["fingerprints"]]
        humans = chosen["counts"].get("collected", 0)
        manifest = {**common, "arm": f"T{size}", "selected_count": int(size), "quotas": chosen["quotas"],
                    "counts_by_stratum": chosen["counts"],
                    "counts_by_kind": {k: v for k, v in (("collected", humans), ("generated", int(size) - humans)) if v},
                    "generated_per_parent": chosen["parents"], "normalization": chosen["normalization"],
                    "contains": plan["base_pool"] or None, "selection": plan["selection"], "rejected": rejected,
                    "method": "nested coverage sets: exact stratum quotas (T recipe scaled); each set contains the "
                              "smaller one; farthest-first within each stratum against all selected; parents "
                              "balanced per stratum; one demo per explicit target; generated starts within "
                              f"{layout_arms.COINCIDENT_M} m joint-XY of an evaluation layout dropped"}
        export(output, picked, manifest)
        verify_lineage(output, verify_pool(output, int(size)), work.qualified)
        print(f"T{size}: published {output} {chosen['counts']}", flush=True)
    chunk_names = [f"c{index:02d}" for index in range(len(plan["chunks"]))]
    for name, fingerprints in zip(chunk_names, plan["chunks"]):
        output = root / "chunks/pools" / name / session
        if output.exists():
            continue
        picked = [by_fingerprint[f] for f in fingerprints]
        kinds = {k: sum(r["kind"] == k for r in picked) for k in ("collected", "generated")}
        export(output, picked, {**common, "arm": f"chunk {name}", "selected_count": len(picked),
                                "counts_by_kind": {k: v for k, v in kinds.items() if v},
                                "method": "render chunk of the nested coverage sets (not a dataset on its own)"})
        verify_pool(output, len(picked))
    print(f"select: sets {sorted(set_pools)}, {len(chunk_names)} render chunks; "
          f"unfilled {plan['selection']['unfilled']}", flush=True)
    if args.stop_after == "select":
        return 0

    # 5: render chunks (one worker per GPU), then assemble each set's PPS file.
    done = state / "coverage_sets_done.json"
    if done.exists():  # chunk renders and chunk PPS files are gone; only re-check the published sets
        for size, pool in sorted(set_pools.items()):
            coverage_sets.verify_merged(root / "pps" / f"T{size}" / session, pool)
        print(f"{session}: sets {sorted(set_pools)} already exported and verified", flush=True)
        return 0
    queue, errors, guard = list(chunk_names), [], threading.Lock()

    def worker(gpu):
        while True:  # a failed chunk does not stop the others; the run still fails and assembles nothing
            with guard:
                if not queue:
                    return
                name = queue.pop(0)
            try:
                export_chunk(name, session, roots, root, gpu, args)
            except Exception:
                with guard:
                    errors.append((name, traceback.format_exc()))

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for name, error in errors:
        print(f"{name}: export failed\n{error}", flush=True)
    if errors:
        return 1
    parts = ([args.base_pps] if args.base_pps else []) + [root / "chunks/pps" / name / session for name in chunk_names]
    for size, pool in sorted(set_pools.items()):
        output_root = root / "pps" / f"T{size}"
        if not (output_root / session).exists():
            coverage_sets.merge_pps(pool, parts, output_root)
        merged = coverage_sets.verify_merged(output_root / session, pool)
        print(f"T{size}: PPS file verified, {len(merged['episodes'])} episodes ({output_root / session})", flush=True)
    if not args.keep_renders:
        shutil.rmtree(root / "chunks/pps", ignore_errors=True)
        for folder in ("chunks/training", "chunks/cache"):
            shutil.rmtree(root / folder, ignore_errors=True)
    atomic_json(done, {"sets": {str(s): str(root / "pps" / f"T{s}" / session)
                                                             for s in sorted(set_pools)},
                                                    "unfilled": plan["selection"]["unfilled"],
                                                    "finished_unix_s": time.time()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
