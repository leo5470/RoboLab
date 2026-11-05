#!/usr/bin/env python3
"""Build single-source OMY datasets end to end; see docs/omy_mimicgen.md.

With no --session, every collected session is processed. GPUs come from
CUDA_VISIBLE_DEVICES (host GPU 0 when unset).
"""

import argparse
import fcntl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robolab.core.omy_teleop.gpu_pool import resolve_gpu_pool  # noqa: E402
from robolab.core.omy_teleop.pipeline import STOP_AFTER, Pipeline  # noqa: E402


def sessions_in(collected):
    return sorted(p.name for p in collected.iterdir()
                  if p.is_dir() and any(p.glob("*/part_*/omy_demos.hdf5")))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", action="append", help="Collection session folder; repeatable; default: all")
    parser.add_argument("--collected-root", type=Path, default=Path("output/omy_teleop"))
    parser.add_argument("--generated-root", type=Path, default=Path("output/omy_mimicgen"))
    parser.add_argument("--pool-root", type=Path, default=Path("output/omy_final_pool50"))
    parser.add_argument("--training-root", type=Path, default=Path("output/omy_training"))
    parser.add_argument("--cache-root", type=Path, default=Path("output/omy_training_cache"))
    parser.add_argument("--pps-root", type=Path, default=Path("output/omy_pps"),
                        help="PPS single-file HDF5 exports (pi05 DROID wire convention)")
    parser.add_argument("--state-root", type=Path, default=Path("output/omy_pipeline"),
                        help="Frozen sources, run status, lock and archives")
    parser.add_argument("--summary", type=Path, default=Path("output/omy_pool_completion/summary.json"),
                        help="Progress file read by the website's Dataset tab")
    parser.add_argument("--count", type=int, default=50, help="Demos per session")
    parser.add_argument("--collected-count", type=int, help="Exact human quota (default: up to 10)")
    parser.add_argument("--max-attempts-per-source", type=int, default=150,
                        help="Generation attempt budget per qualified human")
    parser.add_argument("--candidate-factor", type=float, default=1.5,
                        help="Candidates to generate per human, as a multiple of its share of generated slots")
    parser.add_argument("--stop-after", choices=STOP_AFTER)
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify and select from existing candidates only; start no Isaac process, write nothing")
    parser.add_argument("--reverify", action="store_true",
                        help="Re-hash published training files even when unchanged since their last full check")
    parser.add_argument("--replace-invalid", action="store_true",
                        help="Archive a published pool that fails verification (e.g. multisource) and rebuild it")
    args = parser.parse_args(argv)
    if args.count < 1 or (args.collected_count is not None and not 0 <= args.collected_count <= args.count):
        parser.error("Invalid --count or --collected-count")
    if args.max_attempts_per_source < 1 or args.candidate_factor < 1:
        parser.error("--max-attempts-per-source must be positive and --candidate-factor at least 1")
    roots = {name: getattr(args, f"{name}_root").resolve()
             for name in ("collected", "generated", "pools", "training", "cache", "pps", "state")
             if hasattr(args, f"{name}_root")}
    roots["pools"] = args.pool_root.resolve()
    roots["summary"] = args.summary.resolve()
    sessions = args.session or sessions_in(roots["collected"])
    gpus = [] if args.dry_run else [gpu["host_index"] for gpu in resolve_gpu_pool()]
    if gpus:
        print(f"GPU pool (host indices): {gpus}", flush=True)
    lock = None
    if not args.dry_run:
        roots["state"].mkdir(parents=True, exist_ok=True)
        lock = (roots["state"] / "pipeline.lock").open("a+")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"Another pipeline run holds {roots['state'] / 'pipeline.lock'}")
    pipeline = Pipeline(roots, gpus, count=args.count, collected_count=args.collected_count,
                        max_attempts_per_source=args.max_attempts_per_source,
                        candidate_factor=args.candidate_factor, dry_run=args.dry_run,
                        stop_after=args.stop_after, replace_invalid=args.replace_invalid,
                        reverify=args.reverify)
    return pipeline.run(sessions)


if __name__ == "__main__":
    raise SystemExit(main())
