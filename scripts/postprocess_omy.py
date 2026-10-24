#!/usr/bin/env python3
"""Offline OMY indexing, archival, camera rendering, and strict native-15 rows.

Run --help without starting Isaac Sim. No command opens the leader serial port.
Export is explicitly recipe-gated; diagnostics are not training-ready datasets.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

# Keep this checkout authoritative even when invoked by an absolute script path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "validate"):
        command = sub.add_parser(name, help="Read-only structural/native-15 inspection")
        command.add_argument("--input", required=True, type=Path)
        command.add_argument("--output", type=Path, help="New JSON report path; existing files are refused")
        command.add_argument("--require-training-eligible", action="store_true",
                             help="Exit nonzero for outcomes/quality warnings unsuitable for automatic training selection")
        command.add_argument("--require-native15", action="store_true",
                             help="Exit nonzero unless every episode's recorded intervals verify native 15 Hz")
    package = sub.add_parser("package", help="Publish immutable one-attempt archives; preserve sources")
    package.add_argument("--input", required=True, type=Path)
    package.add_argument("--output", required=True, type=Path, help="Archive root; retry verifies existing matching attempts")
    package.add_argument("--episode", action="append", type=int, help="Zero-based demo index; repeatable")
    package.add_argument("--min-free-mb", type=float, default=64)
    render = sub.add_parser("render", help="Render pre-action policy cameras in a separate headless process")
    render.add_argument("--input", required=True, type=Path)
    render.add_argument("--output", required=True, type=Path, help="New per-attempt render cache directory")
    render.add_argument("--episode", type=int, default=0)
    render.add_argument("--python", type=Path, help="Known-working Isaac Lab interpreter; defaults to current interpreter")
    render.add_argument("--device", default="cuda:0", help="Isolated host GPU in nvidia-smi order")
    render.add_argument("--renderer", choices=("realtime", "pathtracing"), default="realtime")
    render.add_argument("--camera-scale", type=float, default=1.0)
    render.add_argument("--camera-width", type=int)
    render.add_argument("--camera-height", type=int)
    render.add_argument("--renders-per-frame", type=int, default=2)
    render.add_argument("--min-free-mb", type=float, default=64)
    rows = sub.add_parser("rows", help="Verify/write canonical PPS numerical rows; never training certification")
    rows.add_argument("--input", required=True, type=Path)
    rows.add_argument("--output", required=True, type=Path)
    rows.add_argument("--recipe", required=True, type=Path, help="Explicit gripper-state/command mapping recipe")
    rows.add_argument("--episode", type=int, default=0)
    rows.add_argument("--diagnostic", action="store_true", help="Retain failed/flagged attempts for tests, not training")
    rows.add_argument("--review", type=Path, help="Source-bound recorded quality review, not an untracked bypass")
    export = sub.add_parser("export", help="Recipe-gated local PPS/LeRobot conversion; not loader certification")
    export.add_argument("--selection", required=True, type=Path,
                        help="Versioned JSON episode list with explicit roles, splits and persistent identities")
    export.add_argument("--recipe", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--repo-id", required=True, help="Local dataset identity; no upload occurs")
    return parser


def load_selections(path):
    """Resolve paths relative to a versioned selection manifest, not the CWD."""
    from robolab.core.omy_postprocess.export import EpisodeSelection

    path = Path(path).resolve()
    contents = json.loads(path.read_text())
    if not isinstance(contents, dict) or contents.get("schema_version") != 1:
        raise ValueError("Selection manifest must be an object with schema_version=1")
    episodes = contents.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Selection must explicitly list at least one episode")
    selected = []
    for item in episodes:
        if not isinstance(item, dict):
            raise ValueError("Every selection must be an episode object")
        for name in ("source", "group", "attempt_id", "attempt_uuid", "split", "task_instruction"):
            if not isinstance(item.get(name), str) or not item[name]:
                raise ValueError(f"Selection is missing explicit {name}")
        roles = item.get("roles")
        if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
            raise ValueError("Selection roles must be an explicit list")
        review = item.get("review")
        if review is not None and not isinstance(review, dict):
            raise ValueError("Selection review must be a source-bound JSON object")
        source = Path(item["source"])
        if not source.is_absolute():
            source = path.parent / source
        selected.append(EpisodeSelection(source=source.resolve(), group=item["group"],
                                        attempt_id=item["attempt_id"], attempt_uuid=item["attempt_uuid"],
                                        split=item["split"], roles=tuple(roles),
                                        task_instruction=item["task_instruction"], review=review))
    return selected


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in ("index", "validate"):
            from robolab.core.omy_postprocess.io import atomic_write_json
            from robolab.core.omy_postprocess.validation import validate_recording

            report = validate_recording(args.input)
            if args.output:
                atomic_write_json(args.output, report)
            printed = report
            if args.output:
                printed = {"report": str(args.output), "source": report["source"],
                           "errors": report["errors"], "episodes": [
                               {key: item[key] for key in ("group", "sample_count", "outcome", "structural_valid",
                                                          "native_15_hz", "training_eligible", "errors", "warnings")}
                               for item in report["episodes"]]}
            print(json.dumps(printed, indent=2, allow_nan=False))
            invalid = bool(report["errors"] or any(not e["structural_valid"] for e in report["episodes"]))
            if args.require_native15:
                invalid |= any(not e["native_15_hz"] for e in report["episodes"])
            if args.require_training_eligible:
                invalid |= any(not e["training_eligible"] for e in report["episodes"])
            return 2 if invalid else 0
        if args.command == "package":
            from robolab.core.omy_postprocess.archive import package_recording

            result = package_recording(args.input, args.output,
                                       episodes=[f"demo_{i}" for i in args.episode] if args.episode else None,
                                       min_free_bytes=int(args.min_free_mb * 1024**2))
            print(json.dumps(result, indent=2))
        elif args.command == "render":
            from robolab.core.omy_postprocess.render import render_recording

            result = render_recording(args.input, args.output, group=f"data/demo_{args.episode}",
                                      python=args.python, device=args.device, renderer=args.renderer,
                                      camera_scale=args.camera_scale, camera_width=args.camera_width,
                                      camera_height=args.camera_height, renders_per_frame=args.renders_per_frame,
                                      min_free_bytes=int(args.min_free_mb * 1024**2))
            print(result)
        elif args.command == "rows":
            from robolab.core.omy_postprocess.export import export_canonical_rows, load_recipe

            review = json.loads(args.review.read_text()) if args.review else None
            result = export_canonical_rows(args.input, args.output, group=f"data/demo_{args.episode}",
                                           recipe=load_recipe(args.recipe), diagnostic=args.diagnostic, review=review)
            print(result)
        elif args.command == "export":
            from robolab.core.omy_postprocess.export import export_lerobot, load_recipe

            result = export_lerobot(load_selections(args.selection), args.output,
                                   recipe=load_recipe(args.recipe), repo_id=args.repo_id)
            print(result)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"postprocess_omy: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
