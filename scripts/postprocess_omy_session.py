#!/usr/bin/env python3
"""Validate, package and render a finalized handoff or a closed layout session."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--handoff", type=Path)
    source.add_argument("--session", type=Path, help="Closed object-layout session, including older recordings")
    parser.add_argument("--run-id", help="Separate processing output run, or reuse an existing one")
    parser.add_argument("--device", help="Isolated host GPU in nvidia-smi order (saved selection or cuda:1)")
    parser.add_argument("--skip-failed", action="store_true", default=None, help="Process only explicitly successful attempts")
    parser.add_argument("--preview-videos", action="store_true", default=None, help="Encode success verification videos on CPU")
    parser.add_argument("--renderer-mode", choices=("batch", "per-episode"), default="batch",
                        help="Use one Isaac application per batch (default), or reference per-episode launches")
    args = parser.parse_args(argv)
    try:
        from robolab.core.omy_postprocess.session import process_closed_session, process_session

        if args.session is not None:
            return process_closed_session(args.session, run_id=args.run_id, device=args.device or "cuda:1",
                skip_failed=args.skip_failed, preview_videos=args.preview_videos, batch=args.renderer_mode == "batch")
        return process_session(args.handoff, device=args.device, skip_failed=args.skip_failed,
                               preview_videos=args.preview_videos, batch=args.renderer_mode == "batch", run_id=args.run_id)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"postprocess_omy_session: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
