#!/usr/bin/env python3
"""Expand L100 teleoperation demonstrations with MimicGen; see docs/omy_mimicgen.md."""

# isort: skip_file
import cv2  # noqa: F401 -- must precede Isaac Lab imports
import sys
import traceback
from isaaclab.app import AppLauncher

from robolab.core.omy_teleop.mimicgen import build_parser, validate_args


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    app = AppLauncher(headless=True, device=args.device).app
    status = 1
    try:
        from robolab.core.omy_teleop.mimicgen import run

        status = run(args)
    except KeyboardInterrupt:
        status = 130
    except Exception:
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        app.app.post_quit(status)
        app.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
