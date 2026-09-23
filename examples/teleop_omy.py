#!/usr/bin/env python3
"""Run the upstream OMY-L100 device on RoboLab tasks; see docs/omy_teleoperation.md."""

# isort: skip_file
import cv2  # noqa: F401 -- must precede Isaac Lab imports
import sys
import traceback
from isaaclab.app import AppLauncher

from robolab.core.omy_teleop.options import build_parser, validate_args


def main():
    parser = build_parser(AppLauncher)
    args = parser.parse_args()
    validate_args(parser, args)
    try:
        from importlib.metadata import version
        version("omy-leader-isaaclab")
    except ModuleNotFoundError:
        parser.error("Install requirements-omy.txt into the RoboLab Python environment first.")
    args.enable_cameras = args.stream or args.record_images
    app = AppLauncher(args).app
    status = 1
    try:
        from robolab.core.omy_teleop.runner import run
        status = run(args, app)
    except KeyboardInterrupt:
        status = 130
    except Exception:
        # Kit's fast shutdown can exit inside close(), before Python reports an
        # unhandled exception. Report first and pass our status to Kit explicitly.
        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        app.app.post_quit(status)
        app.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
