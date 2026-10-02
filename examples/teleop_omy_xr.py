#!/usr/bin/env python3
"""L100 joint control + the working native Quest viewer; see docs/quest_viewing.md."""

import json
import os
import sys
import traceback

from robolab.core.omy_teleop.xr import (
    XRControlSession, build_parser, launcher_options, preflight, validate_args,
)


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    report = preflight(args)
    print(json.dumps(report, indent=2), flush=True)
    if args.check:
        return 1 if report["errors"] else 0
    if report["errors"]:
        parser.error("\n".join(report["errors"]))
    previous = {key: os.environ.get(key) for key in ("XR_RUNTIME_JSON", "XR")}
    if report["runtime_json"]:
        os.environ["XR_RUNTIME_JSON"] = report["runtime_json"]
    os.environ["XR"] = "0" if args.desktop else "1"
    app = None
    status = 1
    try:
        import cv2  # noqa: F401 -- must precede Isaac Lab
        from isaaclab.app import AppLauncher

        app = AppLauncher(launcher_options(args)).app
        from robolab.core.omy_teleop.runner import run

        status = run(args, app, xr_session=XRControlSession(args, report))
    except KeyboardInterrupt:
        status = 130
    except Exception:
        traceback.print_exc()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if app is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            app.app.post_quit(status)
            app.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
