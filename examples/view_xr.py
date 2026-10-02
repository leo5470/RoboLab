#!/usr/bin/env python3
"""Viewing-only SteamVR/ALVR entry point. See docs/quest_viewing.md."""

import json
import os
import sys
import traceback

from robolab.core.xr_view.options import build_parser, launcher_options, preflight, validate_args


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
    previous_runtime = os.environ.get("XR_RUNTIME_JSON")
    previous_xr = os.environ.get("XR")
    if report["runtime_json"]:
        os.environ["XR_RUNTIME_JSON"] = report["runtime_json"]
    # --desktop must not inherit a previous XR=1 shell setting.
    os.environ["XR"] = "0" if args.desktop else "1"
    app = None
    status = 1
    try:
        import cv2  # noqa: F401 -- must precede Isaac Lab
        from isaaclab.app import AppLauncher

        app = AppLauncher(launcher_options(args)).app
        from robolab.core.xr_view.runner import run

        status = run(args, app, report)
    except KeyboardInterrupt:
        status = 130
    except Exception:
        traceback.print_exc()
    finally:
        for key, previous in (("XR_RUNTIME_JSON", previous_runtime), ("XR", previous_xr)):
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous
        if app is not None:
            # Kit can exit Python during close; set the result before shutdown.
            sys.stdout.flush()
            sys.stderr.flush()
            app.app.post_quit(status)
            app.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
