#!/usr/bin/env python3
"""L100 + Quest collection with F9 reset and deferred dataset images."""

import json
import os
import signal
import sys
import traceback
from contextlib import ExitStack

from robolab.core.omy_teleop.collection import build_parser, validate_args
from robolab.core.omy_teleop.collection_options import selection_report
from robolab.core.omy_teleop.performance import collection_cuda_environment
from robolab.core.omy_teleop.xr import XRControlSession, launcher_options, preflight


def main():
    parser = build_parser()
    args = parser.parse_args()
    handoff = None
    if args.collection_handoff:
        from robolab.core.omy_teleop.collection_handoff import CollectionHandoff

        handoff = CollectionHandoff(args.collection_handoff, args.collection_run_id)
        args._collection_handoff = handoff
    validate_args(parser, args)
    report = preflight(args)
    report["collection"] = selection_report(args)
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
    collection_closed = False
    app_closed = False
    previous_sigint = signal.getsignal(signal.SIGINT)
    environment = ExitStack()
    try:
        environment.enter_context(collection_cuda_environment(args))
        import cv2  # noqa: F401 -- must precede Isaac Lab
        from isaaclab.app import AppLauncher

        app = AppLauncher(launcher_options(args)).app
        # SimulationApp installs an immediate-shutdown SIGINT handler. Collection
        # must finalize its buffered attempt before the simulator is closed.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        from robolab.core.omy_teleop.collection import run

        status = run(args, app, xr_session=XRControlSession(args, report))
        collection_closed = True
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
            try:
                if handoff is not None:
                    # Kit fast shutdown may exit natively in post_quit/close.
                    # The supervisor confirms app shutdown after process exit.
                    handoff.publish(status=status, app_shutdown_pending=(collection_closed
                                                                         and status in (0, 130)))
                app.app.post_quit(status)
                app.close()
                app_closed = True
            except KeyboardInterrupt:
                status = 130
            except Exception:
                traceback.print_exc()
                status = 1
        signal.signal(signal.SIGINT, previous_sigint)
        environment.close()
        if handoff is not None:
            handoff.publish(status=status, cleanup_complete=(collection_closed and app_closed
                                                            and status in (0, 130)))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
