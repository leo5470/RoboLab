"""L100/XR composition; no simulator or hardware imports until launch."""

import importlib.metadata
import os
import time
from pathlib import Path

from robolab.core.xr_view.options import (
    add_view_arguments,
    launcher_options as view_launcher_options,
    preflight as view_preflight,
    validate_args as validate_view_args,
)

from .options import build_parser as omy_parser, validate_args as validate_omy_args


def build_parser():
    parser = omy_parser(include_video=False, include_performance_profile=False)
    parser.description = "OMY-L100 control with native SteamVR/ALVR head-tracked viewing."
    add_view_arguments(parser, desktop_help="Run L100 control in the desktop viewport without XR; not a headset test.")
    parser.add_argument("--device", default="cpu", help="Physics device: cpu (default), cuda, or cuda:N.")
    parser.set_defaults(
        calib=str(Path(__file__).resolve().parents[3] / "omy_calib_robolab.json"),
        performance_profile="cpu", render_interval=2,
    )
    return parser


def validate_args(parser, args):
    validate_omy_args(parser, args)
    validate_view_args(parser, args)
    if args.render_interval not in (2, 4):
        parser.error("XR supports --render-interval 2 (60 Hz) or 4 (30 Hz)")
    if (120 // args.control_hz) % args.render_interval:
        parser.error("XR render interval must divide the control block; interval 4 requires 15 or 30 Hz control")


def preflight(args):
    report = view_preflight(args)
    report["robot_control"] = "omy_l100"
    report["calibration"] = str(Path(args.calib).resolve()) if args.calib else None
    report["source"] = args.source
    try:
        report["omy_version"] = importlib.metadata.version("omy-leader-isaaclab")
    except importlib.metadata.PackageNotFoundError:
        report["errors"].append("Install requirements-omy.txt and run scripts/patch_omy.py first.")
    if args.source == "serial":
        port = Path(args.port)
        if not port.exists():
            report["errors"].append(f"Leader port not found: {port}")
        elif not os.access(port, os.R_OK | os.W_OK):
            report["errors"].append(f"Leader port is not readable/writable: {port}; check the desktop user's dialout group.")
    report["warnings"].append("Preflight does not open the leader or validate live samples. Hold the calibrated READY pose at startup.")
    return report


def launcher_options(args):
    options = view_launcher_options(args)
    options["enable_cameras"] = args.record_images
    options["rendering_mode"] = args.rendering_mode
    if args.control_hz < 60:
        options["kit_args"] += " --/app/runLoops/main/rateLimitEnabled=false"
    return options


class XRControlSession:
    """Gate the existing control loop on native XR display activation.

    While inactive, only render() runs: no mapper advance, physics, or recorded
    actions. Reconnection retains the mapper/last command so the existing rate
    limiter handles any leader movement during the pause; it does not recenter.
    Kit display activation is not a headset-worn or transport-health signal.
    """

    def __init__(self, args, report, *, view_factory=None, clock=time.monotonic):
        self.args = args
        self.clock = clock
        self.view_factory = view_factory
        self.view = None
        self.metadata = {
            "mode": report["mode"], "runtime_json": report["runtime_json"],
            "eye": args.eye, "lookat": args.lookat, "render_gpu": args.render_gpu,
            "xr_resolution_multiplier_override": args.xr_resolution_multiplier,
            "robot_control": "omy_l100", "xr_display_seen": False,
            "display_waits": 0, "completed": False,
            "inactive_behavior": "freeze_physics_and_mapper_until_display_active",
            "resume_behavior": "retain_calibration_and_rate_limited_last_command",
        }

    def configure(self, cfg):
        cfg.viewer.eye = tuple(self.args.eye)
        cfg.viewer.lookat = tuple(self.args.lookat)

    def start(self):
        if self.args.desktop:
            return
        if self.view_factory is None:
            from robolab.core.xr_view.native import create_native_view

            self.view_factory = create_native_view
        self.view = self.view_factory(self.args, self.metadata["runtime_json"])
        self.view.start()

    def ready(self):
        if self.args.desktop:
            return True
        if self.view is None:
            return False
        if self.view.update():
            print("XR display active; L100 control enabled." if self.view.active else
                  "XR display inactive; L100 control and physics paused.", flush=True)
        self.metadata["xr_display_seen"] = self.view.ever_active
        return self.view.active

    def wait_for_display(self, app, render):
        if self.ready():
            return app.is_running()
        self.metadata["display_waits"] += 1
        print("Waiting for XR display; physics and L100 commands are paused.", flush=True)
        start = self.clock()
        while app.is_running():
            render()
            if self.ready():
                # Process schedule_set_camera before opening/advancing input.
                render()
                if self.ready():
                    return app.is_running()
            if self.clock() - start >= self.args.xr_timeout:
                raise RuntimeError("XR display inactive for too long; stopping L100 teleoperation. "
                                   "Check SteamVR/ALVR or increase --xr-timeout.")
        return False

    def close(self, render):
        if self.view is not None:
            try:
                self.metadata["xr_display_seen"] = self.view.ever_active
                self.view.close()
                render()
            finally:
                self.view = None
