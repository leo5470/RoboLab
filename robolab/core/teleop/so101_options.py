"""SO-101 CLI validation that runs before launching Isaac Sim."""

import math
import os
import sys
from pathlib import Path

from .retarget import RetargetConfig


def add_arguments(parser):
    parser.add_argument("--leader-port", help="Server USB device, preferably /dev/serial/by-id/...")
    parser.add_argument("--leader-id", help="ID used for LeRobot calibration")
    parser.add_argument("--leader-calibration-dir")
    parser.add_argument("--leader-urdf", help="Calibrated SO-101 URDF; meshes are not needed")
    parser.add_argument("--leader-python", default=sys.executable, help="Python in the separate LeRobot environment")
    parser.add_argument("--leader-read-hz", type=float, default=60)
    parser.add_argument("--leader-read-retries", type=int, default=2)
    parser.add_argument("--leader-startup-timeout", type=float, default=20)
    parser.add_argument("--leader-joint-signs", type=float, nargs=5, default=[1] * 5)
    parser.add_argument("--leader-joint-offsets-deg", type=float, nargs=5, default=[0] * 5)
    parser.add_argument("--leader-mock", action="store_true", help="Synthetic USB-free source; requires --no-record")
    parser.add_argument(
        "--teleop-config", default=str(Path(__file__).resolve().parents[3] / "configs/teleop/so101_droid.yaml")
    )
    parser.add_argument("--orientation-mode", choices=("fixed", "pose", "wrist-roll"), default=None)
    parser.add_argument(
        "--control-source",
        choices=("evdev", "terminal"),
        default="evdev",
        help="evdev: server-connected USB keyboard; terminal: foreground TTY, opt-in",
    )
    parser.add_argument("--control-device", help="/dev/input/by-id/...-event-kbd on the server")
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Automatically start and exit after N actions; requires --leader-mock --no-record",
    )


def validate_arguments(parser, args):
    if args.leader_mock and args.record:
        parser.error("--leader-mock requires --no-record; synthetic demos must never enter the dataset")
    if args.smoke_steps < 0 or (args.smoke_steps and (not args.leader_mock or args.record)):
        parser.error("--smoke-steps requires --leader-mock --no-record and a positive count")
    if not args.realtime:
        parser.error("SO-101 interactive collection requires real-time pacing")
    if not math.isfinite(args.leader_read_hz) or not 0 < args.leader_read_hz <= 500:
        parser.error("--leader-read-hz must be in (0, 500]")
    if not 0 <= args.leader_read_retries <= 10:
        parser.error("--leader-read-retries must be between 0 and 10")
    if not math.isfinite(args.leader_startup_timeout) or args.leader_startup_timeout <= 0:
        parser.error("--leader-startup-timeout must be positive")
    if any(value not in (-1, 1) for value in args.leader_joint_signs):
        parser.error("--leader-joint-signs must be +1 or -1")
    if not all(math.isfinite(value) for value in args.leader_joint_offsets_deg):
        parser.error("--leader-joint-offsets-deg must be finite")
    if not args.benchmark_steps:
        if not args.leader_mock:
            if not all((args.leader_port, args.leader_id, args.leader_urdf)):
                parser.error("SO-101 hardware requires --leader-port, --leader-id and --leader-urdf")
            if not Path(args.leader_urdf).is_file():
                parser.error(f"URDF does not exist: {args.leader_urdf}")
            if not os.access(args.leader_port, os.R_OK | os.W_OK):
                parser.error(f"USB port is missing or inaccessible: {args.leader_port}")
        if not args.smoke_steps and args.control_source == "evdev" and not args.control_device:
            parser.error("--control-device is required for server-local controls (or select --control-source terminal)")
    try:
        args.retarget_config = RetargetConfig.load(args.teleop_config, args.orientation_mode)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


def reader_arguments(args):
    options = ["--read-hz", str(args.leader_read_hz), "--read-retries", str(args.leader_read_retries)]
    if args.leader_mock:
        return options + ["--mock"]
    options += [
        "--port",
        args.leader_port,
        "--id",
        args.leader_id,
        "--urdf",
        args.leader_urdf,
        "--joint-signs",
        *map(str, args.leader_joint_signs),
        "--joint-offsets-deg",
        *map(str, args.leader_joint_offsets_deg),
    ]
    if args.leader_calibration_dir:
        options += ["--calibration-dir", args.leader_calibration_dir]
    return options
