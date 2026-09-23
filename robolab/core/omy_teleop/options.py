"""Command-line options and provenance without simulator imports."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

UPSTREAM_COMMIT = "ed4608bd8c49133e3b9fc9d231bee651d8489b65"
UPSTREAM_URL = "https://github.com/charlie8612/omy_leader_isaaclab"
ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))


def build_parser(app_launcher=None):
    parser = argparse.ArgumentParser(description="OMY-L100 joint-space teleoperation of RoboLab DROID tasks.")
    parser.add_argument("--task", default="BananaInBowlTask")
    parser.add_argument("--source", choices=("serial", "tcp"), default="serial")
    parser.add_argument("--port", default="/dev/robotis_left")
    parser.add_argument("--baudrate", type=int, default=4_000_000)
    parser.add_argument("--gripper-open", type=float, default=48.2,
                        help="Leader spring rest position in plugin units (0..100).")
    parser.add_argument("--tcp-host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=5555)
    parser.add_argument("--calib", default="", help="Upstream OMY/Franka calibration JSON.")
    parser.add_argument("--auto-zero", action="store_true", help="Use upstream startup auto-zero unchanged.")
    parser.add_argument("--auto-zero-settle-s", type=float, default=3.0)
    parser.add_argument("--stale-s", type=float, default=0.5)
    parser.add_argument("--read-hz", type=float, default=100.0)
    parser.add_argument("--vel-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=0, help="Stop after this many total actions (0: no limit).")
    parser.add_argument("--num-episodes", type=int, default=1, help="Attempts, including task timeouts.")
    parser.add_argument("--episode-length-s", type=float, default=10_000.0,
                        help="Simulation-time budget per attempt (upstream teleop default: 10000 seconds).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--record", action="store_true", help="Record actions, states, and proprioception to HDF5.")
    parser.add_argument("--record-images", action="store_true", help="Record live DROID policy-camera images.")
    parser.add_argument("--save-failures", action="store_true")
    parser.add_argument("--output-dir", help="New run directory; an existing nonempty directory is refused.")
    parser.add_argument("--stream", action="store_true", help="Upstream JPEG-over-TCP overview + wrist video.")
    parser.add_argument("--video-port", type=int, default=5556)
    parser.add_argument("--video-every", type=int, default=2)
    if app_launcher is not None:
        app_launcher.add_app_launcher_args(parser)
    else:
        parser.add_argument("--rendering_mode", choices=("performance", "balanced", "quality"))
    parser.set_defaults(rendering_mode="performance")
    return parser


def validate_args(parser, args):
    import math

    for name in ("stale_s", "read_hz", "vel_scale", "auto_zero_settle_s", "episode_length_s"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.steps < 0 or args.num_episodes < 1 or args.video_every < 1 or args.baudrate < 1:
        parser.error("steps must be nonnegative; episodes, video-every, and baudrate must be positive")
    if not math.isfinite(args.gripper_open) or not 0 <= args.gripper_open <= 100:
        parser.error("--gripper-open must be finite and in [0, 100]")
    if any(not 1 <= port <= 65535 for port in (args.tcp_port, args.video_port)):
        parser.error("TCP/video ports must be between 1 and 65535")
    if args.record_images and not args.record:
        parser.error("--record-images requires --record")
    if args.save_failures and not args.record:
        parser.error("--save-failures requires --record")
    if args.calib and not Path(args.calib).is_file():
        parser.error(f"Calibration not found: {args.calib}")


def upstream_metadata():
    """Capture the installed revision, including overrides of the recommended pin."""
    from omy_leader_isaaclab.omy_serial import TRIGGER_FIX_VERSION
    distribution = importlib.metadata.distribution("omy-leader-isaaclab")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    return {
        "repository": UPSTREAM_URL,
        "recommended_commit": UPSTREAM_COMMIT,
        "installed_version": distribution.version,
        "installed_commit": direct_url.get("vcs_info", {}).get("commit_id"),
        "local_trigger_fix": TRIGGER_FIX_VERSION,
        "gripper_reading_units": "legacy_half_angle_pi_per_4095_ticks",
    }


def calibration_metadata(path):
    if not path:
        return {"path": None, "sha256": None, "contents": None}
    data = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(data).hexdigest(), "contents": json.loads(data)}
