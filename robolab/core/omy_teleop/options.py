"""Command-line options and provenance without simulator imports."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

UPSTREAM_COMMIT = "ed4608bd8c49133e3b9fc9d231bee651d8489b65"
UPSTREAM_URL = "https://github.com/charlie8612/omy_leader_isaaclab"
ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
PHYSICS_HZ = 120


def control_decimation(args):
    """Physics substeps per OMY command at the fixed 120 Hz physics rate."""
    return PHYSICS_HZ // args.control_hz


def build_parser(app_launcher=None, *, include_video=True, include_performance_profile=True):
    parser = argparse.ArgumentParser(description="OMY-L100 joint-space teleoperation of RoboLab DROID tasks.")
    parser.add_argument("--task", default="BananaInBowlTask")
    from robolab.constants import DEFAULT_TASK_SUBFOLDERS

    parser.add_argument("--task-dirs", "--task_dirs", nargs="+", default=list(DEFAULT_TASK_SUBFOLDERS),
                        help="Task folders relative to robolab/tasks (default: benchmark). "
                             "Uses manifest.json when present; imports only the selected task.")
    parser.add_argument("--source", choices=("serial", "tcp"), default="serial")
    parser.add_argument("--port", default="/dev/robotis_left")
    parser.add_argument("--baudrate", type=int, default=4_000_000)
    parser.add_argument("--gripper-open", type=float, default=48.2,
                        help="Leader spring rest position in plugin units (0..100).")
    parser.add_argument("--tcp-host", default="127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=5555)
    parser.add_argument("--calib", default="", help="Upstream OMY/Franka calibration JSON.")
    parser.add_argument("--auto-zero", action="store_true",
                        help="Map the held startup pose to DROID home (direct wrist mode only).")
    parser.add_argument("--auto-zero-settle-s", type=float, default=3.0)
    parser.add_argument("--stale-s", type=float, default=0.5)
    parser.add_argument("--read-hz", type=float, default=100.0)
    parser.add_argument("--control-hz", type=int, choices=(15, 30, 60), default=60,
                        help="OMY commands per simulated second (default: 60; physics stays at 120 Hz).")
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
    parser.set_defaults(stream=False, video_port=5556, video_every=2, performance_profile="standard")
    if include_video:
        parser.add_argument("--stream", action="store_true", help="Upstream JPEG-over-TCP overview + wrist video.")
        parser.add_argument("--video-port", type=int, default=5556)
        parser.add_argument("--video-every", type=int, default=2)
    if include_performance_profile:
        parser.add_argument("--performance-profile", choices=("standard", "cpu"), default="standard",
                            help="cpu: CPU physics, GPU rendering, and bounded workers. Explicit --device wins.")
    parser.add_argument("--contact-history-length", type=int, default=0,
                        help="OMY contact history (0: current forces only; 6: previous behavior).")
    parser.add_argument("--contact-sensors", choices=("full", "required"), default="required",
                        help="required derives used object-pair sensors from task consumers; "
                             "unknown consumers retain the full set.")
    parser.add_argument("--render-interval", type=int,
                        help="Physics steps between renders; auto: 2*video-every for stream-only, otherwise 2.")
    parser.add_argument("--solver-position-iterations", type=int, choices=(16, 32, 64),
                        help="OMY scene position-iteration cap; 64 permits the robot's full iteration request.")
    parser.add_argument("--gpu-max-num-partitions", type=int, choices=(1, 2, 4, 8, 16, 32), default=8,
                        help="GPU solver partition limit (default: 8); 1 is a single-robot performance candidate.")
    parser.add_argument("--placement-checks", choices=("independent", "shared"), default="independent",
                        help="Share identical single-env placement checks within an audited control boundary.")
    parser.add_argument("--task-checks", choices=("native", "cpu"), default="native",
                        help="cpu: share a CPU snapshot for audited single-env placement, event and subtask checks; "
                             "unsupported configurations retain native queries and report why.")
    parser.add_argument("--finger-colliders", choices=("original", "hulls"), default="original",
                        help="hulls: opt-in finger hulls with versioned original mass properties; CUDA only.")
    parser.add_argument("--stepping-profile", choices=("ordinary", "held-target"), default="ordinary",
                        help="held-target: audited native-15 CUDA stepping; logs a reason when falling back.")
    parser.add_argument("--recording-profile", choices=("full", "compact"), default="full",
                        help="compact: bounded state-only recording without derived EE/bounding-box channels.")
    parser.add_argument("--episode-buffer-mb", type=float, default=64.0,
                        help="Maximum allocated compact episode buffer in MiB (default: 64).")
    parser.add_argument("--stats-interval", type=float, default=5.0,
                        help="Seconds between rolling performance reports (last 300 control steps); "
                             "0 disables the entire periodic reporting block.")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Log physics-substep states and timings; stop on invalid simulated joint states.")
    parser.add_argument("--reference-observations", action="store_true",
                        help="Use the original DROID frame-sensor observations for A/B validation.")
    parser.add_argument("--diagnostic-joint-tolerance", type=float, default=0.2,
                        help="With --diagnostics, allowed excursion beyond hard joint limits in rad (default: 0.2).")
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
    if not math.isfinite(args.stats_interval) or args.stats_interval < 0:
        parser.error("--stats-interval must be finite and nonnegative (0 disables periodic reporting)")
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
    if args.contact_history_length < 0:
        parser.error("--contact-history-length must be nonnegative")
    if args.finger_colliders == "hulls" and not str(getattr(args, "device", "cuda:0")).startswith("cuda"):
        parser.error("--finger-colliders hulls requires CUDA physics")
    if not math.isfinite(args.episode_buffer_mb) or args.episode_buffer_mb <= 0:
        parser.error("--episode-buffer-mb must be finite and positive")
    if args.stepping_profile == "held-target" and args.control_hz != 15:
        parser.error("--stepping-profile held-target requires --control-hz 15")
    if args.recording_profile == "compact" and (not args.record or args.record_images or args.stream):
        parser.error("--recording-profile compact requires state-only --record without --stream")
    if not math.isfinite(args.diagnostic_joint_tolerance) or args.diagnostic_joint_tolerance < 0:
        parser.error("--diagnostic-joint-tolerance must be finite and nonnegative")
    if args.render_interval is not None:
        if args.render_interval < 2 or args.render_interval % 2:
            parser.error("--render-interval must be a positive multiple of 2 physics steps")
        if args.record_images and args.render_interval != 2:
            parser.error("--record-images requires --render-interval 2 (a fresh render before recording)")
        if args.stream and (control_decimation(args) * args.video_every) % args.render_interval:
            parser.error("--render-interval must divide control decimation * --video-every so capture follows a fresh render")

    from robolab.core.task.task_index import resolve_selected_task

    try:
        args.task_resolution = resolve_selected_task(args.task, args.task_dirs)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.task = args.task_resolution.class_name
    args.task_dirs = list(args.task_resolution.task_dirs)


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
