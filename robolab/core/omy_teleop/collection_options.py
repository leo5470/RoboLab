"""Collection selection and resume contracts without simulator imports."""

import hashlib
import importlib.metadata
from pathlib import Path
import sys


CONTRACT_OPTIONS = (
    "device", "seed", "control_hz", "episode_length_s", "max_attempt_steps", "max_attempt_seconds",
    "stepping_profile", "recording_profile", "episode_buffer_mb", "contact_sensors", "contact_history_length",
    "solver_position_iterations", "render_interval", "vel_scale", "gripper_open", "source", "baudrate",
    "gpu_max_num_partitions",
    "placement_checks", "finger_colliders", "task_checks", "cuda_connections", "physics_workers", "cpu_workers",
    "read_hz", "stale_s", "ready_tolerance_rad", "start_motion_rad", "ready_stable_s", "reference_observations",
    "scene_settle_min_s", "scene_settle_stable_s", "scene_settle_timeout_s",
    "scene_settle_linear_m_s", "scene_settle_angular_rad_s",
)
PRE_LATENCY_CONTRACT_OPTIONS = tuple(name for name in CONTRACT_OPTIONS if name not in (
    "task_checks", "cuda_connections", "physics_workers", "cpu_workers"))
PRE_OPTIMIZATION_CONTRACT_OPTIONS = tuple(
    name for name in PRE_LATENCY_CONTRACT_OPTIONS if name not in ("placement_checks", "finger_colliders")
)
LEGACY_CONTRACT_OPTIONS = tuple(name for name in PRE_OPTIMIZATION_CONTRACT_OPTIONS if name != "gpu_max_num_partitions")


def add_arguments(parser):
    parser.allow_abbrev = False
    # None distinguishes an explicit selection/quota from defaults.
    parser.set_defaults(task=None, num_episodes=None)
    parser.add_argument("--object", dest="object_name", help="Collect all retained layouts for this simple-PnP object.")
    parser.add_argument("--object-id", type=int, help="Catalog index; disambiguates duplicate object names.")
    parser.add_argument("--episodes-per-layout", type=int, help="Successful episodes per layout (object mode default: 1).")
    parser.add_argument("--resume", type=Path, help="Resume saved total/per-layout quotas from an existing session directory.")
    parser.add_argument("--reset-key", default="F9", help="Unmodified reset/READY key (default: F9).")
    parser.add_argument("--next-layout-key", default="F12", help="Unmodified next-layout key (object mode, default: F12).")
    original_parse = parser.parse_args

    def parse_args(args=None, namespace=None):
        argv = list(sys.argv[1:] if args is None else args)
        parsed = original_parse(argv, namespace)
        parsed._explicit_options = {value.split("=", 1)[0] for value in argv if value.startswith("--")}
        return parsed

    parser.parse_args = parse_args


def run_contract(args):
    from .preparation import PreparationSettings

    stack = {}
    for package in ("isaaclab", "isaacsim"):
        try:
            stack[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            stack[package] = None
    names = getattr(args, "_omy_contract_options", None) or (
        LEGACY_CONTRACT_OPTIONS if getattr(args, "_legacy_gpu_partition_contract", False) else CONTRACT_OPTIONS
    )
    contract = {
        "options": {name: getattr(args, name) for name in names},
        "calibration_sha256": hashlib.sha256(Path(args.calib).read_bytes()).hexdigest(),
        "stack": stack,
        "preparation": PreparationSettings.from_args(args).metadata(),
    }
    if "finger_colliders" in names and args.finger_colliders == "hulls":
        from .finger_colliders import finger_collider_identity
        contract["finger_collider_reference"] = finger_collider_identity()
    return contract


def session_metadata(args):
    return {
        **run_contract(args), "calibration_path": str(Path(args.calib).resolve()),
        "connection": {name: getattr(args, name) for name in ("port", "tcp_host", "tcp_port")},
    }


def configure_selection(parser, args):
    from .collection_controls import validate_keys
    from .layouts import select_layouts

    args.reset_key, args.next_layout_key = validate_keys(args.reset_key, args.next_layout_key)
    args.layout_plan = None
    args.target_total_successes = None
    args._legacy_gpu_partition_contract = False
    args._omy_contract_options = CONTRACT_OPTIONS
    group = args.object_name is not None or args.object_id is not None
    if args.resume is not None:
        from .layout_session import LayoutSession

        if group or args.task is not None or args.num_episodes is not None or args.episodes_per_layout is not None:
            parser.error("--resume cannot be combined with task/object selection or episode quotas")
        if args.output_dir:
            parser.error("--resume uses its session directory; omit --output-dir")
        plan, quota, metadata, current = LayoutSession.inspect(args.resume)
        saved_options = metadata.get("options")
        recognized = (CONTRACT_OPTIONS, PRE_LATENCY_CONTRACT_OPTIONS,
                      PRE_OPTIMIZATION_CONTRACT_OPTIONS, LEGACY_CONTRACT_OPTIONS)
        if not isinstance(saved_options, dict) or set(saved_options) not in tuple(set(names) for names in recognized):
            raise ValueError("Saved session is missing its complete collection/preparation configuration; start a new session")
        args._omy_contract_options = next(names for names in recognized if set(names) == set(saved_options))
        explicit = getattr(args, "_explicit_options", set())
        # Preserve older sessions' exact contract and effective behavior.
        # New defaults apply only when creating a new session.
        for name, value in {"gpu_max_num_partitions": 8, "placement_checks": "independent",
                            "finger_colliders": "original", "task_checks": "native",
                            "cuda_connections": 8, "physics_workers": 2, "cpu_workers": 8}.items():
            if name not in saved_options:
                flag = "--" + name.replace("_", "-")
                if flag in explicit and getattr(args, name) != value:
                    parser.error(f"{flag} conflicts with the resumed session ({value})")
                setattr(args, name, value)
        args._legacy_gpu_partition_contract = "gpu_max_num_partitions" not in saved_options
        for name, value in saved_options.items():
            flag = "--" + name.replace("_", "-")
            action = parser._option_string_actions[flag]
            if value is not None and action.choices is not None and value not in action.choices:
                raise ValueError(f"Saved value for {flag} is invalid: {value!r}")
            expected = getattr(args, name)
            unset_optional = value is None and name in ("solver_position_iterations", "render_interval")
            if not unset_optional and ((isinstance(expected, bool) and type(value) is not bool)
                    or (type(expected) is int and type(value) is not int)
                    or (action.type is int and value is not None and type(value) is not int)
                    or (type(expected) is float and (type(value) not in (int, float)))
                    or (isinstance(expected, str) and not isinstance(value, str))):
                raise ValueError(f"Saved value for {flag} has an invalid type")
            if flag in explicit and getattr(args, name) != value:
                parser.error(f"{flag} conflicts with the resumed session ({value!r})")
            setattr(args, name, value)
        connection = metadata.get("connection", {})
        if not isinstance(connection, dict) or set(connection) - {"port", "tcp_host", "tcp_port"}:
            raise ValueError("Saved session has an invalid leader connection configuration")
        for name, value in connection.items():
            if ((name == "tcp_port" and type(value) is not int)
                    or (name != "tcp_port" and not isinstance(value, str))):
                raise ValueError(f"Saved leader connection has an invalid {name}")
            if "--" + name.replace("_", "-") not in explicit:
                setattr(args, name, value)
        if "--calib" not in explicit and metadata.get("calibration_path"):
            recorded_calibration = Path(metadata["calibration_path"])
            if recorded_calibration.is_file():
                args.calib = str(recorded_calibration)
        contract = run_contract(args)
        if any(contract[key] != metadata.get(key) for key in contract):
            parser.error("Resume configuration, calibration, or simulator stack differs from the saved session")
        args.layout_plan = plan
        args.num_episodes = quota
        args.target_total_successes = metadata.get("target_total_successes")
        args.task = (current or plan.layouts[0]).task
        args.task_dirs = ["single_pnp"]
    elif group:
        if args.task is not None:
            parser.error("Object collection cannot be combined with --task")
        args.layout_plan = select_layouts(args.object_name, args.object_id)
        args.target_total_successes = args.num_episodes
        args.num_episodes = 1 if args.episodes_per_layout is None else args.episodes_per_layout
        args.task = args.layout_plan.layouts[0].task
        args.task_dirs = ["single_pnp"]
    else:
        if args.episodes_per_layout is not None:
            parser.error("--episodes-per-layout requires --object or --object-id")
        args.task = args.task or "BananaInBowlTask"
        args.num_episodes = 1000 if args.num_episodes is None else args.num_episodes
    if args.num_episodes < 1:
        parser.error("Episode quotas must be positive")
    if args.target_total_successes is not None:
        from .layout_session import LayoutSession

        try:
            LayoutSession.validate_total_target(
                args.target_total_successes, args.num_episodes, len(args.layout_plan.layouts))
        except ValueError as exc:
            parser.error(str(exc))


def selection_report(args):
    plan = getattr(args, "layout_plan", None)
    runtime = {name: getattr(args, name) for name in (
        "physics_workers", "cpu_workers", "cuda_connections", "task_checks", "control_hz",
        "stepping_profile", "recording_profile")}
    if plan is None:
        return {"mode": "task", "task": args.task, "target_successes": args.num_episodes,
                "reset_key": args.reset_key, "runtime": runtime}
    return {"mode": "object", "object": plan.object_name, "catalog_index": plan.catalog_index,
            "layouts": len(plan.layouts), "episodes_per_layout": args.num_episodes,
            "target_total_successes": args.target_total_successes,
            "resume": str(args.resume) if args.resume else None,
            "reset_key": args.reset_key, "next_layout_key": args.next_layout_key, "runtime": runtime}
