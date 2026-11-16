"""MimicGen expansion of OMY teleoperation demonstrations.

Generation runs in the teleoperation environment itself: Isaac Lab Mimic plans
end-effector waypoints, and the environment converts each one to the same
absolute joint command the L100 sends. Generated episodes therefore share the
teleoperation HDF5 contract. See docs/omy_mimicgen.md.
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from .options import ARM_JOINTS, PHYSICS_HZ
from .options import build_parser as teleop_parser

EEF = "robot"
EE_BODY = "base_link"
GRASP_SIGNAL = "grasp"
# The six-axis L100 never commands panda_joint3; IK keeps it at its home value.
LOCKED_JOINT = "panda_joint3"
MAX_JOINT_STEP = 0.5

from .mimic_sources import STRATEGIES, checksum, source_references, trajectory_rng
# Source reading lives in mimic_sources (one reader per source kind); re-exported for existing callers.
from .mimic_sources import grasp_index, pre_action, read_source, source_arrays  # noqa: F401


def build_parser():
    parser = argparse.ArgumentParser(description="Expand OMY teleoperation demonstrations with MimicGen.")
    parser.add_argument("--input", nargs="+", required=True,
                        help="Teleoperation run directories (omy_demos.hdf5 with env_cfg.json).")
    parser.add_argument("--episodes", type=int, nargs="+",
                        help="Source episodes to use, as demo_<N> indices of a single --input; default: all successful.")
    parser.add_argument("--source-selection", help="JSON sources list with exact file/episode references and optional sha256")
    parser.add_argument("--target-input", help="Recording directory that defines the target scene independently of the source pool")
    parser.add_argument("--target-episode", type=int, help="Successful target recording episode used as the reset anchor")
    parser.add_argument("--selection-strategy", choices=STRATEGIES, default="random")
    parser.add_argument("--nn-k", type=int, default=3, help="Choose among this many nearest sources (clamped to pool size)")
    parser.add_argument("--selection-seed", type=int, default=0, help="Independent trajectory/source-selection seed")
    parser.add_argument("--fixed-source-index", type=int, help="Force one pool source for qualification or a single-source baseline")
    parser.add_argument("--trial-manifest", help="JSON trials list with fixed ids and scene seeds for paired comparisons")
    parser.add_argument("--follow-source-schedule", action="store_true", help="Use each trial's predetermined source_index as the baseline")
    parser.add_argument("--select-source-per-subtask", action="store_true",
                        help="Choose the placement segment's source separately from the grasp segment's. A fixed or "
                             "scheduled source then applies to the grasp only; a trial's placement_source_index, "
                             "if given, fixes the placement source.")
    parser.add_argument("--target-manifest",
                        help="JSON targets list giving start poses by asset name; the scene, task and success "
                             "rule still come from the target or first input recording. Requires --target-id.")
    parser.add_argument("--target-id", help="Row of --target-manifest to generate at")
    parser.add_argument("--output-dir", help="New run directory; an existing nonempty directory is refused.")
    parser.add_argument("--num-demos", type=int, default=40, help="Successful demonstrations to generate.")
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--pose-range", type=float, default=0.1,
                        help="Uniform x/y reset offset in meters for the object and container.")
    parser.add_argument("--yaw-range", type=float, default=0.0, help="Uniform reset yaw offset in radians.")
    parser.add_argument("--object-pose-range", type=float,
                        help="x/y reset offset in meters for the grasped object alone; default: --pose-range.")
    parser.add_argument("--object-yaw-range", type=float,
                        help="Reset yaw offset in radians for the grasped object alone; default: --yaw-range.")
    parser.add_argument("--no-randomize", action="store_true",
                        help="Keep the recorded arrangement; generation then reproduces the source demonstrations.")
    parser.add_argument("--action-noise", type=float, default=0.0,
                        help="Standard deviation in meters of noise on each target position.")
    parser.add_argument("--interpolation-s", type=float, default=1.0,
                        help="Simulated seconds for approaching each transformed segment.")
    parser.add_argument("--settle-s", type=float, default=2.0,
                        help="Simulated seconds to hold the final target while waiting for success.")
    parser.add_argument("--object", help="Grasped object; default: the success term's object.")
    parser.add_argument("--container", help="Placement reference; default: the success term's container.")
    parser.add_argument("--keep-failed", action="store_true", help="Also export failed attempts.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-exit", action="store_true",
                        help="Exit the dedicated headless worker after closing recordings and the environment; "
                             "avoids Isaac Sim 5.0 plugin-teardown crashes.")
    from .preparation import add_arguments
    add_arguments(parser)
    from .repairs import add_arguments as add_repair_arguments
    add_repair_arguments(parser)
    return parser


def validate_args(parser, args):
    if args.num_demos < 1 or args.max_attempts < args.num_demos:
        parser.error("--num-demos must be positive and no greater than --max-attempts")
    if args.object_pose_range is None:
        args.object_pose_range = args.pose_range
    if args.object_yaw_range is None:
        args.object_yaw_range = args.yaw_range
    if min(args.pose_range, args.yaw_range, args.object_pose_range, args.object_yaw_range,
           args.action_noise, args.settle_s) < 0 or args.interpolation_s <= 0:
        parser.error("ranges, noise, and --settle-s must be nonnegative; --interpolation-s must be positive")
    if args.nn_k < 1 or (args.fixed_source_index is not None and args.fixed_source_index < 0):
        parser.error("--nn-k must be positive and source index nonnegative")
    if args.follow_source_schedule and not args.trial_manifest:
        parser.error("--follow-source-schedule requires --trial-manifest")
    if (args.target_manifest is None) != (args.target_id is None):
        parser.error("--target-manifest and --target-id must be given together")
    if args.target_manifest is not None:
        if not args.no_randomize:
            parser.error("--target-manifest places the scene exactly; it requires --no-randomize")
        if not Path(args.target_manifest).is_file():
            parser.error(f"{args.target_manifest} does not exist")
    from .repairs import settings_from_args
    try:
        args.repairs = settings_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    if args.source_selection and args.episodes is not None:
        parser.error("Use either --source-selection or --episodes")
    if args.episodes and len(args.input) != 1:
        parser.error("--episodes selects from a single --input run")
    for run in [*args.input, *([args.target_input] if args.target_input else [])]:
        for name in ("omy_demos.hdf5", "env_cfg.json"):
            if not (Path(run) / name).is_file():
                parser.error(f"{run} has no {name}")


def reset_poses(env, env_ids, pose_ranges):
    """Reset each named asset uniformly over its own range and every other asset to its default."""
    from robolab.core.events.reset_pose import reset_pose_uniform

    for index, (name, pose_range) in enumerate(pose_ranges.items()):
        reset_pose_uniform(env, env_ids, pose_range, {}, name, reset_to_default_otherwise=index == 0)


def target_poses(manifest, target_id, required):
    """Start poses of one explicit target, by asset name, and its manifest row.

    A manifest is ``{"targets": [{"id": ..., "poses": {name: [x, y, z, qw, qx, qy, qz]}}]}``.
    Scene preparation re-seats heights on the table, so z need only be near it.
    """
    rows = [row for row in json.loads(Path(manifest).read_text())["targets"] if row.get("id") == target_id]
    if len(rows) != 1:
        raise ValueError(f"{manifest} has {len(rows)} targets with id {target_id!r}; expected one")
    poses = rows[0].get("poses") or {}
    missing = sorted(set(required) - set(poses))
    if missing:
        raise ValueError(f"Target {target_id!r} has no pose for {missing}")
    for name, pose in poses.items():
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-8:
            raise ValueError(f"Target {target_id!r} pose of {name} must be 7 finite values with a nonzero quaternion")
    return {name: [float(v) for v in pose] for name, pose in poses.items()}, rows[0]


def execute_trial(generator, subtask_count, boundaries, seed, step, *, forced=None, later=None, per_subtask=False,
                  timing=None, interpolate=None):
    """Generate and execute one trial's subtask segments.

    ``forced`` fixes the first segment's source (None: the generator selects).
    Without ``per_subtask`` that source is kept for every segment, as Isaac Lab
    does by default. With it, the stored choice is replaced before each later
    segment by ``later`` (None: the generator selects again); Isaac Lab then
    selects only because the config's per-subtask flag is also set.
    ``step(waypoint)`` executes one waypoint and returns whether the episode ended.
    ``timing(start, target)``, if given, sets each connection's interpolation
    steps from its 4x4 start and target poses; ``interpolate`` is Isaac Lab's
    pose interpolation (see ``repairs.retime``).

    Returns the executed segments (subtask, source, realized boundary), the last
    executed waypoint, the step count and whether the episode ended.
    """
    selected, previous = {EEF: forced}, {EEF: None}
    segments, waypoint, steps, ended = [], None, 0, False
    for subtask in range(subtask_count):
        if per_subtask and subtask > 0:
            selected[EEF] = later
        # The connection starts from the current hand pose, then from the previous segment's last target.
        start = (generator.env.get_robot_eef_pose(env_ids=[0], eef_name=EEF)[0] if previous[EEF] is None
                 else previous[EEF][-1].pose) if timing is not None else None
        with trajectory_rng(seed + subtask * 100000):
            previous[EEF] = generator.generate_trajectory(0, EEF, subtask, boundaries, {}, selected, previous)
        # Nearest-neighbor strategies can return scalar tensors rather than ints.
        source = int(selected[EEF])
        segment = {"subtask": subtask, "source": source,
                   "boundary": [int(v) for v in boundaries[EEF][source, subtask]]}
        if timing is not None:
            from .repairs import retime
            fixed = generator.env_cfg.subtask_configs[EEF][subtask].num_interpolation_steps
            target = previous[EEF][fixed].pose
            chosen = timing(start, target)
            previous[EEF] = retime(previous[EEF], start, fixed, chosen, interpolate)
            segment.update(interpolation_steps=int(chosen), connection_m=round(float(
                (target[:3, 3] - start[:3, 3]).norm()), 4))
        segments.append(segment)
        for waypoint in previous[EEF]:
            steps += 1
            if step(waypoint):
                ended = True
                break
        if ended:
            break
    return segments, waypoint, steps, ended


def trial_lineage(segments, sources):
    """Parent record of an attempt; the single-parent fields are None when parents differ."""
    parents = [segment["source"] for segment in segments]
    single = len(set(parents)) == 1
    return {"source": parents[0] if single else None,
            "source_reference": sources[parents[0]] if single else None,
            "sources_by_subtask": parents,
            "source_references_by_subtask": [sources[index] for index in parents],
            "segments": segments}


def teleop_args(task, recorded):
    """Teleoperation options that reproduce the recorded control and physics contract."""
    options = ["--task", task, "--record", "--no-realtime",
               "--control-hz", str(PHYSICS_HZ // recorded["decimation"]),
               "--stepping-profile", recorded.get("omy_stepping_profile", "ordinary"),
               "--recording-profile", recorded.get("omy_recording_profile", "full"),
               "--contact-sensors", recorded.get("omy_contact_sensor_profile", "full"),
               "--episode-length-s", str(recorded["episode_length_s"]),
               "--render-interval", str(recorded["sim"]["render_interval"])]
    cap = recorded["sim"]["physx"].get("max_position_iteration_count")
    if cap in (16, 32):
        options += ["--solver-position-iterations", str(cap)]
    return teleop_parser().parse_args(options)


def mimic_env_class():
    """Build the environment class; Isaac Lab is importable only after the app starts."""
    import isaaclab.utils.math as math_utils
    import torch
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.envs import ManagerBasedRLMimicEnv

    from .environment import OmyEnv

    class OmyMimicEnv(OmyEnv, ManagerBasedRLMimicEnv):
        def _mimic_setup(self):
            robot = self.scene["robot"]
            self._mimic_arm_ids = robot.find_joints(list(ARM_JOINTS), preserve_order=True)[0]
            free = [name for name in ARM_JOINTS if name != LOCKED_JOINT]
            self._mimic_ik_ids = robot.find_joints(free, preserve_order=True)[0]
            self._mimic_ik_columns = [ARM_JOINTS.index(name) for name in free]
            self._mimic_body = robot.find_bodies(EE_BODY)[0][0]
            self._mimic_ik = DifferentialIKController(
                DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
                num_envs=1, device=self.device,
            )
            self.mimic_max_joint_step = 0.0
            self._mimic_commands = []

        def _mimic_eef(self):
            if not hasattr(self, "_mimic_ik"):
                self._mimic_setup()
            robot = self.scene["robot"]
            pose = robot.data.body_link_pose_w[:, self._mimic_body]
            return math_utils.subtract_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w, pose[:, :3], pose[:, 3:7])

        def get_object_poses(self, env_ids=None):
            """Reference poses for generation; a symmetric object's heading becomes its bearing (see repairs)."""
            poses = super().get_object_poses(env_ids)
            for name in getattr(self, "mimic_symmetric", ()):
                rotation = poses[name][..., :3, :3]
                # The robot root is the unrotated environment origin, so bearings are taken about it.
                heading = getattr(self, "mimic_heading_rad", None)
                bearing = (torch.atan2(poses[name][..., 1, 3], poses[name][..., 0, 3]) if heading is None
                           else torch.full_like(poses[name][..., 0, 3], heading))
                yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0]) - bearing
                c, s = torch.cos(-yaw), torch.sin(-yaw)
                zero, one = torch.zeros_like(c), torch.ones_like(c)
                turn = torch.stack((torch.stack((c, -s, zero), -1), torch.stack((s, c, zero), -1),
                                    torch.stack((zero, zero, one), -1)), -2)
                poses[name] = poses[name].clone()
                poses[name][..., :3, :3] = turn @ rotation
            return poses

        def get_robot_eef_pose(self, eef_name, env_ids=None):
            pos, quat = self._mimic_eef()
            pose = math_utils.make_pose(pos, math_utils.matrix_from_quat(quat))
            return pose if env_ids is None else pose[env_ids]

        def fk(self, joint_position):
            """End-effector poses for full joint vectors; leaves the robot at the last one."""
            robot = self.scene["robot"]
            poses = []
            for row in torch.as_tensor(joint_position, dtype=torch.float32, device=self.device):
                # Writing joints invalidates link poses; reading them updates kinematics without physics.
                robot.write_joint_position_to_sim(row.unsqueeze(0))
                poses.append(self.get_robot_eef_pose(EEF)[0])
            return torch.stack(poses)

        def target_eef_pose_to_action(self, target_eef_pose_dict, gripper_action_dict,
                                      action_noise_dict=None, env_id=0):
            pos, rot = math_utils.unmake_pose(target_eef_pose_dict[EEF])
            noise = (action_noise_dict or {}).get(EEF)
            if noise:
                pos = pos + noise * torch.randn_like(pos)
            ee_pos, ee_quat = self._mimic_eef()
            robot = self.scene["robot"]
            # The root is the fixed, unrotated environment origin, so world and root axes coincide.
            jacobian = robot.root_physx_view.get_jacobians()[:, self._mimic_body - 1][:, :, self._mimic_ik_ids]
            current = robot.data.joint_pos[:, self._mimic_ik_ids]
            self._mimic_ik.set_command(torch.cat((pos, math_utils.quat_from_matrix(rot))).unsqueeze(0))
            delta = self._mimic_ik.compute(ee_pos, ee_quat, jacobian, current) - current
            # Teleoperation commands lead the arm by more than its velocity limit allows per step,
            # so this only bounds a near-singular solution.
            delta = delta / (delta.abs().max() / MAX_JOINT_STEP).clamp(min=1.0)
            self.mimic_max_joint_step = max(self.mimic_max_joint_step, delta.abs().max().item())
            arm = robot.data.default_joint_pos[env_id, self._mimic_arm_ids].clone()
            arm[self._mimic_ik_columns] = (current + delta)[0]
            self._mimic_commands.append((pos, math_utils.quat_from_matrix(rot), arm))
            return torch.cat((arm, gripper_action_dict[EEF].reshape(1).to(arm))).unsqueeze(0)

        def ik_errors(self):
            """Largest position and rotation error of the joint commands since the last call.

            Each command is compared with its target by forward kinematics, which moves the robot:
            call this only between attempts.
            """
            positions, quats, arms = (torch.stack(rows) for rows in zip(*self._mimic_commands))
            self._mimic_commands = []
            joints = self.scene["robot"].data.joint_pos[:1].repeat(len(arms), 1)
            joints[:, self._mimic_arm_ids] = arms
            reached, rotation = math_utils.unmake_pose(self.fk(joints))
            position_error, rotation_error = math_utils.compute_pose_error(
                reached, math_utils.quat_from_matrix(rotation), positions, quats, rot_error_type="axis_angle")
            return position_error.norm(dim=-1).max().item(), rotation_error.norm(dim=-1).max().item()

        def actions_to_gripper_actions(self, actions):
            return {EEF: actions[:, -1:]}

    return OmyMimicEnv


def mimic_cfg(cfg, object_name, container_name, args):
    """Attach the two pick-and-place segments Isaac Lab Mimic transforms."""
    from isaaclab.envs.mimic_env_cfg import DataGenConfig, MimicEnvCfg, SubTaskConfig

    # The generator requires a MimicEnvCfg instance; the task config is already built.
    cfg.__class__ = type(f"{type(cfg).__name__}Mimic", (type(cfg), MimicEnvCfg), {})
    cfg.datagen_config = DataGenConfig(
        name=args.task, seed=args.seed, generation_num_trials=args.num_demos,
        generation_keep_failed=args.keep_failed,
        # The per-arm flag stays False: setting it would also override fixed and scheduled sources.
        generation_select_src_per_subtask=bool(getattr(args, "select_source_per_subtask", False)),
        generation_transform_first_robot_pose=False, generation_interpolate_from_last_target_pose=True,
    )
    common = dict(selection_strategy=args.selection_strategy,
                  selection_strategy_kwargs=({"nn_k": min(args.nn_k, args.source_pool_size)}
                                             if args.selection_strategy != "random" else {}),
                  action_noise=args.action_noise, num_fixed_steps=0,
                  num_interpolation_steps=max(1, round(args.interpolation_s / (cfg.sim.dt * cfg.decimation))))
    cfg.subtask_configs = {EEF: [
        SubTaskConfig(object_ref=object_name, subtask_term_signal=GRASP_SIGNAL,
                      subtask_term_offset_range=(0, 3), **common),
        SubTaskConfig(object_ref=container_name, subtask_term_signal=None,
                      subtask_term_offset_range=(0, 0), **common),
    ]}
    cfg.task_constraint_configs = []


def pose_recorder_cfg():
    """Achieved pre-action poses for comparison of generated motion paths."""
    from isaaclab.managers import RecorderTermCfg
    from .recording import MimicPoseRecorder
    return RecorderTermCfg(class_type=MimicPoseRecorder)


def load_sources(env, pool, references, object_name, container_name, repairs=None, geometry=None):
    """Add the successful source episodes to the pool; return their provenance.

    Each reference is read by the reader of its kind (``mimic_sources.read_source``), so teleoperation recordings and
    other source kinds enter the pool through the same checks.
    """
    import asyncio

    import isaaclab.utils.math as math_utils
    import torch
    from isaaclab.utils.datasets import EpisodeData

    def poses(rows):
        rows = torch.as_tensor(rows, dtype=torch.float32, device=env.device)
        return math_utils.make_pose(rows[:, :3], math_utils.matrix_from_quat(rows[:, 3:7]))

    arm_ids = env.scene["robot"].find_joints(list(ARM_JOINTS), preserve_order=True)[0]
    locked = ARM_JOINTS.index(LOCKED_JOINT)
    home = env.scene["robot"].data.default_joint_pos[0, arm_ids[locked]].item()
    sources = []
    for reference in references:
        path, name = reference["file"], reference["episode"]
        source = read_source(reference, object_name, container_name)
        if not source.success:
            raise ValueError(f"Source changed or unsuccessful: {path}:{name}")
        if source.joint_names is not None and list(source.joint_names) != list(env.scene["robot"].joint_names):
            raise ValueError(f"{path}:{name} records joints {list(source.joint_names)}, "
                             f"not the generator robot's {list(env.scene['robot'].joint_names)}")
        arrays = {"actions": source.actions, "joint_position": source.joint_position,
                  "object_pose": dict(source.object_pose), "grasp": source.grasp}
        actions, joints = arrays["actions"], arrays["joint_position"]
        raw_object = arrays["object_pose"][object_name]
        if repairs is not None and repairs.symmetric_yaw:
            from .repairs import bearing_reference, heading_reference
            # With a fixed offset the source heading is removed; the target's becomes the offset.
            arrays["object_pose"][object_name] = (bearing_reference(raw_object) if repairs.object_yaw_offset_deg is None
                                                  else heading_reference(raw_object, 0.0))
        if not np.allclose(actions[:, locked], home, atol=1e-4):
            raise ValueError(f"{path}:{name} commands {LOCKED_JOINT}; six-axis IK cannot reproduce it")
        commanded = joints.copy()
        commanded[:, arm_ids] = actions[:, :len(ARM_JOINTS)]
        signal = torch.arange(len(actions), device=env.device) >= arrays["grasp"]
        eef = env.fk(joints)
        extra = {}
        if repairs is not None and repairs.boundaries:
            from .repairs import source_cuts
            try:
                # The pool's grasp segment ends one step after the signal rises.
                extra["cuts"] = source_cuts(actions, eef[:, :3, 3].cpu().numpy(), raw_object,
                                            arrays["object_pose"][container_name], arrays["grasp"] + 1,
                                            geometry, repairs)
            except ValueError as error:
                raise ValueError(f"{path}:{name}: {error}") from None
        episode = EpisodeData()
        episode.data = {
            "actions": torch.as_tensor(actions, device=env.device),
            "obs": {"datagen_info": {
                "eef_pose": {EEF: eef},
                "target_eef_pose": {EEF: env.fk(commanded)},
                "object_pose": {key: poses(value) for key, value in arrays["object_pose"].items()},
                "subtask_term_signals": {GRASP_SIGNAL: signal},
            }},
        }
        asyncio.run(pool.add_episode(episode))
        sources.append({**reference, "file": path, "episode": name, "steps": len(actions), "grasp": arrays["grasp"],
                        **extra})
    if not sources:
        raise ValueError("no successful source demonstrations were found")
    return sources


def run(args):
    """Generate demonstrations until enough succeed or the attempt budget is spent."""
    import h5py
    import torch
    from isaaclab.managers import DatasetExportMode
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab_mimic.datagen.data_generator import DataGenerator
    from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool

    import robolab.constants
    from robolab.core.environments.config import parse_env_cfg
    from robolab.core.environments.factory import get_envs
    from robolab.core.environments.runtime import create_env
    from robolab.core.events.reset_pose import reset_pose_uniform
    from robolab.core.events.subtask_recorder import SubtaskCompletionRecorderTerm
    from robolab.core.events.utils import merge_events_cfg

    from .config import ENV_POSTFIX, configure_performance, register_omy_envs, validate_environment
    from .runner import close_recording, prepare_output

    @torch.inference_mode()
    def generate():
        references = source_references(args)
        args.source_pool_size = len(references)
        if args.fixed_source_index is not None and args.fixed_source_index >= len(references):
            raise ValueError("Fixed source index exceeds pool size")
        trials = json.loads(Path(args.trial_manifest).read_text())["trials"] if args.trial_manifest else None
        if trials is not None and len(trials) != args.max_attempts:
            raise ValueError("Trial manifest must contain exactly --max-attempts trials")
        first = Path(args.target_input or args.input[0]).resolve()
        recorded = json.loads((first / "env_cfg.json").read_text())
        with h5py.File(first / "omy_demos.hdf5", "r") as file:
            source_attrs = {key: file["data"].attrs[key]
                            for key in ("robolab_omy_teleop", "robolab_omy_collection") if key in file["data"].attrs}
        source_meta = json.loads(source_attrs["robolab_omy_teleop"])
        args.task = source_meta["task"]
        from .dataset_paths import named_output

        output = prepare_output(args.output_dir or
                                Path("output/omy_mimicgen") / named_output(
                                    first, args.task, datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
        robolab.constants.set_output_dir(str(output))
        # Same historical flag as collection: it enables the policy-observation recorder terms.
        robolab.constants.RECORD_IMAGE_DATA = True
        from robolab.constants import TASK_DIR
        from robolab.core.task.task_index import resolve_selected_task

        # Verify the recorded task source once: None, or the allowed
        # added-speed_source difference from the recorded task file.
        resolution = resolve_selected_task(args.task, source_meta.get("task_dirs"), task_root=TASK_DIR)
        task_source_difference = resolution.verify_expected_source(source_meta.get("task_source"))
        register_omy_envs(args.task, images=False, stream=False, resolution=resolution,
                          task_dirs=source_meta.get("task_dirs"))
        names = [name for name in get_envs(task=args.task) if name.endswith(ENV_POSTFIX)]
        if len(names) != 1:
            raise ValueError(f"Expected one OMY task environment, found {names}")
        cfg = parse_env_cfg(names[0], device=args.device, num_envs=1, seed=args.seed, use_fabric=True)
        cfg.episode_length_s = recorded["episode_length_s"]
        options = teleop_args(args.task, recorded)
        for key, value in vars(args).items():
            if key.startswith("scene_settle_"):
                setattr(options, key, value)
        configure_performance(cfg, options)
        cfg.recorders.mimic_poses = pose_recorder_cfg()
        # Collection prepares supported starts rather than the authored airborne poses.
        # Center resets on an actual successful source's initial arrangement.
        with h5py.File(first / "omy_demos.hdf5", "r") as file:
            eligible = [group for name, group in file["data"].items()
                        if group.attrs.get("success", False)
                        and (args.target_episode is None or int(name.rsplit("_", 1)[1]) == args.target_episode)
                        and (args.target_input is not None or args.source_selection is not None
                             or not args.episodes or int(name.rsplit("_", 1)[1]) in args.episodes)]
            if not eligible:
                raise ValueError("The first input has no selected successful source")
            args.target_episode = int(eligible[0].name.rsplit("_", 1)[1])
            for name, state in eligible[0]["initial_state/rigid_object"].items():
                pose = state["root_pose"][0]
                asset = getattr(cfg.scene, name)
                asset.init_state.pos = tuple(float(v) for v in pose[:3])
                asset.init_state.rot = tuple(float(v) for v in pose[3:7])
        success = cfg.terminations.success.params
        object_name = args.object or success["object"]
        container_name = args.container or success["container"]
        target = None
        if args.target_manifest is not None:
            poses, row = target_poses(args.target_manifest, args.target_id, (object_name, container_name))
            unknown = sorted(name for name in poses if getattr(cfg.scene, name, None) is None)
            if unknown:
                raise ValueError(f"Target {args.target_id!r} names assets absent from {args.task}: {unknown}")
            for name, pose in poses.items():
                asset = getattr(cfg.scene, name)
                asset.init_state.pos, asset.init_state.rot = tuple(pose[:3]), tuple(pose[3:])
            target = {"id": args.target_id, "manifest": str(Path(args.target_manifest).resolve()),
                      "manifest_sha256": checksum(args.target_manifest), "poses": poses,
                      **{k: v for k, v in row.items() if k not in ("id", "poses")}}
        mimic_cfg(cfg, object_name, container_name, args)
        pose_range, object_pose_range = (
            {"x": (-xy, xy), "y": (-xy, xy), "yaw": (-yaw, yaw)}
            for xy, yaw in ((args.pose_range, args.yaw_range), (args.object_pose_range, args.object_yaw_range)))
        if not args.no_randomize:
            # As in the randomized tasks, this replaces the default reset: merged alongside it,
            # the default reset would run second and restore the recorded arrangement.
            term = EventTerm(func=reset_pose_uniform, mode="reset",
                             params={"pose_range": pose_range, "velocity_range": {},
                                     "asset_cfg": [object_name, container_name]})
            if object_pose_range != pose_range:
                term = EventTerm(func=reset_poses, mode="reset", params={
                    "pose_ranges": {container_name: pose_range, object_name: object_pose_range}})
            cfg.events = merge_events_cfg(None, {"randomize_init_pose": term})
        env = None
        results = []
        completion_status = None
        try:
            env, cfg = create_env(cfg, policy="omy_joint_position", env_class=mimic_env_class())
            validate_environment(env)
            recorder = env.recorder_manager
            recorder.set_buffered_recording()
            recorder.cfg.dataset_export_mode = (
                DatasetExportMode.EXPORT_ALL if args.keep_failed else DatasetExportMode.EXPORT_SUCCEEDED_ONLY)
            recorder.set_hdf5_compression(None)
            recorder.set_flush_interval(0)
            recorder.set_flush_memory_cleanup(False)
            recorder.set_hdf5_file("omy_demos")
            env.reset(seed=args.seed)
            pool = DataGenInfoPool(env, cfg, env.device)
            repairs, geometry = args.repairs, None
            if repairs.boundaries:
                from .repairs import placement_geometry
                from .target_sampler import scene_geometry
                bounds = scene_geometry(cfg.scene.scene.spawn.usd_path, (object_name, container_name))
                geometry = placement_geometry(bounds[object_name], bounds[container_name])
            sources = load_sources(env, pool, references, object_name, container_name, repairs, geometry)
            if repairs.symmetric_yaw:
                env.mimic_symmetric = (object_name,)
                if repairs.object_yaw_offset_deg is not None:
                    env.mimic_heading_rad = math.radians(repairs.object_yaw_offset_deg)
            robot = env.scene["robot"]
            arm_ids = robot.find_joints(list(ARM_JOINTS), preserve_order=True)[0]
            recorder.set_dataset_attrs({**source_attrs, "robolab_omy_mimicgen": json.dumps({
                # 2: per-subtask parents in attempt and episode records, optional explicit target.
                "schema_version": 2, "sources": sources, "seed": args.seed,
                "select_source_per_subtask": args.select_source_per_subtask, "target": target,
                "task_source": resolution.metadata(), "task_source_difference": task_source_difference,
                "selection_strategy": args.selection_strategy, "selection_seed": args.selection_seed,
                "nn_k": min(args.nn_k, len(sources)), "fixed_source_index": args.fixed_source_index,
                "follow_source_schedule": args.follow_source_schedule,
                "target_input": str(first.resolve()), "target_episode": args.target_episode,
                "target_source_sha256": checksum(first / "omy_demos.hdf5"),
                "target_config_sha256": checksum(first / "env_cfg.json"),
                "joint_position_limits": robot.data.soft_joint_pos_limits[0, arm_ids].tolist(),
                "object": object_name, "container": container_name,
                "pose_range": None if args.no_randomize else pose_range,
                "object_pose_range": None if args.no_randomize else object_pose_range,
                "action_noise": args.action_noise, "interpolation_s": args.interpolation_s,
                "synthetic": True, "task_dirs": source_meta.get("task_dirs"),
                "preparation": cfg.omy_preparation,
                **({"repairs": {**repairs.metadata(), "geometry": geometry}} if repairs.active else {}),
            })})
            generator = DataGenerator(env=env, src_demo_datagen_info_pool=pool)
            timing = interpolate = None
            if repairs.connection_speed_m_s is not None:
                import isaaclab.utils.math as math_utils

                from .repairs import connection_steps
                minimum = cfg.subtask_configs[EEF][0].num_interpolation_steps
                hz = 1.0 / env.step_dt

                def timing(start, target):
                    return connection_steps(start.cpu().numpy(), target.cpu().numpy(), minimum,
                                            repairs.connection_speed_m_s, repairs.connection_rate_rad_s, hz)

                def interpolate(start, target, steps):
                    return math_utils.interpolate_poses(pose_1=start, pose_2=target, num_steps=steps)[0]
            print(f"Task: {cfg.instruction}\nOutput: {output}\nSource demonstrations: {len(sources)}", flush=True)
            successes = 0
            settle_steps = round(args.settle_s / env.step_dt)
            tracker = recorder.get_term(SubtaskCompletionRecorderTerm)
            while successes < args.num_demos and len(results) < args.max_attempts:
                trial_index = len(results)
                trial = trials[trial_index] if trials is not None else {"id": trial_index, "seed": args.seed + trial_index}
                seed = int(trial["seed"])
                env.reset(seed=seed)  # also exports or discards an unfinished attempt
                from .preparation_runtime import ScenePreparation
                preparation = ScenePreparation(env, options)
                while True:
                    prepared = preparation.advance()
                    if prepared["status"] != "settling":
                        break
                initial_poses = {name: asset.data.root_pose_w[0].cpu().tolist()
                                 for name, asset in env.scene.rigid_objects.items()}
                trial_info = {"trial_id": trial["id"], "selection_seed": args.selection_seed + trial_index,
                              "prepared_initial_poses": initial_poses}
                if prepared["status"] != "ready":
                    results.append({**trial_info, "seed": seed, "success": False, "steps": 0, "source": None,
                                    "sources_by_subtask": [], "demo": None, "preparation": prepared})
                    print(f"Attempt {len(results)}: setup rejected: {prepared['reasons']}", flush=True)
                    if args.no_randomize and any("Cannot prepare layout:" in reason for reason in prepared["reasons"]):
                        break
                    continue
                index = len(results) if args.keep_failed else successes
                recorder.set_episode_index(index, env_ids=[0])
                recorder.set_episode_seed(seed, env_ids=[0])
                recorder.set_episode_preparation(prepared, env_ids=[0])
                with trajectory_rng(args.selection_seed + trial_index):
                    boundaries = generator.randomize_subtask_boundaries()
                if repairs.boundaries:
                    from .repairs import edit_boundaries
                    boundaries[EEF] = edit_boundaries(boundaries[EEF], pool.subtask_boundaries[EEF],
                                                      [source["cuts"] for source in sources], repairs)
                fixed_source = int(trial["source_index"]) if args.follow_source_schedule else args.fixed_source_index
                placement_source = trial.get("placement_source_index")
                if placement_source is not None:
                    if not args.select_source_per_subtask:
                        raise ValueError("Trial placement_source_index requires --select-source-per-subtask")
                    placement_source = int(placement_source)
                if any(index is not None and not 0 <= index < len(sources)
                       for index in (fixed_source, placement_source)):
                    raise ValueError("Scheduled source index exceeds the pool")

                def execute(waypoint):
                    env.step(env.target_eef_pose_to_action(
                        {EEF: waypoint.pose}, {EEF: waypoint.gripper_action}, {EEF: waypoint.noise}))
                    return bool(env.all_terminated)

                segments, waypoint, steps, _ = execute_trial(
                    generator, len(cfg.subtask_configs[EEF]), boundaries, args.selection_seed + trial_index, execute,
                    forced=fixed_source, later=placement_source, per_subtask=args.select_source_per_subtask,
                    timing=timing, interpolate=interpolate)
                lineage = trial_lineage(segments, sources)
                # Recordings stop at success, leaving the release no margin; hold the final target.
                for _ in range(settle_steps):
                    if env.all_terminated:
                        break
                    env.step(env.target_eef_pose_to_action(
                        {EEF: waypoint.pose}, {EEF: waypoint.gripper_action}, {EEF: None}))
                    steps += 1
                succeeded = bool(env.all_terminated and env.get_env_results()[0]["success"])
                successes += succeeded
                # The attempt is over, so forward kinematics may move the robot. See quality.py for the consumers.
                position_error, rotation_error = env.ik_errors()
                results.append({**trial_info, **lineage,
                                "seed": seed, "success": succeeded, "steps": steps,
                                "preparation": prepared,
                                "demo": f"demo_{index}" if succeeded or args.keep_failed else None,
                                "ik_position_error_m": position_error, "ik_orientation_error_rad": rotation_error,
                                "events": None if tracker is None else [
                                    {key: event[key] for key in ("step", "name", "info")}
                                    for event in tracker.get_events(0)]})
                print(f"Attempt {len(results)}: {'success' if succeeded else 'failure'} after {steps} steps "
                      f"(source {'/'.join(map(str, lineage['sources_by_subtask']))}); "
                      f"{successes}/{args.num_demos} generated.", flush=True)
                # Preserve diagnostics after every completed attempt, including interrupted runs.
                (output / "mimicgen_summary.json").write_text(json.dumps({
                    "successes": successes, "attempts": results,
                    "max_joint_step_rad": env.mimic_max_joint_step,
                }, indent=2) + "\n")
            env.reset(seed=args.seed)
            completion_status = 0 if successes >= args.num_demos else 1
            return completion_status
        finally:
            if env is not None:
                (output / "mimicgen_summary.json").write_text(json.dumps({
                    "successes": sum(result["success"] for result in results), "attempts": results,
                    "max_joint_step_rad": getattr(env, "mimic_max_joint_step", None),
                }, indent=2) + "\n")
                close_recording(env)
                recording = output / "omy_demos.hdf5"
                if recording.exists():
                    with h5py.File(recording, "r+") as stream:
                        for attempt in results:
                            name = attempt.get("demo")
                            if name is not None and name in stream["data"]:
                                stream["data"][name].attrs["mimicgen_source"] = json.dumps({
                                    "source_index": attempt["source"], "source": attempt.get("source_reference"),
                                    "parents": attempt["sources_by_subtask"],
                                    "parent_sources": attempt["source_references_by_subtask"],
                                    "segments": attempt["segments"],
                                    "strategy": args.selection_strategy, "selection_seed": attempt["selection_seed"],
                                    "trial_id": attempt["trial_id"], "target_input": str(first.resolve()),
                                    "target_episode": args.target_episode,
                                    "target_id": None if target is None else target["id"]})
                (output / "generation_finalized.json").write_text(json.dumps({
                    "recording_closed": True, "attempts": len(results),
                    "successes": sum(result["success"] for result in results),
                    "requested_successes": args.num_demos,
                    "generation_exit_status": completion_status,
                }, indent=2) + "\n")
                print("MimicGen recording finalized; closing simulation environment.", flush=True)
                env.close()
                print("MimicGen environment closed.", flush=True)

    return generate()
