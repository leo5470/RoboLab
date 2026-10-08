"""MimicGen expansion of OMY teleoperation demonstrations.

Generation runs in the teleoperation environment itself: Isaac Lab Mimic plans
end-effector waypoints, and the environment converts each one to the same
absolute joint command the L100 sends. Generated episodes therefore share the
teleoperation HDF5 contract. See docs/omy_mimicgen.md.
"""

import argparse
import json
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


def build_parser():
    parser = argparse.ArgumentParser(description="Expand OMY teleoperation demonstrations with MimicGen.")
    parser.add_argument("--input", nargs="+", required=True,
                        help="Teleoperation run directories (omy_demos.hdf5 with env_cfg.json).")
    parser.add_argument("--episodes", type=int, nargs="+",
                        help="Source episodes to use, as demo_<N> indices of a single --input; default: all successful.")
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
    if args.episodes and len(args.input) != 1:
        parser.error("--episodes selects from a single --input run")
    for run in args.input:
        for name in ("omy_demos.hdf5", "env_cfg.json"):
            if not (Path(run) / name).is_file():
                parser.error(f"{run} has no {name}")


def reset_poses(env, env_ids, pose_ranges):
    """Reset each named asset uniformly over its own range and every other asset to its default."""
    from robolab.core.events.reset_pose import reset_pose_uniform

    for index, (name, pose_range) in enumerate(pose_ranges.items()):
        reset_pose_uniform(env, env_ids, pose_range, {}, name, reset_to_default_otherwise=index == 0)


def grasp_index(gripper_command, object_z, lift=0.02):
    """First step at which the closed gripper has raised the object by ``lift`` meters."""
    held = (np.asarray(gripper_command) < 0) & (np.asarray(object_z) - object_z[0] > lift)
    if not held.any():
        raise ValueError("the object is never lifted with the gripper closed")
    return int(np.argmax(held))


def pre_action(initial, states):
    """Rows seen before each action: the initial state, then post-step state ``t - 1``."""
    return np.concatenate((np.asarray(initial)[:1], np.asarray(states)[:-1]))


def source_arrays(episode, object_name, container_name):
    """Pre-action joints and reference poses of one recorded episode, as NumPy arrays."""
    actions = np.asarray(episode["actions"])
    initial, states = episode["initial_state"], episode["states"]
    if len(states["articulation"]["robot"]["joint_position"]) != len(actions):
        raise ValueError("recorded actions and states are not aligned")
    poses = {name: pre_action(initial["rigid_object"][name]["root_pose"], states["rigid_object"][name]["root_pose"])
             for name in (object_name, container_name)}
    return {
        "actions": actions,
        "joint_position": pre_action(initial["articulation"]["robot"]["joint_position"],
                                     states["articulation"]["robot"]["joint_position"]),
        "object_pose": poses,
        "grasp": grasp_index(actions[:, -1], poses[object_name][:, 2]),
    }


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

        def _mimic_eef(self):
            if not hasattr(self, "_mimic_ik"):
                self._mimic_setup()
            robot = self.scene["robot"]
            pose = robot.data.body_link_pose_w[:, self._mimic_body]
            return math_utils.subtract_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w, pose[:, :3], pose[:, 3:7])

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
            return torch.cat((arm, gripper_action_dict[EEF].reshape(1).to(arm))).unsqueeze(0)

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
        generation_keep_failed=args.keep_failed, generation_select_src_per_subtask=False,
        generation_transform_first_robot_pose=False, generation_interpolate_from_last_target_pose=True,
    )
    common = dict(selection_strategy="random", action_noise=args.action_noise, num_fixed_steps=0,
                  num_interpolation_steps=max(1, round(args.interpolation_s / (cfg.sim.dt * cfg.decimation))))
    cfg.subtask_configs = {EEF: [
        SubTaskConfig(object_ref=object_name, subtask_term_signal=GRASP_SIGNAL,
                      subtask_term_offset_range=(0, 3), **common),
        SubTaskConfig(object_ref=container_name, subtask_term_signal=None,
                      subtask_term_offset_range=(0, 0), **common),
    ]}
    cfg.task_constraint_configs = []


def load_sources(env, pool, runs, object_name, container_name, episodes=None):
    """Add the successful recorded episodes, or only ``episodes``, to the pool; return their provenance."""
    import asyncio

    import h5py
    import isaaclab.utils.math as math_utils
    import torch
    from isaaclab.utils.datasets import EpisodeData

    from robolab.core.replay.materialize import episode_names, read_group_tree

    def poses(rows):
        rows = torch.as_tensor(rows, dtype=torch.float32, device=env.device)
        return math_utils.make_pose(rows[:, :3], math_utils.matrix_from_quat(rows[:, 3:7]))

    arm_ids = env.scene["robot"].find_joints(list(ARM_JOINTS), preserve_order=True)[0]
    locked = ARM_JOINTS.index(LOCKED_JOINT)
    home = env.scene["robot"].data.default_joint_pos[0, arm_ids[locked]].item()
    wanted = None if episodes is None else {f"demo_{index}" for index in episodes}
    sources = []
    for run in runs:
        path = str(Path(run) / "omy_demos.hdf5")
        for name in episode_names(path):
            if wanted is not None and name not in wanted:
                continue
            with h5py.File(path, "r") as file:
                group = file["data"][name]
                if not group.attrs.get("success", False):
                    continue
                tree = read_group_tree(group)
            arrays = source_arrays(tree, object_name, container_name)
            actions, joints = arrays["actions"], arrays["joint_position"]
            if not np.allclose(actions[:, locked], home, atol=1e-4):
                raise ValueError(f"{path}:{name} commands {LOCKED_JOINT}; six-axis IK cannot reproduce it")
            commanded = joints.copy()
            commanded[:, arm_ids] = actions[:, :len(ARM_JOINTS)]
            signal = torch.arange(len(actions), device=env.device) >= arrays["grasp"]
            episode = EpisodeData()
            episode.data = {
                "actions": torch.as_tensor(actions, device=env.device),
                "obs": {"datagen_info": {
                    "eef_pose": {EEF: env.fk(joints)},
                    "target_eef_pose": {EEF: env.fk(commanded)},
                    "object_pose": {key: poses(value) for key, value in arrays["object_pose"].items()},
                    "subtask_term_signals": {GRASP_SIGNAL: signal},
                }},
            }
            asyncio.run(pool.add_episode(episode))
            sources.append({"file": path, "episode": name, "steps": len(actions), "grasp": arrays["grasp"]})
    if wanted is not None and (missing := wanted - {source["episode"] for source in sources}):
        raise ValueError(f"--episodes selects missing or unsuccessful episodes: {sorted(missing)}")
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
    from robolab.core.events.utils import merge_events_cfg

    from .config import ENV_POSTFIX, configure_performance, register_omy_envs, validate_environment
    from .runner import close_recording, prepare_output

    @torch.inference_mode()
    def generate():
        first = Path(args.input[0])
        recorded = json.loads((first / "env_cfg.json").read_text())
        with h5py.File(first / "omy_demos.hdf5", "r") as file:
            source_attrs = {key: file["data"].attrs[key]
                            for key in ("robolab_omy_teleop", "robolab_omy_collection") if key in file["data"].attrs}
        args.task = json.loads(source_attrs["robolab_omy_teleop"])["task"]
        output = prepare_output(args.output_dir or
                                Path("output/omy_mimicgen") / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        robolab.constants.set_output_dir(str(output))
        # Same historical flag as collection: it enables the policy-observation recorder terms.
        robolab.constants.RECORD_IMAGE_DATA = True
        register_omy_envs(args.task, images=False, stream=False)
        names = [name for name in get_envs(task=args.task) if name.endswith(ENV_POSTFIX)]
        if len(names) != 1:
            raise ValueError(f"Expected one OMY task environment, found {names}")
        cfg = parse_env_cfg(names[0], device=args.device, num_envs=1, seed=args.seed, use_fabric=True)
        cfg.episode_length_s = recorded["episode_length_s"]
        configure_performance(cfg, teleop_args(args.task, recorded))
        success = cfg.terminations.success.params
        object_name = args.object or success["object"]
        container_name = args.container or success["container"]
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
            sources = load_sources(env, pool, args.input, object_name, container_name, args.episodes)
            recorder.set_dataset_attrs({**source_attrs, "robolab_omy_mimicgen": json.dumps({
                "schema_version": 1, "sources": sources, "seed": args.seed,
                "object": object_name, "container": container_name,
                "pose_range": None if args.no_randomize else pose_range,
                "object_pose_range": None if args.no_randomize else object_pose_range,
                "action_noise": args.action_noise, "interpolation_s": args.interpolation_s,
            })})
            generator = DataGenerator(env=env, src_demo_datagen_info_pool=pool)
            print(f"Task: {cfg.instruction}\nOutput: {output}\nSource demonstrations: {len(sources)}", flush=True)
            successes = 0
            settle_steps = round(args.settle_s / env.step_dt)
            while successes < args.num_demos and len(results) < args.max_attempts:
                seed = args.seed + len(results)
                env.reset(seed=seed)  # also exports or discards an unfinished attempt
                recorder.set_episode_index(len(results) if args.keep_failed else successes, env_ids=[0])
                recorder.set_episode_seed(seed, env_ids=[0])
                boundaries = generator.randomize_subtask_boundaries()
                selected, previous = {EEF: None}, {EEF: None}
                steps = 0
                for subtask in range(len(cfg.subtask_configs[EEF])):
                    previous[EEF] = generator.generate_trajectory(0, EEF, subtask, boundaries, {}, selected, previous)
                    for waypoint in previous[EEF]:
                        env.step(env.target_eef_pose_to_action(
                            {EEF: waypoint.pose}, {EEF: waypoint.gripper_action}, {EEF: waypoint.noise}))
                        steps += 1
                        if env.all_terminated:
                            break
                    if env.all_terminated:
                        break
                # Recordings stop at success, leaving the release no margin; hold the final target.
                for _ in range(settle_steps):
                    if env.all_terminated:
                        break
                    env.step(env.target_eef_pose_to_action(
                        {EEF: waypoint.pose}, {EEF: waypoint.gripper_action}, {EEF: None}))
                    steps += 1
                succeeded = bool(env.all_terminated and env.get_env_results()[0]["success"])
                successes += succeeded
                results.append({"seed": seed, "success": succeeded, "steps": steps, "source": selected[EEF]})
                print(f"Attempt {len(results)}: {'success' if succeeded else 'failure'} after {steps} steps "
                      f"(source {selected[EEF]}); {successes}/{args.num_demos} generated.", flush=True)
            env.reset(seed=args.seed)
            return 0 if successes >= args.num_demos else 1
        finally:
            if env is not None:
                (output / "mimicgen_summary.json").write_text(json.dumps({
                    "successes": sum(result["success"] for result in results), "attempts": results,
                    "max_joint_step_rad": getattr(env, "mimic_max_joint_step", None),
                }, indent=2) + "\n")
                close_recording(env)
                env.close()

    return generate()
