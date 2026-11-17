"""Scripted pick-and-place in the OMY teleoperation environment, recorded like an OMY collection.

``collect.Session`` runs the scripted policy in the stock DROID registration and writes PPS rows itself. ``OmySession``
runs the same planner and controller in the environment ``robolab.core.omy_teleop`` collects in: the OMY env built
from the task with the options OMY collection sessions use (``omy_teleop.collection.build_parser`` defaults: 15 Hz,
held-target stepping, compact state-only recording, required contact sensors, ...), the unrecorded scene preparation
that starts every attempt from objects resting on their support, and the OMY recorder. Its output is an OMY recording
(``<part>/omy_demos.hdf5`` with ``env_cfg.json``), so it can be a MimicGen target and source like a teleoperation
recording and goes through the same postprocessing; only ``robolab_omy_teleop.operator`` says who commanded it.

Differences from ``collect.Session``: no settle rows (preparation replaces them), the gripper is commanded +1 open /
-1 closed, nothing is rendered (OMY collection records state only; images are rendered later), and every attempt,
failed ones included, is exported, as OMY collection keeps failures.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from robolab.core.scripted_demos import collect, select, sim
from robolab.core.scripted_demos.grasps import library as grasp_library
from robolab.core.scripted_demos.grasps import objects
from robolab.core.scripted_demos.kinematics import Chain

OPERATOR_SCHEMA = 1


class PreparationFailed(RuntimeError):
    def __init__(self, report: dict):
        super().__init__("; ".join(report.get("reasons", [])) or report.get("status", "failed"))
        self.report = report


def collection_options(task: str, task_dirs=("single_pnp",), extra=()):
    """The options of an OMY collection session for ``task`` (its parser's defaults), without the XR checks.
    ``task_dirs``: where the task file is, as OMY sessions record it (``single_pnp`` for the Simple-PnP layouts)."""
    from robolab.core.omy_teleop.collection import build_parser
    from robolab.core.task.task_index import resolve_selected_task

    args = build_parser().parse_args(["--task", task, "--task-dirs", *task_dirs, *extra])
    args.task_resolution = resolve_selected_task(args.task, args.task_dirs)
    args.task = args.task_resolution.class_name
    args.task_dirs = list(args.task_resolution.task_dirs)
    return args


class OmySession(collect.Session):
    """One layout task in the OMY env; ``run`` is ``collect.Session.run`` with this env's reset, step and recording."""

    def __init__(self, obj: objects.BatchObject, layout: str, *, output: pathlib.Path,
                 library_path: pathlib.Path | None = None, locked_joints: tuple[str, ...] = (), seed: int = 0):
        sim.launch_app()
        import robolab.constants
        from isaaclab.managers import DatasetExportMode
        from robolab.core.environments.config import parse_env_cfg
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env
        from robolab.core.omy_teleop.config import (ENV_POSTFIX, configure_performance, register_omy_envs,
                                                    validate_environment)
        from robolab.core.omy_teleop.environment import OmyEnv
        from robolab.core.omy_teleop.options import ARM_JOINTS
        from robolab.core.omy_teleop.runner import performance_settings
        from robolab.core.world.world_state import get_world

        unknown = sorted(set(locked_joints) - set(sim.ARM_JOINTS))
        if unknown:
            raise ValueError(f"cannot lock {unknown}: not arm joints {sim.ARM_JOINTS}")
        self.obj, self.layout = obj, layout
        self.task = obj.task(layout)
        self.render = False
        self.locked_joints = tuple(locked_joints)
        self.output = pathlib.Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.options = collection_options(self.task)
        robolab.constants.set_output_dir(str(self.output))
        robolab.constants.RECORD_IMAGE_DATA = True  # enables the policy-observation recorder terms, as collection does
        register_omy_envs(self.task, images=False, stream=False, optimized_observations=True,
                          task_dirs=self.options.task_dirs, resolution=self.options.task_resolution)
        names = [name for name in get_envs(task=self.task) if name.endswith(ENV_POSTFIX)]
        if len(names) != 1:
            raise ValueError(f"Expected one OMY task environment, found {names}")
        cfg = parse_env_cfg(names[0], device="cuda:0", num_envs=1, seed=seed, use_fabric=True)
        cfg.episode_length_s = self.options.episode_length_s
        configure_performance(cfg, self.options)
        self.env, self.env_cfg = create_env(cfg, policy="omy_joint_position", env_class=OmyEnv)
        validate_environment(self.env)
        self.world = get_world(self.env)
        root = objects.grasp_data_root()
        self.chain = Chain.load(root / "robot_chain.npz").with_locked(sim.ARM_JOINTS.index(j) for j in locked_joints)
        self.ik = sim.ArmIK(self.env, self.chain)
        self.library_path = library_path or root / "library" / f"{obj.key}.npz"
        self.library = grasp_library.load(self.library_path)
        self.client = None
        self.instruction = self.env_cfg.instruction
        self.obs = None
        self.attempts = 0
        self.preparation = None

        recorder = self.env.recorder_manager
        recorder.set_buffered_recording()
        recorder.cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL  # failed attempts are kept, as in collection
        recorder.set_hdf5_compression(None)
        recorder.set_flush_interval(0)
        recorder.set_flush_memory_cleanup(False)
        from robolab.core.scripted_demos.grasps.objects import GRASPDATAGEN_COMMIT
        from robolab.core.scripted_demos.records import file_sha256, git_revision

        operator = {"schema_version": OPERATOR_SCHEMA, "kind": "scripted",
                    "planner": "robolab.core.scripted_demos (GraspDataGen grasps, MolmoSpaces-style selection, "
                               "DLS IK one step per control step)",
                    "locked_joints": list(self.locked_joints), "library": str(self.library_path),
                    "library_sha256": file_sha256(self.library_path), "graspdatagen_commit": GRASPDATAGEN_COMMIT,
                    "robolab_revision": git_revision(objects.ROBOLAB_DIR)}
        metadata = {
            "schema_version": 1, "task": self.task, "task_dirs": list(self.options.task_dirs),
            "task_source": self.options.task_resolution.metadata(),
            "action_type": "absolute_joint_position", "action_names": [*ARM_JOINTS, "gripper"], "joint_units": "rad",
            "gripper_encoding": "+1=open,-1=closed",
            "gripper_observation": {"joint": "finger_joint", "units": "normalized_closure", "range": [0, 1],
                                    "raw_state_units": "rad"},
            "control_hz": 1.0 / self.env.step_dt, "images": "none", "save_failures": True,
            "calibration": {"applies": False, "reason": "commands come from the scripted planner, not the L100 leader"},
            "reset_behavior": "environment_home_open_gripper_then_scene_preparation",
            "operator": operator, "performance": performance_settings(self.env, self.options),
        }
        collection = {"schema_version": 1, "start": "scripted_after_scene_preparation", "images": "deferred",
                      "recording": "state_only", "max_attempt_steps": self.options.max_attempt_steps,
                      "max_attempt_seconds": self.options.max_attempt_seconds,
                      "preparation": self.env_cfg.omy_preparation, "operator": "scripted",
                      "object": obj.name, "catalog_index": obj.index, "layout": {"task": self.task}}
        recorder.set_hdf5_file("omy_demos", attrs={"robolab_omy_teleop": json.dumps(metadata),
                                                   "robolab_omy_collection": json.dumps(collection)})

    def close(self) -> None:
        from robolab.core.omy_teleop.runner import close_recording

        try:
            close_recording(self.env)
        finally:
            self.env.close()

    def reset(self, seed: int):
        """A fresh scene as OMY collection starts one: reset, then the unrecorded preparation until objects rest."""
        import torch

        from robolab.core.omy_teleop.preparation_runtime import ScenePreparation

        if self.attempts:
            self.env.reset_eval_state()
        self.env.reset(seed=int(seed))
        preparation = ScenePreparation(self.env, self.options)
        while (report := preparation.advance())["status"] == "settling":
            pass
        self.preparation = report
        if report["status"] != "ready":
            self.env.recorder_manager.clear()
            raise PreparationFailed(report)
        recorder = self.env.recorder_manager
        recorder.set_episode_index(self.attempts, env_ids=[0])
        recorder.set_episode_seed(int(seed), env_ids=[0])
        recorder.set_episode_preparation(report, env_ids=[0])
        self.attempts += 1
        self.obs = self.env.obs_buf
        torch.cuda.synchronize()
        self.ik.reset_command()
        return self.obs

    def settle(self, buffer, writer, seed, **_) -> int:
        return 0  # the preparation has settled the scene, unrecorded

    def _record(self, *args, **kwargs) -> None:
        return  # the OMY recorder records every step

    def step(self, action: np.ndarray):
        """Scripted wire action (gripper 0 open / 1 closed) → OMY command (+1 open / -1 closed)."""
        command = np.asarray(action, dtype=np.float32).copy()
        command[7] = 1.0 - 2.0 * command[7]
        return super().step(command)

    def run(self, seed: int, **kwargs):
        """``collect.Session.run``; the attempt is exported when the env terminates (success or time-out), or as a
        failure when the scripted executor stops first."""
        from robolab.core.omy_teleop.runner import finish_interrupted

        kwargs.pop("record", None), kwargs.pop("video", None)
        try:
            result, _ = super().run(seed, record=False, video=None, **kwargs)
        except PreparationFailed as failed:
            return collect.Result(self.task, self.layout, seed, False, "preparation_failed", 0, None,
                                  {"preparation": failed.report}, None, 0.0), None
        if not self.env.all_terminated:
            finish_interrupted(self.env, True)
        result.extra["omy"] = {"episode": f"demo_{self.attempts - 1}",
                               "preparation": {k: self.preparation.get(k) for k in ("status", "elapsed_s", "placement")}}
        return result, None


def default_select_params(obj: objects.BatchObject, layout: str, qualified: bool):
    """Selection as ``collect_demos.py`` makes it: screened grasps when ``qualified``."""
    if not qualified:
        return None, None
    found = collect.qualified_grasps(obj.task(layout))
    if found is None:
        return None, None
    ids, params = found
    return params, ids


__all__ = ["OmySession", "PreparationFailed", "collection_options", "default_select_params", "select"]
