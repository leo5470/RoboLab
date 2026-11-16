"""One env, many scripted attempts: the loop shared by the transfer test, the trials, screening and collection.

The collector owns the env loop (D6): ``create_env`` + its own steps, so it has in-process ``WorldState``.
Every recorded row goes through ``Pi0DroidJointposClient``'s own observation conversion (requirement R1).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time

import numpy as np

from robolab.core.scripted_demos.grasps import library as grasp_library
from robolab.core.scripted_demos.grasps import objects
from robolab.core.scripted_demos import motion, select, sim
from robolab.core.scripted_demos.kinematics import Chain
from robolab.core.scripted_demos.writer import EpisodeBuffer


def attempt_rng(obj: objects.BatchObject, layout: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([obj.index, int(layout), int(seed)]))


def wire_client():
    """``Pi0DroidJointposClient`` for its conversion methods only. Its constructor connects to a server, so it is
    bypassed; ``_extract_observation`` and ``_pack_request`` use no instance state."""
    from policies.pi0_family.client import Pi0DroidJointposClient

    return Pi0DroidJointposClient.__new__(Pi0DroidJointposClient)


class VideoWriter:
    """Exterior camera (main, 848x480) | wrist camera over the viewport camera (432 wide), one frame per row, H.264."""

    def __init__(self, path: pathlib.Path, fps: int = 15):
        import imageio.v2 as imageio

        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(str(self.path), fps=fps, codec="libx264", quality=7, macro_block_size=8)

    def write(self, viewport: np.ndarray, exterior: np.ndarray, wrist: np.ndarray, label: str) -> None:
        import cv2

        main = cv2.resize(exterior, (848, 480), interpolation=cv2.INTER_AREA)
        right = np.vstack([cv2.resize(wrist, (432, 243), interpolation=cv2.INTER_AREA),
                           cv2.resize(viewport, (432, 237), interpolation=cv2.INTER_AREA)])
        frame = np.ascontiguousarray(np.hstack([main, right]))
        cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        self.writer.append_data(frame)

    def close(self) -> None:
        self.writer.close()


@dataclasses.dataclass
class Result:
    task: str
    layout: str
    seed: int
    success: bool
    failure: str | None
    steps: int
    plan: dict | None
    selection: dict
    executor: dict | None
    wall_s: float
    extra: dict = dataclasses.field(default_factory=dict)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


class Session:
    """An env for one layout task of one batch object, plus everything needed to run attempts in it."""

    def __init__(self, obj: objects.BatchObject, layout: str, *, render: bool, robolab_output: pathlib.Path,
                 library_path: pathlib.Path | None = None, events: dict | None = None):
        sim.launch_app()
        from robolab.core.world.world_state import get_world

        self.obj, self.layout = obj, layout
        self.task = obj.task(layout)
        self.render = render
        self.env, self.env_cfg = sim.make_env(self.task, robolab_output / self.task, render=render, events=events)
        self.world = get_world(self.env)
        root = objects.grasp_data_root()
        self.chain = Chain.load(root / "robot_chain.npz")
        self.ik = sim.ArmIK(self.env, self.chain)
        self.library_path = library_path or root / "library" / f"{obj.key}.npz"
        self.library = grasp_library.load(self.library_path)
        self.client = wire_client()
        self.instruction = self.env_cfg.instruction
        self.obs = None

    def close(self) -> None:
        self.env.close()

    def reset(self, seed: int):
        """Deterministic: these tasks' only reset event is ``reset_scene_to_default``. Two resets, like the runner."""
        import torch

        self.env.reset_eval_state()
        self.env.reset(seed=int(seed))
        self.obs, _ = self.env.reset()
        torch.cuda.synchronize()
        self.ik.reset_command()
        return self.obs

    def _record(self, buffer, writer, action, seed, steps, label) -> None:
        if buffer is None and writer is None:
            return
        request = self.client._pack_request(self.client._extract_observation(self.obs), self.instruction)
        if buffer is not None:
            state, joint_names = sim.pre_action_state(self.env, (self.obj.name, sim.BIN))
            buffer.add(request, action, self.ik.finger_position(), self.sim_time(), state, joint_names)
        if writer is not None:
            from robolab.core.observations.observation_utils import unpack_viewport_cams

            viewport = unpack_viewport_cams(self.obs, env_id=0)["combined_image"]
            raw = self.client._extract_observation(self.obs)  # full-resolution camera frames
            writer.write(viewport, raw["right_image"], raw["wrist_image"], f"{self.task}  seed {seed}  step {steps:3d}  {label}")

    def settle(self, buffer, writer, seed, *, min_steps: int = 4, max_steps: int = 45, speed: float = 0.01) -> int:
        """Hold the arm (gripper open) until the object is still for 3 consecutive steps. Returns steps taken."""
        hold = np.r_[self.ik.joints(), 0.0].astype(np.float32)
        still = 0
        for k in range(max_steps):
            self._record(buffer, writer, hold, seed, k, "settle")
            self.step(hold)
            still = still + 1 if sim.object_speed(self.world, self.obj.name) < speed else 0
            if k + 1 >= min_steps and still >= 3:
                return k + 1
        return max_steps

    def place_objects(self, poses: dict[str, np.ndarray], hold_steps: int = 3) -> dict:
        """Sampled scenes: teleport objects (name → env-local pose) after a reset, holding the arm for ``hold_steps``
        unrecorded steps and writing the poses again after each, so the next observation shows the objects in place.

        A single write is not enough: on the first physics step after it the pitcher is driven back toward its
        pre-write pose (moved 8 cm and turned 135° from layout 06's: thrown 13 cm and tipped 87°, every time in a fresh
        env); writing again after each of 3 steps leaves it within 1 mm (randomized-plan results, Phase 2).
        ``replay_check.py`` repeats exactly this, so an episode's history is reset, these writes and steps, then the
        recorded rows."""
        def write():
            for name, T in poses.items():
                sim.write_pose(self.env, name, T)

        write()
        hold = np.r_[self.ik.joints(), 0.0].astype(np.float32)
        for _ in range(hold_steps):
            self.step(hold)
            write()
        # Full precision: replay writes these again, and 1e-6 differences grow once the gripper holds the object.
        return {"placed": {k: np.asarray(v, dtype=np.float64).tolist() for k, v in poses.items()},
                "hold_steps": hold_steps, "rewrite_each_step": True}

    def sim_time(self) -> float:
        try:
            return float(self.env.sim.current_time)
        except Exception:
            return float(self.env.episode_length_buf[0]) * self.env.step_dt

    def step(self, action: np.ndarray):
        import torch

        a = torch.as_tensor(np.asarray(action, dtype=np.float32), device=self.env.device)[None]
        self.obs, _, term, trunc, _ = self.env.step(a)
        return bool(term.any()), bool(trunc.any())

    def env_success(self) -> bool | None:
        return self.env.get_env_results()[0]["success"]

    def run(self, seed: int, *, mode: str = "pnp", select_params: select.Params | None = None,
            motion_params: motion.MotionParams | None = None, allowed=None, plan: select.Plan | None = None,
            record: bool = False, video: pathlib.Path | None = None,
            keep_failed_video: bool = False, prepare=None, plan_fn=None) -> tuple[Result, EpisodeBuffer | None]:
        """One attempt from a fresh reset. ``plan`` skips selection (transfer test, screening).

        Sampled scenes (MimicGen): ``prepare(session) -> dict`` runs right after the reset, before anything is
        recorded (it places the objects); ``plan_fn(scene) -> (plan or None, report)`` replaces selection and is
        given the settled scene."""
        t0 = time.time()
        motion_params = motion_params or motion.MotionParams()
        select_params = select_params or select.Params(motion=motion_params)
        self.reset(seed)
        prepared = prepare(self) if prepare is not None else None
        rng = attempt_rng(self.obj, self.layout, seed)
        buffer = EpisodeBuffer(source_sim_origin_s=self.sim_time()) if record else None
        writer = VideoWriter(video.with_suffix(".partial.mp4")) if video is not None else None
        # The VOMP jugs' origin is at their base but the layouts place them as if centred, so they start up to
        # 14 cm above the table; hold the arm still until the object has come to rest, then plan.
        settle_steps = self.settle(buffer, writer, seed)
        report: dict = {"settle_steps": settle_steps}
        if prepared is not None:
            report["prepare"] = prepared
        if plan is None and plan_fn is not None:
            plan, chosen = plan_fn(sim.read_scene(self.world, self.ik, self.obj))
            report |= chosen
        elif plan is None:
            plan, chosen = select.select(self.library, sim.read_scene(self.world, self.ik, self.obj), self.chain, rng,
                                         select_params, allowed)
            report |= chosen
        if plan is None:
            if writer is not None:
                writer.close()
                if keep_failed_video and prepared is not None:  # a sampled scene rejected after settling: keep it
                    video.with_suffix(".partial.mp4").replace(video.with_suffix(".failed.mp4"))
                else:
                    video.with_suffix(".partial.mp4").unlink(missing_ok=True)
            return Result(self.task, self.layout, seed, False, report.get("failure", "no_feasible_grasp"), settle_steps,
                          None, report, None, time.time() - t0), None

        self.ik.reset_command()
        executor = motion.Executor(plan, self.ik, rng, motion_params, mode=mode)
        object_start = sim.object_pose(self.world, self.obj.name)
        success, failure, steps = False, None, settle_steps
        try:
            while True:
                action = executor.action()
                self._record(buffer, writer, action, seed, steps, executor.name)
                term, trunc = self.step(action)
                steps += 1
                if term or trunc:
                    success = bool(self.env_success())
                    if not success:
                        failure = executor.failure or "env_terminated_other"
                    break
                failure = executor.update(sim.object_pose(self.world, self.obj.name), sim.closedness(self.ik),
                                          sim.finger_contact(self.world, self.obj))
                if failure or executor.finished:
                    break
        finally:
            if writer is not None:
                writer.close()
        extra = {}
        if mode == "transfer":
            obj_now = sim.object_pose(self.world, self.obj.name)
            lift = float(obj_now[2, 3] - object_start[2, 3])
            summary = executor.summary()
            contact = summary["held_finger_contact_fraction"]  # None when the scene has no gripper contact sensor
            extra = {"object_lift_m": lift,
                     "lifted": bool(executor.finished and lift > 0.05 and (contact is None or contact >= 0.8))}
            success = extra["lifted"]
            if not success and failure is None:
                failure = "slip"  # finished the hold without a lifted, held object
        if writer is not None:
            final = video.with_suffix(".mp4")
            if success or keep_failed_video:
                video.with_suffix(".partial.mp4").replace(final if success else video.with_suffix(".failed.mp4"))
            else:
                video.with_suffix(".partial.mp4").unlink(missing_ok=True)
        result = Result(self.task, self.layout, seed, success, None if success else failure, steps, plan.to_json(),
                        report, executor.summary(), time.time() - t0, extra)
        return result, buffer


def qualified_grasps(task: str) -> tuple[list[int], select.Params] | None:
    """Phase 4b screening of ``task`` (``screen`` in scripts/scripted_demos/trials.py): the library ids that succeeded end to
    end, and the selection/motion configuration they were screened with, or None when the task was not screened.

    Draws from qualified grasps must reuse that configuration (corridor filter, retreat, durations, ...): under another
    one a qualified grasp can be filtered out again. Files written before the configuration was saved are refused.
    """
    path = objects.grasp_data_root() / "qualified" / f"{task}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if "params" not in data:
        raise ValueError(f"{path} was screened without a saved configuration; re-screen it (trials.py screen)")
    return sorted({r["grasp_id"] for r in data["trials"] if r["success"]}), select.Params.from_json(data["params"])


def append_jsonl(path: pathlib.Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n")
