"""Scripted motion: segment timing, reference interpolation, the per-step executor, and a numpy replay of its
joint commands (Phase 4 step 2).

``Executor`` is a per-control-step state machine: ``action()`` gives the 8-D command for the current step (seven
absolute joint targets from ``sim.ArmIK`` + gripper 0 open / 1 close) and ``update()``, called after ``env.step``,
advances the segment and runs the failure checks. The caller owns ``env.step`` so it can record every row.

Segments: pregrasp → approach → close → lift → preplace → seat → open → retreat → retreat_up → settle
(``mode="pnp"``), or pregrasp → approach → close → lift → hold (``mode="transfer"``, the Phase 3 lift test).

The joint commands depend only on the plan (``sim.ArmIK`` integrates IsaacLab's DLS on the previous command), so
``emulate`` reproduces them kinematically, batched over grasp candidates; the selector keeps only grasps whose
commanded motion converges on every segment and whose path to pregrasp clears the object.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from robolab.core.scripted_demos.grasps import frames
from robolab.core.scripted_demos.kinematics import orientation_error

if TYPE_CHECKING:
    from robolab.core.scripted_demos.select import Plan

FAILURES = ("no_feasible_grasp", "ik_unreachable", "collision_abort", "slip", "tracking_error", "release_failure",
            "timeout", "env_terminated_other")

# Nominal control steps per moving segment (15 Hz), before the random scaling.
NOMINAL = {"pregrasp": 36, "approach": 18, "close": 12, "lift": 16, "preplace": 30, "seat": 14, "open": 8,
           "retreat": 8, "retreat_up": 8}
TRANSFER = {"pregrasp": 36, "approach": 18, "close": 12, "lift": 16, "hold": 15}
GRIPPER = {"pregrasp": 0.0, "approach": 0.0, "close": 1.0, "lift": 1.0, "preplace": 1.0, "seat": 1.0, "open": 0.0,
           "retreat": 0.0, "retreat_up": 0.0, "settle": 0.0, "hold": 1.0}
STATIONARY = {"close", "open", "settle", "hold"}
HOLDING = {"lift", "preplace", "seat", "hold"}


@dataclasses.dataclass
class MotionParams:
    duration_scale: tuple[float, float] = (0.8, 1.2)
    motion_steps: tuple[int, int] = (112, 132)  # moving segments; settling before and success detection add rows
    settle_allowance: int = 15  # extra steps a segment may take to converge
    converge_pos: float = 0.005
    converge_rot: float = np.radians(3.0)
    abort_pos: float = 0.02
    abort_rot: float = np.radians(10.0)
    slip_closedness: float = 0.95
    slip_drift: float = 0.05  # held object dropping this far relative to the gripper (m)
    disturb_before_grasp: float = 0.03  # object pushed before closing (m)
    retreat_back: float = 0.12  # 8 cm left tall objects touching the open fingers (utilityjug_a02 release failures)
    retreat_up: float = 0.15
    settle_steps: int = 45  # wait for the success termination after the retreat
    max_steps: int = 300
    segment_scale: dict = dataclasses.field(default_factory=dict)  # e.g. {"preplace": 1.6}: slower, longer episode


def interpolate(T0: np.ndarray, T1: np.ndarray, s: float) -> np.ndarray:
    """Pose between ``T0`` and ``T1``: linear position, slerp orientation."""
    return batch_interpolate(np.asarray(T0)[None], np.asarray(T1)[None], s)[0]


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def segment_steps(rng: np.random.Generator, params: MotionParams) -> dict[str, int]:
    """Random per-segment durations paced to a total in ``motion_steps``; ``segment_scale`` lengthens chosen segments
    and the total with them (the other segments keep their share)."""
    base = {k: v * params.segment_scale.get(k, 1.0) for k, v in NOMINAL.items()}
    scaled = {k: v * rng.uniform(*params.duration_scale) for k, v in base.items()}
    total = rng.integers(params.motion_steps[0], params.motion_steps[1] + 1) * sum(base.values()) / sum(NOMINAL.values())
    factor = total / sum(scaled.values())
    return {k: max(4, int(round(v * factor))) for k, v in scaled.items()}


def segment_targets(waypoints: dict, params: MotionParams, mode: str = "pnp") -> dict[str, np.ndarray]:
    """The pose each segment moves to (or holds), from the plan's waypoints."""
    w = waypoints
    targets = {"pregrasp": w["pregrasp"], "approach": w["grasp"], "close": w["grasp"], "lift": w["lift"],
               "hold": w["lift"]}
    if mode == "pnp":
        seat = w["seat"]
        back = np.array(seat, copy=True)
        back[..., :3, 3] -= params.retreat_back * seat[..., :3, frames.APPROACH_AXIS]
        up = back.copy()
        up[..., 2, 3] += params.retreat_up
        targets |= {"preplace": w["preplace"], "seat": seat, "open": seat, "retreat": back, "retreat_up": up,
                    "settle": up}
    return targets


def batch_interpolate(T0: np.ndarray, T1: np.ndarray, s: float, q0=None, q1=None) -> np.ndarray:
    """``interpolate`` over a batch (N, 4, 4); pass the endpoint quaternions to skip converting them again."""
    q0 = np.stack([frames.matrix_to_quat(R) for R in T0[:, :3, :3]]) if q0 is None else q0
    q1 = np.stack([frames.matrix_to_quat(R) for R in T1[:, :3, :3]]) if q1 is None else q1
    q1 = np.where((np.sum(q0 * q1, -1) < 0)[:, None], -q1, q1)
    dot = np.clip(np.sum(q0 * q1, -1), -1.0, 1.0)
    th = np.arccos(dot)
    near = dot > 0.9995
    sin = np.where(near, 1.0, np.sin(th))
    w0 = np.where(near, 1 - s, np.sin((1 - s) * th) / sin)
    w1 = np.where(near, s, np.sin(s * th) / sin)
    q = w0[:, None] * q0 + w1[:, None] * q1
    out = np.zeros_like(T0)
    out[:, :3, :3] = frames.quat_to_matrix(q)
    out[:, :3, 3] = (1 - s) * T0[:, :3, 3] + s * T1[:, :3, 3]
    out[:, 3, 3] = 1.0
    return out


def emulate(chain, q0: np.ndarray, T_start: np.ndarray, targets: dict[str, np.ndarray], order: list[str],
            steps: dict[str, int], params: MotionParams, limits: np.ndarray, *, clamp: float = 0.04,
            lag: int = 5, record: tuple[str, ...] = ("pregrasp",)) -> dict:
    """Replay the executor's commands for N candidates at once.

    A moving segment follows the smoothstep reference for its planned steps and may then hold the target for up to
    ``settle_allowance - lag`` more steps (``lag`` leaves room for the joint PD following the commands); it converges
    when the commanded pose is within the executor's thresholds. Returns per-candidate ``converged`` (all moving
    segments), the first failing segment, and the commanded poses of the ``record`` segments (every second step).
    """
    n = len(T_start)
    q = np.broadcast_to(np.asarray(q0, dtype=np.float64), (n, 7)).copy()
    start = np.array(T_start, copy=True)
    ok = np.ones(n, dtype=bool)
    first_fail = np.full(n, "", dtype=object)
    poses: dict[str, list] = {name: [] for name in record}
    lo, hi = limits[:, 0], limits[:, 1]

    def step(ref):
        nonlocal q
        dq, T = chain.dls_delta(q, ref)
        q = np.clip(q + np.clip(dq, -clamp, clamp), lo, hi)
        return T

    for name in order:
        target = targets[name]
        k_total = steps[name]
        stationary = name in STATIONARY
        if not stationary:
            qa = np.stack([frames.matrix_to_quat(R) for R in start[:, :3, :3]])
            qb = np.stack([frames.matrix_to_quat(R) for R in target[:, :3, :3]])
        for k in range(k_total):
            ref = target if stationary else batch_interpolate(start, target, smoothstep((k + 1) / k_total), qa, qb)
            T = step(ref)
            if name in poses and k % 2 == 0:
                poses[name].append(T)
        if not stationary:
            converged = np.zeros(n, dtype=bool)
            for _ in range(max(1, params.settle_allowance - lag) + 1):
                T = chain.fk(q)
                pe = np.linalg.norm(T[:, :3, 3] - target[:, :3, 3], axis=1)
                re = np.linalg.norm(orientation_error(T[:, :3, :3], target[:, :3, :3]), axis=1)
                converged |= (pe < params.converge_pos) & (re < params.converge_rot)
                if converged.all():
                    break
                step(target)
            newly = ok & ~converged
            first_fail[newly] = name
            ok &= converged
        start = np.array(target, copy=True)
    return {"converged": ok, "first_fail": first_fail, "q": q,
            "poses": {k: np.stack(v, 1) if v else np.zeros((n, 0, 4, 4)) for k, v in poses.items()}}


class Executor:
    def __init__(self, plan: "Plan", ik, rng: np.random.Generator, params: MotionParams = MotionParams(),
                 mode: str = "pnp"):
        self.plan, self.ik, self.params, self.mode = plan, ik, params, mode
        self.targets = segment_targets(plan.waypoints, params, mode)
        if mode == "transfer":
            self.steps = dict(TRANSFER)
            self.order = list(TRANSFER)
        else:
            # Durations chosen (and the commands checked) by the selector, or drawn here.
            self.steps = dict(plan.steps) if getattr(plan, "steps", None) else segment_steps(rng, params)
            self.order = list(NOMINAL) + ["settle"]
        self.segment = 0
        self.k = 0
        self.total = 0
        self.start = ik.ee_pose()
        self.failure: str | None = None
        self.failure_detail = ""
        self.done_moving = False
        self.finished = False  # transfer mode: the hold completed
        self.timings: dict[str, int] = {}
        self.max_tracking = {"pos_m": 0.0, "rot_deg": 0.0}
        self.closedness_after_close: float | None = None
        self._held_offset: float | None = None
        self._object_start: np.ndarray | None = None
        self.held_contact_steps = 0
        self.held_steps = 0

    @property
    def name(self) -> str:
        return self.order[self.segment]

    def reference(self) -> np.ndarray:
        name = self.name
        target = self.targets[name]
        if name in STATIONARY:
            return target
        return interpolate(self.start, target, smoothstep((self.k + 1) / self.steps[name]))

    def action(self) -> np.ndarray:
        joints = self.ik.joint_target(self.reference())
        return np.r_[joints, GRIPPER[self.name]].astype(np.float32)

    def _advance(self) -> None:
        self.timings[self.name] = self.k
        self.start = self.targets[self.name]
        self.segment += 1
        self.k = 0

    def update(self, object_pose: np.ndarray, closedness: float, finger_contact: bool | None) -> str | None:
        """Advance after one ``env.step``. Returns a failure class, or None."""
        p = self.params
        self.k += 1
        self.total += 1
        name = self.name
        if self._object_start is None:
            self._object_start = object_pose[:3, 3].copy()
        if self.total > p.max_steps:
            return self._fail("timeout")
        if name in ("pregrasp", "approach") and np.linalg.norm(object_pose[:3, 3] - self._object_start) > p.disturb_before_grasp:
            return self._fail("collision_abort", f"object pushed {np.linalg.norm(object_pose[:3, 3] - self._object_start):.3f} m in {name}")
        if name in HOLDING:
            if finger_contact is not None:
                self.held_steps += 1
                self.held_contact_steps += int(finger_contact)
            if closedness > p.slip_closedness:
                return self._fail("slip", f"closedness {closedness:.3f} in {name}")
            if name != "seat" and self._held_offset is not None:  # seating may legitimately push the object
                # Vertical drop relative to the gripper: a held object may swing about the grasp, but must not fall.
                drop = float(self._held_offset - (self.ik.ee_pose()[2, 3] - object_pose[2, 3]))
                if drop < -p.slip_drift:
                    return self._fail("slip", f"object dropped {-drop:.3f} m relative to the gripper in {name}")

        if name == "settle":
            if self.k >= p.settle_steps:
                return self._fail("release_failure")
            return None
        if self.k < self.steps[name]:
            return None
        # End of the segment's budget.
        if name == "close":
            self.closedness_after_close = closedness
            if closedness > p.slip_closedness:
                return self._fail("slip", f"closed on nothing (closedness {closedness:.3f})")
            self._held_offset = float(self.ik.ee_pose()[2, 3] - object_pose[2, 3])  # gripper height above the object
            self._advance()
            return None
        if name == "hold":
            self.timings[name] = self.k
            self.finished = True
            return None
        if name in STATIONARY:
            self._advance()
            return None
        pos_err, rot_err = self.ik.pose_error(self.targets[name])
        if pos_err < p.converge_pos and rot_err < p.converge_rot:
            self._advance()
        elif self.k >= self.steps[name] + p.settle_allowance:
            self.max_tracking["pos_m"] = max(self.max_tracking["pos_m"], pos_err)
            self.max_tracking["rot_deg"] = max(self.max_tracking["rot_deg"], float(np.degrees(rot_err)))
            if pos_err > p.abort_pos or rot_err > p.abort_rot:
                return self._fail("tracking_error", f"{name}: {pos_err:.3f} m, {np.degrees(rot_err):.1f} deg")
            self._advance()
        if self.name == "settle":
            self.done_moving = True
        return None

    def _fail(self, failure: str, detail: str = "") -> str:
        self.failure = failure
        self.failure_detail = f"{detail} (step {self.k} of {self.name})" if detail else ""
        self.timings[self.name] = self.k
        return failure

    def summary(self) -> dict:
        return {"segment_steps_planned": self.steps, "segment_steps_taken": self.timings, "total_steps": self.total,
                "failure_segment": self.name if self.failure else None, "failure_detail": self.failure_detail,
                "max_tracking_error": self.max_tracking,
                "closedness_after_close": self.closedness_after_close,
                "held_finger_contact_fraction": (self.held_contact_steps / self.held_steps) if self.held_steps else None}
