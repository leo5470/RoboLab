"""One feedback-driven manipulation state machine for all normalized targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from .geometry import Pose
from .task_spec import ManipulationTarget, ScriptedTaskSpec


class ScriptedState(str, Enum):
    WARMUP = "WARMUP"
    SELECT_TARGET = "SELECT_TARGET"
    SELECT_GRASP = "SELECT_GRASP"
    MOVE_TO_PREGRASP = "MOVE_TO_PREGRASP"
    APPROACH = "APPROACH"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    VERIFY_GRASP = "VERIFY_GRASP"
    LIFT = "LIFT"
    CLEAR_SOURCE = "CLEAR_SOURCE"
    SELECT_PLACEMENT = "SELECT_PLACEMENT"
    MOVE_TO_PREPLACE = "MOVE_TO_PREPLACE"
    LOWER = "LOWER"
    OPEN_GRIPPER = "OPEN_GRIPPER"
    VERIFY_PLACEMENT = "VERIFY_PLACEMENT"
    RETREAT = "RETREAT"
    ADVANCE_TARGET = "ADVANCE_TARGET"
    SUCCESS = "SUCCESS"
    RECOVER = "RECOVER"
    FAILED = "FAILED"


class FailureReason(str, Enum):
    UNSUPPORTED_TASK_SPEC = "UNSUPPORTED_TASK_SPEC"
    MISSING_ENTITY = "MISSING_ENTITY"
    NO_VALID_GRASP_CANDIDATE = "NO_VALID_GRASP_CANDIDATE"
    NO_VALID_PLACEMENT = "NO_VALID_PLACEMENT"
    PREGRASP_TIMEOUT = "PREGRASP_TIMEOUT"
    APPROACH_TIMEOUT = "APPROACH_TIMEOUT"
    GRASP_NOT_VERIFIED = "GRASP_NOT_VERIFIED"
    SOURCE_CLEARANCE_TIMEOUT = "SOURCE_CLEARANCE_TIMEOUT"
    PLACEMENT_TIMEOUT = "PLACEMENT_TIMEOUT"
    TASK_NOT_TERMINATED = "TASK_NOT_TERMINATED"
    IK_DIVERGENCE = "IK_DIVERGENCE"


@dataclass(frozen=True)
class MotionCommand:
    pose_robot: Pose | None
    gripper_command: float
    state: ScriptedState


@dataclass(frozen=True)
class StateMachineConfig:
    warmup_steps: int = 20
    close_steps: int = 12
    open_steps: int = 12
    settle_steps: int = 20
    motion_timeout_steps: int = 180
    grasp_verify_timeout_steps: int = 100
    source_clearance_timeout_steps: int = 180
    placement_timeout_steps: int = 180
    task_success_timeout_steps: int = 80
    max_grasp_attempts: int = 4
    max_placement_attempts: int = 3


class ManipulationContext(Protocol):
    """Relation/planning adapter used by the task-independent FSM."""

    def hold_pose(self) -> Pose: ...
    def target_already_satisfied(self, target: ManipulationTarget) -> bool: ...
    def prepare_target(self, target: ManipulationTarget) -> None: ...
    def select_grasp(self, attempt: int) -> bool: ...
    def pregrasp_pose(self) -> Pose: ...
    def grasp_pose(self) -> Pose: ...
    def verification_lift_pose(self) -> Pose: ...
    def lift_pose(self) -> Pose: ...
    def source_clearance_pose(self) -> Pose: ...
    def source_is_clear(self) -> bool: ...
    def select_placement(self, attempt: int) -> bool: ...
    def preplace_pose(self) -> Pose: ...
    def release_pose(self) -> Pose: ...
    def retreat_pose(self) -> Pose: ...
    def recovery_pose(self) -> Pose: ...
    def pose_reached(self, pose: Pose) -> bool: ...
    def grasp_pose_reached(self, pose: Pose) -> bool: ...
    def gripper_object_contact(self) -> bool: ...
    def verified_grasp(self) -> bool: ...
    def placement_verified(self) -> bool: ...
    def task_success(self) -> bool: ...


_MOTION_TIMEOUTS = {
    ScriptedState.MOVE_TO_PREGRASP: FailureReason.PREGRASP_TIMEOUT,
    ScriptedState.APPROACH: FailureReason.APPROACH_TIMEOUT,
    ScriptedState.LIFT: FailureReason.GRASP_NOT_VERIFIED,
    ScriptedState.CLEAR_SOURCE: FailureReason.SOURCE_CLEARANCE_TIMEOUT,
    ScriptedState.MOVE_TO_PREPLACE: FailureReason.PLACEMENT_TIMEOUT,
    ScriptedState.LOWER: FailureReason.PLACEMENT_TIMEOUT,
    ScriptedState.RETREAT: FailureReason.PLACEMENT_TIMEOUT,
}


@dataclass
class ScriptedManipulationStateMachine:
    spec: ScriptedTaskSpec
    cfg: StateMachineConfig = field(default_factory=StateMachineConfig)
    state: ScriptedState = ScriptedState.WARMUP
    failure_reason: FailureReason | None = None
    failure_detail: str = ""
    target_index: int = -1
    completed_indices: set[int] = field(default_factory=set)
    skipped_indices: set[int] = field(default_factory=set)
    state_steps: int = 0
    grasp_attempt: int = 0
    placement_attempt: int = 0
    _recovery_after_placement: bool = False

    def reset(self) -> None:
        self.state = ScriptedState.WARMUP
        self.failure_reason = None
        self.failure_detail = ""
        self.target_index = -1
        self.completed_indices.clear()
        self.skipped_indices.clear()
        self.state_steps = 0
        self.grasp_attempt = 0
        self.placement_attempt = 0
        self._recovery_after_placement = False

    @property
    def done(self) -> bool:
        return self.state in {ScriptedState.SUCCESS, ScriptedState.FAILED}

    @property
    def current_target(self) -> ManipulationTarget | None:
        if 0 <= self.target_index < len(self.spec.targets):
            return self.spec.targets[self.target_index]
        return None

    def _transition(self, state: ScriptedState) -> None:
        self.state = state
        self.state_steps = 0

    def _fail(self, reason: FailureReason, detail: str) -> None:
        self.failure_reason = reason
        self.failure_detail = detail
        self._transition(ScriptedState.FAILED)

    def fail(self, reason: FailureReason, detail: str) -> None:
        """Stop safely when the controller adapter detects an external failure."""
        self._fail(reason, detail)

    def _completion_reached(self) -> bool:
        count = len(self.completed_indices)
        if self.spec.completion_mode == "any":
            return count >= 1
        if self.spec.completion_mode == "choose":
            return count >= int(self.spec.required_count or 0)
        return count >= len(self.spec.targets)

    def _next_target_index(self) -> int | None:
        unavailable = self.completed_indices | self.skipped_indices
        remaining = [index for index in range(len(self.spec.targets)) if index not in unavailable]
        if not remaining:
            return None
        if self.spec.ordered:
            minimum_stage = min(self.spec.targets[index].stage for index in remaining)
            remaining = [index for index in remaining if self.spec.targets[index].stage == minimum_stage]
        return remaining[0]

    def step(self, context: ManipulationContext) -> MotionCommand:
        """Advance from feedback and return an explicit absolute-pose/gripper command."""
        self.state_steps += 1
        # Selection states are zero-time; consume them in one call until a motion is emitted.
        for _ in range(8):
            hold = context.hold_pose()
            if self.state == ScriptedState.WARMUP:
                if self.state_steps >= self.cfg.warmup_steps:
                    self._transition(ScriptedState.SELECT_TARGET)
                    continue
                return MotionCommand(hold, 0.0, self.state)

            if self.state == ScriptedState.SELECT_TARGET:
                if self._completion_reached():
                    self._transition(ScriptedState.SUCCESS)
                    continue
                next_index = self._next_target_index()
                if next_index is None:
                    self._fail(FailureReason.TASK_NOT_TERMINATED, "No remaining target but completion rule is unmet")
                    continue
                self.target_index = next_index
                target = self.spec.targets[next_index]
                if context.target_already_satisfied(target):
                    self.completed_indices.add(next_index)
                    continue
                context.prepare_target(target)
                self.grasp_attempt = 0
                self.placement_attempt = 0
                self._transition(ScriptedState.SELECT_GRASP)
                continue

            if self.state == ScriptedState.SELECT_GRASP:
                if self.grasp_attempt >= self.cfg.max_grasp_attempts or not context.select_grasp(self.grasp_attempt):
                    if self._skip_current_target_if_allowed():
                        continue
                    self._fail(FailureReason.NO_VALID_GRASP_CANDIDATE,
                               f"target={self.current_target} attempts={self.grasp_attempt}")
                    continue
                self._transition(ScriptedState.MOVE_TO_PREGRASP)
                continue

            if self.state == ScriptedState.SELECT_PLACEMENT:
                if self.placement_attempt >= self.cfg.max_placement_attempts or not context.select_placement(self.placement_attempt):
                    self._fail(FailureReason.NO_VALID_PLACEMENT,
                               f"target={self.current_target} attempts={self.placement_attempt}")
                    continue
                self._transition(ScriptedState.MOVE_TO_PREPLACE)
                continue

            if self.state == ScriptedState.MOVE_TO_PREGRASP:
                pose = context.pregrasp_pose()
                if context.pose_reached(pose):
                    self._transition(ScriptedState.APPROACH)
                elif self.state_steps > self.cfg.motion_timeout_steps:
                    self._start_grasp_recovery(FailureReason.PREGRASP_TIMEOUT)
                return MotionCommand(pose, 0.0, self.state)

            if self.state == ScriptedState.APPROACH:
                pose = context.grasp_pose()
                if context.grasp_pose_reached(pose):
                    self._transition(ScriptedState.CLOSE_GRIPPER)
                elif self.state_steps > self.cfg.motion_timeout_steps:
                    self._start_grasp_recovery(FailureReason.APPROACH_TIMEOUT)
                return MotionCommand(pose, 0.0, self.state)

            if self.state == ScriptedState.CLOSE_GRIPPER:
                pose = context.grasp_pose()
                if self.state_steps >= self.cfg.close_steps:
                    if context.gripper_object_contact():
                        self._transition(ScriptedState.VERIFY_GRASP)
                    else:
                        self._start_grasp_recovery(FailureReason.GRASP_NOT_VERIFIED)
                return MotionCommand(pose, 1.0, self.state)

            if self.state == ScriptedState.VERIFY_GRASP:
                pose = context.verification_lift_pose()
                if context.pose_reached(pose) and context.verified_grasp():
                    self._transition(ScriptedState.LIFT)
                elif self.state_steps > self.cfg.grasp_verify_timeout_steps or not context.gripper_object_contact():
                    self._start_grasp_recovery(FailureReason.GRASP_NOT_VERIFIED)
                return MotionCommand(pose, 1.0, self.state)

            if self.state == ScriptedState.LIFT:
                pose = context.lift_pose()
                if context.pose_reached(pose):
                    self._transition(ScriptedState.CLEAR_SOURCE)
                elif self.state_steps > self.cfg.motion_timeout_steps:
                    self._start_grasp_recovery(FailureReason.GRASP_NOT_VERIFIED)
                return MotionCommand(pose, 1.0, self.state)

            if self.state == ScriptedState.CLEAR_SOURCE:
                pose = context.source_clearance_pose()
                if context.pose_reached(pose) and context.source_is_clear():
                    self._transition(ScriptedState.SELECT_PLACEMENT)
                elif self.state_steps > self.cfg.source_clearance_timeout_steps:
                    self._start_grasp_recovery(FailureReason.SOURCE_CLEARANCE_TIMEOUT)
                return MotionCommand(pose, 1.0, self.state)

            if self.state == ScriptedState.MOVE_TO_PREPLACE:
                pose = context.preplace_pose()
                if context.pose_reached(pose):
                    self._transition(ScriptedState.LOWER)
                elif self.state_steps > self.cfg.placement_timeout_steps:
                    self._start_placement_recovery(FailureReason.PLACEMENT_TIMEOUT)
                return MotionCommand(pose, 1.0, self.state)

            if self.state == ScriptedState.LOWER:
                pose = context.release_pose()
                if context.pose_reached(pose):
                    self._transition(ScriptedState.OPEN_GRIPPER)
                elif self.state_steps > self.cfg.placement_timeout_steps:
                    self._start_placement_recovery(FailureReason.PLACEMENT_TIMEOUT)
                return MotionCommand(pose, 1.0, self.state)

            if self.state == ScriptedState.OPEN_GRIPPER:
                pose = context.release_pose()
                if self.state_steps >= self.cfg.open_steps:
                    self._transition(ScriptedState.VERIFY_PLACEMENT)
                return MotionCommand(pose, 0.0, self.state)

            if self.state == ScriptedState.VERIFY_PLACEMENT:
                pose = context.release_pose()
                if self.state_steps >= self.cfg.settle_steps and context.placement_verified():
                    self._transition(ScriptedState.RETREAT)
                elif self.state_steps > self.cfg.placement_timeout_steps:
                    self._start_placement_recovery(FailureReason.PLACEMENT_TIMEOUT)
                return MotionCommand(pose, 0.0, self.state)

            if self.state == ScriptedState.RETREAT:
                pose = context.retreat_pose()
                if context.pose_reached(pose):
                    self._transition(ScriptedState.ADVANCE_TARGET)
                elif self.state_steps > self.cfg.motion_timeout_steps:
                    self._fail(FailureReason.PLACEMENT_TIMEOUT, "retreat timed out")
                return MotionCommand(pose, 0.0, self.state)

            if self.state == ScriptedState.ADVANCE_TARGET:
                if self.target_index >= 0:
                    self.completed_indices.add(self.target_index)
                if context.task_success() or self._completion_reached():
                    self._transition(ScriptedState.SUCCESS)
                else:
                    self._transition(ScriptedState.SELECT_TARGET)
                continue

            if self.state == ScriptedState.RECOVER:
                pose = context.recovery_pose()
                if context.pose_reached(pose):
                    self._transition(ScriptedState.SELECT_GRASP)
                elif self.state_steps > self.cfg.motion_timeout_steps:
                    reason = self.failure_reason or FailureReason.PREGRASP_TIMEOUT
                    self._fail(
                        reason, f"{self.failure_detail}; recovery retreat timed out"
                    )
                return MotionCommand(pose, 0.0, self.state)

            return MotionCommand(hold, 0.0, self.state)

        self._fail(FailureReason.TASK_NOT_TERMINATED, "Exceeded zero-time transition limit")
        return MotionCommand(context.hold_pose(), 0.0, self.state)

    def _skip_current_target_if_allowed(self) -> bool:
        if self.spec.completion_mode == "all" or self.target_index < 0:
            return False
        self.skipped_indices.add(self.target_index)
        required = 1 if self.spec.completion_mode == "any" else int(self.spec.required_count or 0)
        possible = len(self.completed_indices) + len(self.spec.targets) - len(
            self.completed_indices | self.skipped_indices
        )
        if possible < required:
            return False
        self._transition(ScriptedState.SELECT_TARGET)
        return True

    def _start_grasp_recovery(self, reason: FailureReason) -> None:
        self.failure_reason = reason
        self.failure_detail = f"grasp attempt {self.grasp_attempt} failed"
        self._recovery_after_placement = False
        self.grasp_attempt += 1
        if self.grasp_attempt >= self.cfg.max_grasp_attempts:
            self._fail(reason, f"exhausted {self.grasp_attempt} grasp attempts")
        else:
            self._transition(ScriptedState.RECOVER)

    def _start_placement_recovery(self, reason: FailureReason) -> None:
        self.failure_reason = reason
        self.failure_detail = f"placement attempt {self.placement_attempt} failed"
        self._recovery_after_placement = True
        self.placement_attempt += 1
        self.grasp_attempt += 1
        if self.placement_attempt >= self.cfg.max_placement_attempts:
            self._fail(reason, f"exhausted {self.placement_attempt} placement attempts")
        elif self.grasp_attempt >= self.cfg.max_grasp_attempts:
            self._fail(reason, f"placement retry exhausted {self.grasp_attempt} total grasps")
        else:
            self._transition(ScriptedState.RECOVER)

