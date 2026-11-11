"""RoboLab WorldState adapter and batched absolute-IK scripted expert policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from robolab.core.task import conditionals
from robolab.core.world.world_state import get_world
from robolab.robots.droid import EEF_OFFSET_POS, EEF_OFFSET_ROT

from .geometry import (
    Pose,
    env_pose_to_robot,
    oriented_box_from_world,
    pose_from_world_state,
    quat_angle_wxyz,
    quat_slerp_wxyz,
    robot_pose_to_env,
    robot_root_pose_env,
)
from .grasp_generation import GeometryGraspCandidateGenerator, GraspCandidate, GraspGeneratorConfig
from .placement_generation import GeometryPlacementGenerator, PlacementCandidate, PlacementGeneratorConfig, source_clearance_height
from .profiles import GraspProfileLibrary
from .state_machine import FailureReason, ScriptedManipulationStateMachine, ScriptedState, StateMachineConfig
from .task_spec import ManipulationTarget, ScriptedTaskSpec


@dataclass(frozen=True)
class ScriptedPolicyConfig:
    """Physical and feedback tolerances for the generic controller."""

    position_tolerance: float = 0.012
    grasp_position_tolerance: float = 0.004
    orientation_tolerance_rad: float = np.deg2rad(8.0)
    max_translation_per_step: float = 0.012
    max_rotation_per_step_rad: float = np.deg2rad(8.0)
    maximum_eef_step_distance: float = 0.08
    maximum_eef_step_rotation_rad: float = np.deg2rad(35.0)
    maximum_tracking_position_error: float = 0.06
    maximum_tracking_orientation_error_rad: float = np.deg2rad(35.0)
    verification_lift: float = 0.04
    transport_lift: float = 0.10
    source_clearance_margin: float = 0.035
    preplace_height: float = 0.12
    retreat_height: float = 0.12
    recovery_height: float = 0.10
    grasp_stability_orientation_tolerance_rad: float = np.deg2rad(5.0)
    grasp_stability_position_tolerance: float = 0.008
    grasp_stability_steps: int = 5
    minimum_follow_ratio: float = 0.70
    minimum_verification_motion: float = 0.018
    stationary_linear_speed: float = 0.05
    stationary_angular_speed: float = 0.75
    minimum_reach: float = 0.35
    maximum_reach: float = 0.72
    hard_minimum_reach: float = 0.20
    hard_maximum_reach: float = 0.90
    # Penalize pregrasps outside the arm's preferred radial workspace.  A soft
    # penalty keeps unusual but feasible targets available for retries while
    # preferring less extended configurations with better IK conditioning.
    preferred_reach_penalty_per_meter: float = 6.0
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    grasp_generator: GraspGeneratorConfig = field(default_factory=GraspGeneratorConfig)
    placement_generator: PlacementGeneratorConfig = field(default_factory=PlacementGeneratorConfig)


_PREDICATES = {
    "object_in_container": conditionals.object_in_container,
    "object_on_top": conditionals.object_on_top,
    "object_outside_of_and_on_surface": conditionals.object_outside_of_and_on_surface,
}


class _EnvironmentContext:
    def __init__(self, env, env_id, cfg, profiles, grasp_generator, placement_generator) -> None:
        self.env, self.env_id, self.cfg = env, env_id, cfg
        self.world, self.profiles = get_world(env), profiles
        self.grasp_generator, self.placement_generator = grasp_generator, placement_generator
        self.target: ManipulationTarget | None = None
        self.grasps: list[GraspCandidate] = []
        self.grasp: GraspCandidate | None = None
        self.placements: list[PlacementCandidate] = []
        self.placement: PlacementCandidate | None = None
        self.object_in_eef: Pose | None = None
        self.grasp_start_object_pose: Pose | None = None
        self.grasp_start_object_in_eef: Pose | None = None
        self.grasp_start_eef_pose: Pose | None = None
        self._last_object_in_eef: Pose | None = None
        self._stable_grasp_steps = 0
        self._clearance_pose: Pose | None = None
        self._lift_pose: Pose | None = None
        self._release_pose: Pose | None = None
        self._recovery_pose: Pose | None = None
        self.latest_success = False
        self.grasp_attempt_history: list[dict[str, Any]] = []

    def hold_pose(self) -> Pose:
        return self.current_eef_pose_robot()

    def current_eef_pose_env(self) -> Pose:
        pos_w, quat_w = self.world.get_frame_pose("eef_frame", env_id=self.env_id)
        origin = self.env.scene.env_origins[self.env_id].detach().cpu().numpy()
        return Pose(pos_w.detach().cpu().numpy() - origin, quat_w.detach().cpu().numpy())

    def current_eef_pose_robot(self) -> Pose:
        return env_pose_to_robot(self.current_eef_pose_env(), robot_root_pose_env(self.world, self.env_id))

    def target_already_satisfied(self, target: ManipulationTarget) -> bool:
        return self._verify_relation(target)

    def prepare_target(self, target: ManipulationTarget) -> None:
        self.target = target
        self.grasps, self.placements = [], []
        self.grasp = self.placement = None
        self.object_in_eef = self.grasp_start_object_pose = self.grasp_start_object_in_eef = None
        self.grasp_start_eef_pose = self._last_object_in_eef = None
        self._stable_grasp_steps = 0
        self._clearance_pose = self._lift_pose = self._release_pose = self._recovery_pose = None

    def select_grasp(self, attempt: int) -> bool:
        target = self._target()
        # Regenerate from the live pose on every retry. This is essential after a
        # failed placement, because the released object is no longer at its source.
        profile = self.profiles.select(
            target.object_name, self._physical_dimensions(target.object_name),
            metadata=self._asset_metadata(target.object_name),
        )
        obstacles = [name for name in self.world.objects
                     if name not in {target.object_name, target.destination_name, target.source_name}]
        self.grasps = self.grasp_generator.generate(
            target.object_name, self.world, profile, self.env_id, obstacle_names=obstacles,
            source_name=target.source_name,
        )
        reach_center = (self.cfg.minimum_reach + self.cfg.maximum_reach) / 2.0
        ranked_grasps = []
        for index, candidate in enumerate(self.grasps):
            radius = float(np.linalg.norm(self._env_to_robot(candidate.pregrasp_pose_env).position))
            if not self.cfg.hard_minimum_reach <= radius <= self.cfg.hard_maximum_reach:
                continue
            reach_violation = max(
                self.cfg.minimum_reach - radius,
                radius - self.cfg.maximum_reach,
                0.0,
            )
            adjusted_score = (
                candidate.score
                - self.cfg.preferred_reach_penalty_per_meter * reach_violation
            )
            ranked_grasps.append((-adjusted_score, abs(radius - reach_center), index, candidate))
        ranked = [item[-1] for item in sorted(ranked_grasps, key=lambda item: item[:-1])]
        self.grasps = self._diversify_grasps(ranked)
        if attempt >= len(self.grasps):
            return False
        self.grasp = self.grasps[attempt]
        self.grasp_attempt_history.append({
            "attempt": attempt,
            "label": self.grasp.label,
            "grasp_type": self.grasp.grasp_type,
            "score": float(self.grasp.score),
            "required_opening": float(self.grasp.required_opening),
            "pregrasp_position_env": self.grasp.pregrasp_pose_env.position.tolist(),
            "grasp_position_env": self.grasp.pose_env.position.tolist(),
            "approach_axis_env": self.grasp.approach_axis_env.tolist(),
            "closing_axis_env": self.grasp.closing_axis_env.tolist(),
        })
        # A prior placement failure may have changed the object's pose. Rebuild
        # footprint and release candidates after the new grasp is verified.
        self.placements = []
        self.placement = None
        self._release_pose = None
        self._recovery_pose = None
        self.grasp_start_object_pose = self.grasp_start_object_in_eef = self.object_in_eef = None
        self.grasp_start_eef_pose = self._last_object_in_eef = None
        self._stable_grasp_steps = 0
        return True

    @staticmethod
    def _diversify_grasps(ranked: list[GraspCandidate]) -> list[GraspCandidate]:
        """Interleave pose-distinct approach families for useful retries."""
        distinct: list[GraspCandidate] = []
        for candidate in ranked:
            if any(
                np.linalg.norm(candidate.pose_env.position - prior.pose_env.position) < 0.008
                and quat_angle_wxyz(candidate.pose_env.quaternion, prior.pose_env.quaternion)
                < np.deg2rad(8.0)
                for prior in distinct
            ):
                continue
            distinct.append(candidate)
        groups: dict[tuple[float, ...], list[GraspCandidate]] = {}
        for candidate in distinct:
            key = tuple(np.round(np.concatenate((
                candidate.approach_axis_env, candidate.closing_axis_env
            )), 1))
            groups.setdefault(key, []).append(candidate)
        diversified: list[GraspCandidate] = []
        depth = 0
        while True:
            added = False
            for group in groups.values():
                if depth < len(group):
                    diversified.append(group[depth])
                    added = True
            if not added:
                return diversified
            depth += 1

    def pregrasp_pose(self) -> Pose:
        return self._env_to_robot(self._grasp().pregrasp_pose_env)

    def grasp_pose(self) -> Pose:
        return self._env_to_robot(self._grasp().pose_env)

    def verification_lift_pose(self) -> Pose:
        env_pose = self._robot_to_env(self.grasp_pose())
        return self._env_to_robot(Pose(env_pose.position + [0, 0, self.cfg.verification_lift], env_pose.quaternion))

    def lift_pose(self) -> Pose:
        if self._lift_pose is None:
            env_pose = self._robot_to_env(self.verification_lift_pose())
            z = env_pose.position[2] + self.cfg.transport_lift
            self._lift_pose = self._env_to_robot(
                Pose([env_pose.position[0], env_pose.position[1], z], env_pose.quaternion)
            )
        return self._lift_pose

    def source_clearance_pose(self) -> Pose:
        target = self._target()
        if target.source_name is None:
            return self.lift_pose()
        if self._clearance_pose is None:
            eef = self.current_eef_pose_env()
            obj = pose_from_world_state(self.world, target.object_name, self.env_id)
            required_z = source_clearance_height(
                self.world, target.object_name, target.source_name, self.env_id, self.cfg.source_clearance_margin
            )
            self._clearance_pose = self._env_to_robot(Pose(
                eef.position + [0, 0, max(0.0, required_z - obj.position[2])], eef.quaternion
            ))
        return self._clearance_pose

    def source_is_clear(self) -> bool:
        target = self._target()
        if target.source_name is None:
            return True
        source_corners, _ = self.world.get_bbox(target.source_name, env_id=self.env_id)
        object_corners, _ = self.world.get_bbox(target.object_name, env_id=self.env_id)
        return min(float(c[2]) for c in object_corners) >= (
            max(float(c[2]) for c in source_corners) + self.cfg.source_clearance_margin
        )

    def select_placement(self, attempt: int) -> bool:
        target = self._target()
        if self.object_in_eef is None:
            return False
        if not self.placements:
            profile = self.profiles.select(
                target.object_name, self._physical_dimensions(target.object_name),
                metadata=self._asset_metadata(target.object_name),
            )
            self.placements = self.placement_generator.generate(
                target, self.world, self.env_id, occupied_names=list(self.world.objects),
                object_dimensions=profile.dimensions_override,
            )
        while attempt < len(self.placements):
            candidate = self.placements[attempt]
            release_robot = self._env_to_robot(candidate.object_pose_env.compose(self.object_in_eef.inverse()))
            radius = float(np.linalg.norm(release_robot.position))
            if self.cfg.hard_minimum_reach <= radius <= self.cfg.hard_maximum_reach:
                self.placement, self._release_pose = candidate, release_robot
                return True
            attempt += 1
        return False

    def preplace_pose(self) -> Pose:
        release = self._robot_to_env(self.release_pose())
        z = release.position[2] + self.cfg.preplace_height
        target = self._target()
        if target.source_name:
            corners, _ = self.world.get_bbox(target.source_name, env_id=self.env_id)
            z = max(z, max(float(c[2]) for c in corners) + self.cfg.source_clearance_margin)
        return self._env_to_robot(Pose([release.position[0], release.position[1], z], release.quaternion))

    def release_pose(self) -> Pose:
        if self._release_pose is None:
            raise RuntimeError("Placement has not been selected")
        return self._release_pose

    def retreat_pose(self) -> Pose:
        pose = self._robot_to_env(self.release_pose())
        return self._env_to_robot(Pose(pose.position + [0, 0, self.cfg.retreat_height], pose.quaternion))

    def recovery_pose(self) -> Pose:
        if self._recovery_pose is None:
            current = self.current_eef_pose_env()
            # Vertical-only retreat from the achieved pose avoids retrying the
            # same unreachable x/y/orientation that triggered recovery.
            self._recovery_pose = self._env_to_robot(
                Pose(current.position + [0, 0, self.cfg.recovery_height], current.quaternion)
            )
        return self._recovery_pose

    def pose_reached(self, pose: Pose) -> bool:
        return self._pose_reached(pose, self.cfg.position_tolerance)

    def grasp_pose_reached(self, pose: Pose) -> bool:
        """Require fingertip-scale precision before beginning closure."""
        return self._pose_reached(pose, self.cfg.grasp_position_tolerance)

    def _pose_reached(self, pose: Pose, position_tolerance: float) -> bool:
        actual = self.current_eef_pose_robot()
        return (np.linalg.norm(actual.position - pose.position) <= position_tolerance
                and quat_angle_wxyz(actual.quaternion, pose.quaternion) <= self.cfg.orientation_tolerance_rad)

    def gripper_object_contact(self) -> bool:
        target = self._target()
        contact = bool(self.world.in_contact(target.object_name, "gripper", env_id=self.env_id))
        if contact and self.grasp_start_object_pose is None:
            self.grasp_start_object_pose = pose_from_world_state(self.world, target.object_name, self.env_id)
            eef = self.current_eef_pose_env()
            self.grasp_start_eef_pose = eef
            self.grasp_start_object_in_eef = self.grasp_start_object_pose.relative_to(eef)
            self._last_object_in_eef = self.grasp_start_object_in_eef
            self._stable_grasp_steps = 0
        return contact

    def verified_grasp(self) -> bool:
        if (
            not self.gripper_object_contact()
            or self.grasp_start_object_pose is None
            or self.grasp_start_eef_pose is None
            or self._last_object_in_eef is None
        ):
            return False
        obj = pose_from_world_state(self.world, self._target().object_name, self.env_id)
        eef = self.current_eef_pose_env()
        relative = obj.relative_to(eef)
        incrementally_stable = (
            np.linalg.norm(relative.position - self._last_object_in_eef.position)
            <= self.cfg.grasp_stability_position_tolerance
            and quat_angle_wxyz(
                relative.quaternion, self._last_object_in_eef.quaternion
            ) <= self.cfg.grasp_stability_orientation_tolerance_rad
        )
        self._stable_grasp_steps = (
            self._stable_grasp_steps + 1 if incrementally_stable else 0
        )
        self._last_object_in_eef = relative
        object_motion = float(np.linalg.norm(
            obj.position - self.grasp_start_object_pose.position
        ))
        eef_motion = float(np.linalg.norm(
            eef.position - self.grasp_start_eef_pose.position
        ))
        follow_ratio = object_motion / max(eef_motion, 1e-9)
        if (
            object_motion >= self.cfg.minimum_verification_motion
            and eef_motion >= self.cfg.minimum_verification_motion
            and follow_ratio >= self.cfg.minimum_follow_ratio
            and self._stable_grasp_steps >= self.cfg.grasp_stability_steps
        ):
            self.object_in_eef = relative
            return True
        return False

    def placement_verified(self) -> bool:
        target = self._target()
        velocity = self.world.get_velocity(target.object_name, env_id=self.env_id).detach().cpu().numpy()
        stationary = (np.linalg.norm(velocity[:3]) <= self.cfg.stationary_linear_speed
                      and np.linalg.norm(velocity[3:]) <= self.cfg.stationary_angular_speed)
        detached = not bool(self.world.in_contact(target.object_name, "gripper", env_id=self.env_id))
        return stationary and detached and self._verify_relation(target)

    def task_success(self) -> bool:
        return self.latest_success

    def _verify_relation(self, target: ManipulationTarget) -> bool:
        default_name = {
            "inside": "object_in_container",
            "on_surface": "object_on_top",
            "outside_container_and_on_surface": "object_outside_of_and_on_surface",
        }[target.destination_relation]
        predicate = _PREDICATES.get(target.verification_predicate or default_name)
        if predicate is None:
            return False
        params = dict(target.verification_params)
        params["object"], params["logical"] = target.object_name, "all"
        params.pop("K", None)
        if target.destination_relation == "inside":
            params.setdefault("container", target.destination_name)
        elif target.destination_relation == "on_surface":
            params.setdefault("reference_object", target.destination_name)
        else:
            params.setdefault("container", target.source_name)
            params.setdefault("surface", target.destination_name)
        return bool(predicate(self.env, env_id=self.env_id, **params))

    def _asset_metadata(self, object_name: str) -> dict[str, Any]:
        spawn = getattr(getattr(self.world.get_body(object_name), "cfg", None), "spawn", None)
        path = getattr(spawn, "usd_path", None)
        return {"usd_path": path} if path else {}

    def _physical_dimensions(self, object_name: str) -> np.ndarray:
        try:
            return oriented_box_from_world(self.world, object_name, self.env_id).dimensions
        except ValueError:
            return np.asarray(self.world.get_dimensions(object_name), dtype=np.float64)

    def _target(self) -> ManipulationTarget:
        if self.target is None:
            raise RuntimeError("No target selected")
        return self.target

    def _grasp(self) -> GraspCandidate:
        if self.grasp is None:
            raise RuntimeError("No grasp selected")
        return self.grasp

    def _env_to_robot(self, pose: Pose) -> Pose:
        return env_pose_to_robot(pose, robot_root_pose_env(self.world, self.env_id))

    def _robot_to_env(self, pose: Pose) -> Pose:
        return robot_pose_to_env(pose, robot_root_pose_env(self.world, self.env_id))


class ScriptedExpertPolicy:
    """Batched action producer for RoboLab's ``DroidIKActionCfg``."""

    def __init__(self, env, spec, *, profiles=None, cfg=None) -> None:
        self.env, self.spec = env, spec
        self.cfg = cfg or ScriptedPolicyConfig()
        self.world = get_world(env)
        spec.validate_entities(self.world.entities)
        self.profiles = profiles or GraspProfileLibrary()
        grasp_generator = GeometryGraspCandidateGenerator(self.cfg.grasp_generator)
        placement_generator = GeometryPlacementGenerator(self.cfg.placement_generator)
        self.contexts = [_EnvironmentContext(env, i, self.cfg, self.profiles, grasp_generator, placement_generator)
                         for i in range(env.num_envs)]
        self.machines = [ScriptedManipulationStateMachine(spec, self.cfg.state_machine) for _ in range(env.num_envs)]
        self._last_commanded_eef: list[Pose | None] = [None] * env.num_envs
        self._last_actual_eef: list[Pose | None] = [None] * env.num_envs
        self._base_to_eef = Pose(np.asarray(EEF_OFFSET_POS), np.asarray(EEF_OFFSET_ROT))

    def reset(self) -> None:
        for machine in self.machines:
            machine.reset()
        for context in self.contexts:
            context.latest_success = False
            context.grasp_attempt_history.clear()
        self._last_commanded_eef = [None] * self.env.num_envs
        self._last_actual_eef = [None] * self.env.num_envs

    def act(self) -> torch.Tensor:
        """Return ``[x,y,z,qw,qx,qy,qz,gripper]`` in robot-root frame."""
        actions = torch.zeros((self.env.num_envs, 8), device=self.env.device, dtype=torch.float32)
        frozen = getattr(self.env, "_frozen_envs", None)
        for env_id, (machine, context) in enumerate(zip(self.machines, self.contexts)):
            if frozen is not None and bool(frozen[env_id]):
                continue
            actual_eef = context.current_eef_pose_robot()
            safety_failure = self._eef_safety_failure(env_id, actual_eef)
            if safety_failure is not None and not machine.done:
                machine.fail(FailureReason.IK_DIVERGENCE, safety_failure)
            self._last_actual_eef[env_id] = actual_eef
            command = machine.step(context)
            limited_eef = self._rate_limit(
                env_id, command.pose_robot or actual_eef, actual_eef
            )
            target_base_robot = limited_eef.compose(self._base_to_eef.inverse())
            actions[env_id, :3] = torch.as_tensor(target_base_robot.position, device=self.env.device)
            actions[env_id, 3:7] = torch.as_tensor(target_base_robot.quaternion, device=self.env.device)
            actions[env_id, 7] = float(command.gripper_command)
        return actions

    def observe_step(self, terminated, truncated, info=None) -> None:
        del truncated, info
        for env_id, context in enumerate(self.contexts):
            context.latest_success = bool(terminated[env_id])

    def failed_env_ids(self) -> list[int]:
        return [index for index, machine in enumerate(self.machines) if machine.state == ScriptedState.FAILED]

    def result(self, env_id: int) -> dict[str, Any]:
        machine = self.machines[env_id]
        context = self.contexts[env_id]
        return {
            "env_id": env_id,
            "state": machine.state.value,
            "success": machine.state == ScriptedState.SUCCESS,
            "failure_reason": machine.failure_reason.value if machine.failure_reason else None,
            "failure_detail": machine.failure_detail,
            "grasp_attempts": machine.grasp_attempt + int(machine.grasp_attempt == 0 and machine.current_target is not None),
            "completed_targets": len(machine.completed_indices),
            "grasp_candidates": context.grasp_attempt_history,
        }

    def _eef_safety_failure(self, env_id: int, actual: Pose) -> str | None:
        previous_actual = self._last_actual_eef[env_id]
        if previous_actual is not None:
            step_distance = float(np.linalg.norm(actual.position - previous_actual.position))
            step_angle = quat_angle_wxyz(actual.quaternion, previous_actual.quaternion)
            if (step_distance > self.cfg.maximum_eef_step_distance
                    or step_angle > self.cfg.maximum_eef_step_rotation_rad):
                return (
                    f"measured EEF discontinuity: translation={step_distance:.3f}m "
                    f"rotation={np.rad2deg(step_angle):.1f}deg"
                )
        commanded = self._last_commanded_eef[env_id]
        if commanded is not None:
            position_error = float(np.linalg.norm(actual.position - commanded.position))
            orientation_error = quat_angle_wxyz(actual.quaternion, commanded.quaternion)
            if (position_error > self.cfg.maximum_tracking_position_error
                    or orientation_error > self.cfg.maximum_tracking_orientation_error_rad):
                return (
                    f"absolute-IK tracking diverged: position_error={position_error:.3f}m "
                    f"orientation_error={np.rad2deg(orientation_error):.1f}deg"
                )
        return None

    def _rate_limit(self, env_id: int, desired: Pose, actual: Pose) -> Pose:
        # Anchor every incremental command to feedback. A stale commanded pose can
        # otherwise keep walking the local DLS solver through a singularity.
        previous = actual
        delta = desired.position - previous.position
        distance = float(np.linalg.norm(delta))
        pos_fraction = min(1.0, self.cfg.max_translation_per_step / max(distance, 1e-9))
        angle = quat_angle_wxyz(previous.quaternion, desired.quaternion)
        rot_fraction = min(1.0, self.cfg.max_rotation_per_step_rad / max(angle, 1e-9))
        fraction = min(pos_fraction, rot_fraction)
        limited = Pose(previous.position + fraction * delta,
                       quat_slerp_wxyz(previous.quaternion, desired.quaternion, fraction))
        self._last_commanded_eef[env_id] = limited
        return limited

