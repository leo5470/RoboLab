"""Deterministic geometry-based parallel-jaw grasp candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .geometry import (
    Pose,
    bbox_min_max,
    make_tool_orientation,
    normalize,
    oriented_box_from_world,
    pose_from_world_state,
)
from .profiles import GraspProfile, profile_from_geometry


@dataclass(frozen=True)
class GraspCandidate:
    """EEF-frame targets and metadata for one parallel-jaw grasp."""

    grasp_type: str
    pose_env: Pose
    pregrasp_pose_env: Pose
    contact_point_object: np.ndarray
    approach_axis_env: np.ndarray
    closing_axis_env: np.ndarray
    required_opening: float
    score: float
    label: str


class GraspCandidateGenerator(Protocol):
    def generate(
        self,
        object_name: str,
        world,
        profile: GraspProfile | None,
        env_id: int,
    ) -> list[GraspCandidate]: ...


@dataclass(frozen=True)
class GraspGeneratorConfig:
    max_aperture: float = 0.085
    aperture_margin: float = 0.002
    # Robotiq 2F-85 base_link to the leading fingertip face in the checked-in
    # DROID USD (the pad occupies x=0.1119..0.1500 m in base_link).
    tool_reach: float = 0.1500
    wrist_radius: float = 0.045
    obstacle_margin: float = 0.015
    edge_offset_fraction: float = 0.28
    tilted_approach_downward_fraction: float = 0.45


class GeometryGraspCandidateGenerator:
    """Generate top, side, and thickness/edge grasps from local object geometry."""

    def __init__(self, cfg: GraspGeneratorConfig | None = None) -> None:
        self.cfg = cfg or GraspGeneratorConfig()

    def generate(
        self,
        object_name: str,
        world,
        profile: GraspProfile | None,
        env_id: int,
        *,
        obstacle_names: list[str] | None = None,
        source_name: str | None = None,
    ) -> list[GraspCandidate]:
        physical_box = None
        if hasattr(world, "get_bbox"):
            try:
                physical_box = oriented_box_from_world(world, object_name, env_id)
            except ValueError:
                physical_box = None
        profile_dimensions = (
            physical_box.dimensions
            if physical_box is not None
            else np.asarray(world.get_dimensions(object_name), dtype=np.float64)
        )
        profile = profile or profile_from_geometry(profile_dimensions, max_aperture=self.cfg.max_aperture)
        rigid_object_pose_env = pose_from_world_state(world, object_name, env_id)
        use_obb_frame = profile.geometry_frame == "obb" and physical_box is not None
        if use_obb_frame:
            object_pose_env = physical_box.pose_env
            center_local = np.zeros(3)
            dimensions = np.asarray(profile.dimensions_override or physical_box.dimensions, dtype=np.float64)
        else:
            if profile.geometry_position_object is not None or profile.geometry_quaternion_object is not None:
                geometry_pose_object = Pose(
                    profile.geometry_position_object or (0.0, 0.0, 0.0),
                    profile.geometry_quaternion_object or (1.0, 0.0, 0.0, 0.0),
                )
                object_pose_env = rigid_object_pose_env.compose(geometry_pose_object)
                center_local = np.zeros(3)
            else:
                object_pose_env = rigid_object_pose_env
                center_local = self._local_center(world, object_name, env_id)
            dimensions = np.asarray(
                profile.dimensions_override or world.get_dimensions(object_name), dtype=np.float64
            )
        grasp_dimensions = np.asarray(
            profile.grasp_dimensions_override or dimensions, dtype=np.float64
        )
        if not profile.allowed_grasp_types:
            return []
        local_axes = np.eye(3)
        short_idx = int(np.argmin(grasp_dimensions))
        long_idx = int(np.argmax(dimensions))
        world_down_object = object_pose_env.rotation.T @ np.array([0.0, 0.0, -1.0])

        raw: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str]] = []
        if "top" in profile.allowed_grasp_types:
            approach_object = object_pose_env.rotation.T @ np.array([0.0, 0.0, -1.0])
            closing_choices = sorted(
                (i for i in range(3) if abs(float(np.dot(local_axes[i], approach_object))) < 0.75),
                key=lambda i: (grasp_dimensions[i], i),
            )
            for close_idx in closing_choices:
                raw.append(("top", local_axes[close_idx], approach_object, -approach_object,
                            center_local, 0.0, f"top-close-{close_idx}"))

        if "side" in profile.allowed_grasp_types:
            for approach_idx in range(3):
                remaining = [i for i in range(3) if i != approach_idx]
                close_idx = min(remaining, key=lambda i: (grasp_dimensions[i], i))
                for sign in (-1.0, 1.0):
                    side_axis = sign * local_axes[approach_idx]
                    # A parallel-jaw approach must stay perpendicular to its
                    # closing axis.  Tilting directly toward world-down skewed
                    # thin-object jaws so one finger struck the top face long
                    # before the other could reach the underside.
                    down_perpendicular = (
                        world_down_object
                        - np.dot(world_down_object, local_axes[close_idx])
                        * local_axes[close_idx]
                    )
                    tilted = normalize(
                        side_axis
                        + self.cfg.tilted_approach_downward_fraction * down_perpendicular
                    )
                    raw.append(("side", local_axes[close_idx], tilted, -side_axis, center_local,
                                0.35 - 0.1 * dimensions[approach_idx],
                                f"side-tilted-{approach_idx}-{int(sign):+d}"))
                    raw.append(("side", local_axes[close_idx], side_axis, -side_axis, center_local,
                                -0.1 * dimensions[approach_idx],
                                f"side-{approach_idx}-{int(sign):+d}"))

        if "edge" in profile.allowed_grasp_types:
            approach_indices = [i for i in range(3) if i != short_idx]
            offsets = [0.0]
            if long_idx != short_idx:
                offsets.extend([-self.cfg.edge_offset_fraction * dimensions[long_idx],
                                self.cfg.edge_offset_fraction * dimensions[long_idx]])
            for approach_idx in approach_indices:
                for sign in (-1.0, 1.0):
                    for offset in offsets:
                        point = center_local.copy()
                        point[long_idx] += offset
                        side_axis = sign * local_axes[approach_idx]
                        down_perpendicular = (
                            world_down_object
                            - np.dot(world_down_object, local_axes[short_idx])
                            * local_axes[short_idx]
                        )
                        tilted = normalize(
                            side_axis
                            + self.cfg.tilted_approach_downward_fraction * down_perpendicular
                        )
                        contact_span_idx = next(
                            i for i in range(3) if i not in {short_idx, approach_idx}
                        )
                        contact_span = min(float(dimensions[contact_span_idx]), 0.25)
                        base_score = 1.0 + 2.0 * contact_span - abs(offset) / max(dimensions[long_idx], 1e-9)
                        raw.append((
                            "edge", local_axes[short_idx], tilted, -side_axis, point, base_score + 0.35,
                            f"edge-tilted-{approach_idx}-{int(sign):+d}-offset-{offset:+.3f}",
                        ))
                        raw.append((
                            "edge", local_axes[short_idx], side_axis, -side_axis, point, base_score,
                            f"edge-{approach_idx}-{int(sign):+d}-offset-{offset:+.3f}",
                        ))

        base_candidates = tuple(raw)
        for offset_index, offset in enumerate(profile.candidate_offsets):
            if np.linalg.norm(offset) == 0:
                continue
            for grasp_type, close, approach, surface_normal, point, base_score, label in base_candidates:
                raw.append((grasp_type, close, approach, surface_normal,
                            point + np.asarray(offset), base_score - 0.1,
                            f"{label}-profile-{offset_index}"))

        candidates: list[GraspCandidate] = []
        for grasp_type, close_object, approach_object, surface_normal_object, point_object, score, label in raw:
            close_object = normalize(close_object)
            approach_object = normalize(approach_object)
            approach_env = normalize(object_pose_env.rotation @ approach_object)
            close_env_raw = normalize(object_pose_env.rotation @ close_object)
            close_env = normalize(close_env_raw - np.dot(close_env_raw, approach_env) * approach_env)
            effective_close_object = object_pose_env.rotation.T @ close_env
            opening = float(np.sum(np.abs(effective_close_object) * grasp_dimensions)) + self.cfg.aperture_margin
            if opening > self.cfg.max_aperture:
                continue
            half_dimensions = 0.5 * dimensions
            surface_normal_object = normalize(surface_normal_object)
            surface_axis = int(np.argmax(np.abs(surface_normal_object)))
            surface_point_object = point_object.copy()
            surface_point_object[surface_axis] = (
                center_local[surface_axis]
                + np.sign(surface_normal_object[surface_axis]) * half_dimensions[surface_axis]
            )
            contact_env = object_pose_env.transform_point(surface_point_object)
            wrist_distance = max(0.0, self.cfg.tool_reach - profile.grasp_depth)
            grasp_position = contact_env - approach_env * wrist_distance
            if source_name is not None:
                source_corners, _ = world.get_bbox(source_name, env_id=env_id)
                source_bottom = min(float(corner[2]) for corner in source_corners)
                if grasp_position[2] < source_bottom + self.cfg.wrist_radius:
                    continue
            quaternion = make_tool_orientation(close_env, approach_env)
            pose_env = Pose(grasp_position, quaternion)
            # Keep the fingertips outside the object by approach_distance. The
            # pregrasp must not move inward for a deeper final pad overlap.
            pregrasp_position = contact_env - approach_env * (
                self.cfg.tool_reach + profile.approach_distance
            )
            if source_name is not None:
                object_corners, _ = world.get_bbox(object_name, env_id=env_id)
                source_top = max(float(corner[2]) for corner in source_corners)
                object_top = max(float(corner[2]) for corner in object_corners)
                vertical_tool_extent = max(0.0, -float(approach_env[2])) * self.cfg.tool_reach
                safe_pregrasp_z = (
                    max(source_top, object_top)
                    + vertical_tool_extent
                    + self.cfg.wrist_radius
                    + self.cfg.obstacle_margin
                )
                if safe_pregrasp_z > pregrasp_position[2]:
                    # Once the complete tool is staged above the source, a
                    # second horizontal approach-distance offset only hurts
                    # reachability. Descend from directly above the final wrist.
                    pregrasp_position[:2] = grasp_position[:2]
                    pregrasp_position[2] = safe_pregrasp_z
            pregrasp_pose_env = Pose(pregrasp_position, quaternion)
            if self._path_blocked(pregrasp_pose_env.position, pose_env.position, world, env_id, obstacle_names or []):
                continue
            axis_preference = 0.2 if "short" in profile.preferred_local_axes and np.argmax(np.abs(close_object)) == short_idx else 0.0
            score += profile.score_biases.get(grasp_type, 0.0) + axis_preference
            score -= 2.0 * float(np.linalg.norm(point_object - center_local))
            candidates.append(GraspCandidate(
                grasp_type=grasp_type,
                pose_env=pose_env,
                pregrasp_pose_env=pregrasp_pose_env,
                contact_point_object=rigid_object_pose_env.inverse().transform_point(contact_env),
                approach_axis_env=approach_env,
                closing_axis_env=close_env,
                required_opening=opening,
                score=score,
                label=label,
            ))
        return sorted(candidates, key=lambda item: (-item.score, item.label))

    @staticmethod
    def _local_center(world, object_name: str, env_id: int = 0) -> np.ndarray:
        # Infer the rigid-root-local center from the live OBB; composed USD assets
        # may have an authored geometry translation that makes AABB midpoints unsafe.
        if hasattr(world, "get_bbox"):
            _, centroid_env = world.get_bbox(object_name, env_id=env_id)
            object_pose_env = pose_from_world_state(world, object_name, env_id)
            return object_pose_env.rotation.T @ (np.asarray(centroid_env) - object_pose_env.position)
        if hasattr(world, "get_local_geometric_center"):
            return np.asarray(world.get_local_geometric_center(object_name), dtype=np.float64)
        lower, upper = world.get_aabb(object_name)
        return (np.asarray(lower) + np.asarray(upper)) / 2.0

    def _path_blocked(self, start, end, world, env_id: int, obstacles: list[str]) -> bool:
        """Reject straight approaches intersecting conservative inflated obstacle AABBs."""
        if not obstacles:
            return False
        samples = np.linspace(np.asarray(start), np.asarray(end), 9)
        inflation = self.cfg.wrist_radius + self.cfg.obstacle_margin
        for obstacle in obstacles:
            corners, _ = world.get_bbox(obstacle, env_id=env_id)
            lower, upper = bbox_min_max(corners)
            if np.any(np.all((samples >= lower - inflation) & (samples <= upper + inflation), axis=1)):
                return True
        return False

