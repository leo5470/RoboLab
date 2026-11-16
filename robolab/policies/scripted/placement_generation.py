"""Relation-specific, geometry-aware deterministic placement generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import (
    Pose,
    matrix_to_quat_wxyz,
    oriented_box_from_world,
    projected_half_extents,
    pose_from_world_state,
)
from .task_spec import ManipulationTarget


@dataclass(frozen=True)
class PlacementCandidate:
    """Desired object pose; the policy converts it using the verified grasp transform."""

    object_pose_env: Pose
    relation: str
    score: float
    label: str


@dataclass(frozen=True)
class PlacementGeneratorConfig:
    surface_edge_margin: float = 0.002
    occupied_margin: float = 0.005
    occupied_vertical_margin: float = 0.002
    container_wall_margin: float = 0.025
    release_clearance: float = 0.008
    container_bottom_clearance: float = 0.025
    grid_fraction: float = 0.45


class GeometryPlacementGenerator:
    def __init__(self, cfg: PlacementGeneratorConfig | None = None) -> None:
        self.cfg = cfg or PlacementGeneratorConfig()

    def generate(
        self,
        target: ManipulationTarget,
        world,
        env_id: int,
        *,
        occupied_names: list[str] | None = None,
        object_dimensions: np.ndarray | tuple[float, float, float] | None = None,
    ) -> list[PlacementCandidate]:
        if target.destination_relation == "inside":
            return self._container_candidates(target, world, env_id, object_dimensions)
        if target.destination_relation in {"on_surface", "outside_container_and_on_surface"}:
            return self._surface_candidates(target, world, env_id, occupied_names or [], object_dimensions)
        return []

    def _surface_candidates(self, target, world, env_id, occupied_names, object_dimensions) -> list[PlacementCandidate]:
        destination_pose, dest_lower, dest_upper = self._surface_geometry(
            world, target.destination_name, env_id
        )
        object_pose = pose_from_world_state(world, target.object_name, env_id)
        try:
            object_box = oriented_box_from_world(world, target.object_name, env_id)
        except ValueError:
            object_box = None
        if object_dimensions is not None:
            dimensions = np.asarray(object_dimensions, dtype=np.float64)
            geometry_rotation = object_pose.rotation
        elif object_box is not None:
            dimensions, geometry_rotation = object_box.dimensions, object_box.axes_env
        else:
            dimensions = np.asarray(world.get_dimensions(target.object_name), dtype=np.float64)
            geometry_rotation = object_pose.rotation

        orientations = self._surface_orientations(
            object_pose, geometry_rotation, dimensions, destination_pose
        )
        center_xy = (dest_lower[:2] + dest_upper[:2]) / 2.0
        grid_values = (0.0, -0.5, 0.5, -1.0, 1.0)
        offsets = [(x, y) for x in grid_values for y in grid_values]
        candidates: list[PlacementCandidate] = []
        for orientation_index, (root_rotation, target_geometry_rotation, label) in enumerate(orientations):
            relative_rotation = destination_pose.rotation.T @ target_geometry_rotation
            half = projected_half_extents(relative_rotation, dimensions)
            available = (
                (dest_upper[:2] - dest_lower[:2]) / 2.0
                - half[:2]
                - self.cfg.surface_edge_margin
            )
            if np.any(available <= 0):
                continue
            for grid_index, (ox, oy) in enumerate(offsets):
                center_local = np.array([
                    center_xy[0] + ox * available[0],
                    center_xy[1] + oy * available[1],
                    dest_upper[2] + half[2] + self.cfg.release_clearance,
                ])
                center_env = destination_pose.transform_point(center_local)
                if self._occupied(
                    center_env, half, destination_pose, world, env_id,
                    occupied_names, target,
                ):
                    continue
                root_position = self._root_from_center(
                    center_env, root_rotation, object_pose, target.object_name, world, env_id
                )
                candidates.append(PlacementCandidate(
                    object_pose_env=Pose(root_position, matrix_to_quat_wxyz(root_rotation)),
                    relation=target.destination_relation,
                    score=1.0 - orientation_index * 0.1 - grid_index * 0.002,
                    label=f"surface-{label}-grid-{grid_index}",
                ))
        return candidates

    @staticmethod
    def _surface_orientations(object_pose, geometry_rotation, dimensions, destination_pose):
        """Enumerate current and axis-aligned support orientations deterministically."""
        geometry_in_root = object_pose.rotation.T @ geometry_rotation
        options = [(object_pose.rotation, geometry_rotation, "current")]
        aligned = []
        for up_idx in range(3):
            remaining = [axis for axis in range(3) if axis != up_idx]
            support_area = float(dimensions[remaining[0]] * dimensions[remaining[1]])
            for quarter_turn in (0, 1):
                target_geometry = np.zeros((3, 3), dtype=np.float64)
                horizontal = destination_pose.rotation[:, :2]
                target_geometry[:, up_idx] = destination_pose.rotation[:, 2]
                target_geometry[:, remaining[0]] = horizontal[:, quarter_turn]
                target_geometry[:, remaining[1]] = horizontal[:, 1 - quarter_turn]
                if np.linalg.det(target_geometry) < 0:
                    target_geometry[:, remaining[1]] *= -1.0
                target_root = target_geometry @ geometry_in_root.T
                aligned.append((
                    -support_area,
                    up_idx,
                    quarter_turn,
                    target_root,
                    target_geometry,
                    f"axis-{up_idx}-yaw-{quarter_turn}",
                ))
        for _, _, _, root_rotation, target_geometry, label in sorted(aligned):
            if any(np.allclose(target_geometry, existing[1], atol=1e-5) for existing in options):
                continue
            options.append((root_rotation, target_geometry, label))
        return options

    @staticmethod
    def _surface_geometry(world, name: str, env_id: int):
        """Build a horizontal support frame from the destination physical OBB."""
        try:
            box = oriented_box_from_world(world, name, env_id)
        except ValueError:
            pose = pose_from_world_state(world, name, env_id)
            lower, upper = GeometryPlacementGenerator._local_bounds(world, name, env_id)
            return pose, lower, upper
        world_up = np.array([0.0, 0.0, 1.0])
        up_idx = int(np.argmax(np.abs(box.axes_env.T @ world_up)))
        horizontal_indices = [axis for axis in range(3) if axis != up_idx]
        x_idx = max(
            horizontal_indices,
            key=lambda axis: abs(float(np.dot(box.axes_env[:, axis], [1.0, 0.0, 0.0]))),
        )
        y_idx = next(axis for axis in horizontal_indices if axis != x_idx)
        z_axis = box.axes_env[:, up_idx].copy()
        if np.dot(z_axis, world_up) < 0:
            z_axis *= -1.0
        x_axis = box.axes_env[:, x_idx].copy()
        if x_axis[0] < 0:
            x_axis *= -1.0
        y_axis = np.cross(z_axis, x_axis)
        axes = np.column_stack((x_axis, y_axis, z_axis))
        dimensions = np.array([
            box.dimensions[x_idx], box.dimensions[y_idx], box.dimensions[up_idx]
        ])
        pose = Pose(box.center_env, matrix_to_quat_wxyz(axes))
        return pose, -dimensions / 2.0, dimensions / 2.0

    def _container_candidates(self, target, world, env_id, object_dimensions) -> list[PlacementCandidate]:
        container_pose = pose_from_world_state(world, target.destination_name, env_id)
        object_pose = pose_from_world_state(world, target.object_name, env_id)
        lower, upper = self._local_bounds(world, target.destination_name, env_id)
        if object_dimensions is not None:
            dimensions = np.asarray(object_dimensions, dtype=np.float64)
            geometry_rotation = object_pose.rotation
        else:
            try:
                object_box = oriented_box_from_world(world, target.object_name, env_id)
                dimensions, geometry_rotation = object_box.dimensions, object_box.axes_env
            except ValueError:
                dimensions = np.asarray(world.get_dimensions(target.object_name), dtype=np.float64)
                geometry_rotation = object_pose.rotation
        inner_half_xy = (upper[:2] - lower[:2]) / 2.0 - self.cfg.container_wall_margin
        center = (lower + upper) / 2.0
        offsets = [(0.0, 0.0), (-0.35, 0.0), (0.35, 0.0), (0.0, -0.35), (0.0, 0.35)]
        candidates = []
        orientations = self._surface_orientations(
            object_pose, geometry_rotation, dimensions, container_pose
        )
        for orientation_index, (root_rotation, target_geometry_rotation, label) in enumerate(orientations):
            relative_rotation = container_pose.rotation.T @ target_geometry_rotation
            half = projected_half_extents(relative_rotation, dimensions)
            available = inner_half_xy - half[:2]
            if np.any(available < 0):
                continue
            for grid_index, (ox, oy) in enumerate(offsets):
                center_local = np.array([
                    center[0] + ox * available[0],
                    center[1] + oy * available[1],
                    lower[2] + half[2] + self.cfg.container_bottom_clearance,
                ])
                center_env = container_pose.transform_point(center_local)
                root_position = self._root_from_center(
                    center_env, root_rotation, object_pose, target.object_name, world, env_id
                )
                candidates.append(PlacementCandidate(
                    object_pose_env=Pose(root_position, matrix_to_quat_wxyz(root_rotation)),
                    relation="inside",
                    score=1.0 - orientation_index * 0.1 - grid_index * 0.01,
                    label=f"container-{label}-grid-{grid_index}",
                ))
        return candidates

    @staticmethod
    def _local_bounds(world, name: str, env_id: int) -> tuple[np.ndarray, np.ndarray]:
        pose = pose_from_world_state(world, name, env_id)
        corners, _ = world.get_bbox(name, env_id=env_id)
        local = np.asarray([pose.rotation.T @ (np.asarray(corner) - pose.position) for corner in corners])
        return local.min(axis=0), local.max(axis=0)

    def _occupied(self, center_env, half_dest, dest_pose, world, env_id, occupied_names, target) -> bool:
        center_dest = dest_pose.rotation.T @ (center_env - dest_pose.position)
        for name in occupied_names:
            if name in {target.object_name, target.destination_name, target.source_name}:
                continue
            try:
                box = oriented_box_from_world(world, name, env_id)
            except (ValueError, KeyError):
                continue
            other_center_dest = dest_pose.rotation.T @ (box.center_env - dest_pose.position)
            other_half = projected_half_extents(
                dest_pose.rotation.T @ box.axes_env, box.dimensions
            )
            margin = np.array([
                self.cfg.occupied_margin,
                self.cfg.occupied_margin,
                self.cfg.occupied_vertical_margin,
            ])
            if np.all(np.abs(center_dest - other_center_dest) < half_dest + other_half + margin):
                return True
        return False

    @staticmethod
    def _root_from_center(
        center_env, target_root_rotation, current_object_pose, object_name, world, env_id
    ) -> np.ndarray:
        if hasattr(world, "get_bbox"):
            _, current_center = world.get_bbox(object_name, env_id=env_id)
            local_center = current_object_pose.rotation.T @ (
                np.asarray(current_center) - current_object_pose.position
            )
            return np.asarray(center_env) - target_root_rotation @ local_center
        return np.asarray(center_env)


def source_clearance_height(world, object_name: str, source_name: str, env_id: int, margin: float = 0.03) -> float:
    """Minimum env-frame root z that puts the complete object OBB above a source rim."""
    source_corners, _ = world.get_bbox(source_name, env_id=env_id)
    object_corners, _ = world.get_bbox(object_name, env_id=env_id)
    source_top = max(float(corner[2]) for corner in source_corners)
    object_bottom = min(float(corner[2]) for corner in object_corners)
    object_pose = pose_from_world_state(world, object_name, env_id)
    return float(object_pose.position[2] + (source_top + margin - object_bottom))

