"""Semantic and geometry-derived parallel-jaw grasp profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import json
import numpy as np
import yaml


@dataclass(frozen=True)
class GraspProfile:
    allowed_grasp_types: tuple[str, ...] = ("top", "side", "edge")
    preferred_local_axes: tuple[str, ...] = ()
    forbidden_local_axes: tuple[str, ...] = ()
    candidate_offsets: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)
    approach_distance: float = 0.10
    grasp_depth: float = 0.02
    score_biases: dict[str, float] = field(default_factory=dict)
    geometry_class: str | None = None
    geometry_frame: str = "object"
    geometry_position_object: tuple[float, float, float] | None = None
    geometry_quaternion_object: tuple[float, float, float, float] | None = None
    dimensions_override: tuple[float, float, float] | None = None
    grasp_dimensions_override: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.geometry_frame not in {"object", "obb"}:
            raise ValueError("geometry_frame must be 'object' or 'obb'")
        for field_name in ("dimensions_override", "grasp_dimensions_override"):
            value = getattr(self, field_name)
            if value is not None and (len(value) != 3 or any(length <= 0 for length in value)):
                raise ValueError(f"{field_name} must contain three positive lengths")
        if self.geometry_position_object is not None and len(self.geometry_position_object) != 3:
            raise ValueError("geometry_position_object must contain three values")
        if self.geometry_quaternion_object is not None and len(self.geometry_quaternion_object) != 4:
            raise ValueError("geometry_quaternion_object must be a wxyz quaternion")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GraspProfile":
        values = dict(data)
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown grasp profile fields: {sorted(unknown)}")
        for key in ("allowed_grasp_types", "preferred_local_axes", "forbidden_local_axes"):
            if key in values:
                values[key] = tuple(values[key])
        if "candidate_offsets" in values:
            values["candidate_offsets"] = tuple(tuple(float(v) for v in item) for item in values["candidate_offsets"])
        for key in (
            "dimensions_override", "grasp_dimensions_override",
            "geometry_position_object", "geometry_quaternion_object",
        ):
            if values.get(key) is not None:
                values[key] = tuple(float(v) for v in values[key])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_geometry(dimensions, *, max_aperture: float = 0.085) -> str:
    """Classify only coarse shape; no semantic-part claims are made."""
    dims = np.sort(np.asarray(dimensions, dtype=np.float64))
    if dims.shape != (3,) or np.any(dims <= 0):
        raise ValueError(f"Expected three positive object dimensions, got {dimensions}")
    small, middle, large = dims
    if small > max_aperture:
        return "too_large"
    if small / large <= 0.18 and middle / large >= 0.25:
        return "thin_plate"
    if large / max(middle, 1e-9) >= 2.5:
        return "elongated"
    if abs(middle - large) / large < 0.18 and small / large >= 0.65:
        return "approximately_cylindrical_or_round"
    if large <= max_aperture:
        return "small_graspable"
    return "box_like"


def profile_from_geometry(dimensions, *, max_aperture: float = 0.085) -> GraspProfile:
    geometry_class = classify_geometry(dimensions, max_aperture=max_aperture)
    if geometry_class == "thin_plate":
        return GraspProfile(
            allowed_grasp_types=("edge", "side"),
            preferred_local_axes=("short",),
            candidate_offsets=((0.0, 0.0, 0.0),),
            approach_distance=0.12,
            grasp_depth=0.015,
            score_biases={"edge": 2.0, "side": 0.5},
            geometry_class=geometry_class,
            geometry_frame="obb",
        )
    if geometry_class == "elongated":
        return GraspProfile(
            allowed_grasp_types=("top", "side"),
            preferred_local_axes=("short",),
            candidate_offsets=((0.0, 0.0, 0.0),),
            score_biases={"top": 1.0},
            geometry_class=geometry_class,
            geometry_frame="obb",
        )
    if geometry_class == "too_large":
        return GraspProfile(allowed_grasp_types=(), geometry_class=geometry_class, geometry_frame="obb")
    return GraspProfile(score_biases={"top": 2.0}, geometry_class=geometry_class, geometry_frame="obb")


class GraspProfileLibrary:
    """Optional exact/category/asset profile data with geometry fallback."""

    def __init__(self, profiles: Mapping[str, GraspProfile] | None = None) -> None:
        self.profiles = dict(profiles or {})
        self.generated: dict[str, GraspProfile] = {}

    @classmethod
    def load(cls, path: str | Path | None) -> "GraspProfileLibrary":
        paths = [Path(__file__).with_name("default_grasp_profiles.yaml")]
        if path is not None:
            paths.append(Path(path))
        profiles: dict[str, GraspProfile] = {}
        for current_path in paths:
            with current_path.open("r", encoding="utf-8") as stream:
                data = (
                    json.load(stream)
                    if current_path.suffix.lower() == ".json"
                    else yaml.safe_load(stream)
                )
            if not isinstance(data, Mapping):
                raise ValueError("Grasp-profile file root must be a mapping")
            raw_profiles = data.get("profiles", data)
            if not isinstance(raw_profiles, Mapping):
                raise ValueError("profiles must be a mapping of selector to profile")
            profiles.update(
                {str(key): GraspProfile.from_dict(value) for key, value in raw_profiles.items()}
            )
        return cls(profiles)

    def select(self, object_name: str, dimensions, *, metadata: Mapping[str, Any] | None = None) -> GraspProfile:
        metadata = metadata or {}
        selectors = [
            metadata.get("asset_id"),
            metadata.get("usd_path"),
            metadata.get("category"),
            object_name,  # Exact object-name match is intentionally last.
        ]
        for selector in selectors:
            if selector and selector in self.profiles:
                return self.profiles[selector]
        if object_name not in self.generated:
            self.generated[object_name] = profile_from_geometry(dimensions)
        return self.generated[object_name]

    def export_generated(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as stream:
            yaml.safe_dump({"profiles": {key: value.to_dict() for key, value in self.generated.items()}}, stream)

