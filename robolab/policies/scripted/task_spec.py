"""Serializable, task-class-independent manipulation specifications."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

DestinationRelation = Literal["inside", "on_surface", "outside_container_and_on_surface"]
CompletionMode = Literal["all", "any", "choose"]


@dataclass(frozen=True)
class ManipulationTarget:
    """One object-to-destination manipulation, independent of execution details."""

    object_name: str
    destination_name: str
    destination_relation: DestinationRelation
    source_name: str | None = None
    source_relation: str | None = None
    stage: int = 0
    verification_predicate: str | None = None
    verification_params: dict[str, Any] = field(default_factory=dict)

    def required_entities(self) -> set[str]:
        names = {self.object_name, self.destination_name}
        if self.source_name:
            names.add(self.source_name)
        return names

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ManipulationTarget":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown manipulation-target fields: {sorted(unknown)}")
        return cls(**dict(data))

    def __post_init__(self) -> None:
        if not self.object_name or not self.destination_name:
            raise ValueError("object_name and destination_name must be non-empty")
        if self.destination_relation not in {
            "inside", "on_surface", "outside_container_and_on_surface"
        }:
            raise ValueError(f"Unsupported destination relation: {self.destination_relation}")
        if self.destination_relation == "outside_container_and_on_surface" and not self.source_name:
            raise ValueError("outside_container_and_on_surface requires source_name")
        if self.stage < 0:
            raise ValueError("stage must be non-negative")


@dataclass(frozen=True)
class ScriptedTaskSpec:
    """Normalized manipulation semantics consumed by the generic state machine."""

    targets: tuple[ManipulationTarget, ...]
    completion_mode: CompletionMode = "all"
    required_count: int | None = None
    ordered: bool = False
    task_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        if not self.targets:
            raise ValueError("A scripted task spec must contain at least one target")
        if self.completion_mode not in {"all", "any", "choose"}:
            raise ValueError(f"Invalid completion_mode: {self.completion_mode}")
        if self.completion_mode == "choose":
            if self.required_count is None or not 1 <= self.required_count <= len(self.targets):
                raise ValueError("choose completion requires required_count in [1, len(targets)]")
        elif self.required_count is not None:
            raise ValueError("required_count is only valid for choose completion")
        stages = [target.stage for target in self.targets]
        if self.ordered and sorted(set(stages)) != list(range(max(stages) + 1)):
            raise ValueError("Ordered specs require contiguous stage numbers beginning at zero")

    @property
    def required_entities(self) -> tuple[str, ...]:
        return tuple(sorted(set().union(*(target.required_entities() for target in self.targets))))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["targets"] = [asdict(target) for target in self.targets]
        return data

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScriptedTaskSpec":
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown task-spec fields: {sorted(unknown)}")
        values = dict(data)
        raw_targets = values.get("targets")
        if not isinstance(raw_targets, (list, tuple)):
            raise ValueError("task spec field 'targets' must be a list")
        values["targets"] = tuple(ManipulationTarget.from_dict(item) for item in raw_targets)
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> "ScriptedTaskSpec":
        path = Path(path)
        with path.open("r", encoding="utf-8") as stream:
            if path.suffix.lower() == ".json":
                data = json.load(stream)
            elif path.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.safe_load(stream)
            else:
                raise ValueError("Task specs must use .json, .yaml, or .yml")
        if not isinstance(data, Mapping):
            raise ValueError("Task spec root must be a mapping")
        return cls.from_dict(data)

    def validate_entities(self, entities: Mapping[str, Any] | set[str] | list[str]) -> None:
        available = set(entities)
        missing = set(self.required_entities) - available
        if missing:
            raise ValueError(
                f"Task spec references missing entities {sorted(missing)}; "
                f"available entities: {sorted(available)}"
            )

