"""Explicit predicate-to-spec extraction for supported RoboLab task families."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

from robolab.core.task import conditionals

from .task_spec import ManipulationTarget, ScriptedTaskSpec


class UnsupportedTaskSpecError(ValueError):
    """Raised before execution when task semantics cannot be normalized safely."""


PredicateExtractor = Callable[[Mapping[str, Any]], tuple[list[ManipulationTarget], str, int | None, bool]]


class PredicateExtractorRegistry:
    """Registry keyed by predicate function identity, never instruction text."""

    def __init__(self) -> None:
        self._extractors: dict[Callable, PredicateExtractor] = {}

    def register(self, predicate: Callable):
        def decorator(extractor: PredicateExtractor) -> PredicateExtractor:
            self._extractors[predicate] = extractor
            return extractor
        return decorator

    def extract(self, predicate: Callable, params: Mapping[str, Any]):
        extractor = self._extractors.get(predicate)
        if extractor is None:
            raise UnsupportedTaskSpecError(f"unsupported predicate '{_callable_name(predicate)}'")
        return extractor(params)

    def supports(self, predicate: Callable) -> bool:
        return predicate in self._extractors


REGISTRY = PredicateExtractorRegistry()


def _callable_name(func: Callable) -> str:
    return getattr(func, "__name__", func.__class__.__name__)


def _objects(params: Mapping[str, Any], key: str = "object") -> list[str]:
    raw = params.get(key)
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple)) and all(isinstance(item, str) for item in raw):
        return list(raw)
    raise UnsupportedTaskSpecError(f"predicate parameter '{key}' must be a string or list of strings")


def _mode(params: Mapping[str, Any]) -> tuple[str, int | None]:
    mode = params.get("logical", "all")
    if mode not in {"all", "any", "choose"}:
        raise UnsupportedTaskSpecError(f"unsupported logical mode '{mode}'")
    count = params.get("K") if mode == "choose" else None
    return mode, count


def _verification_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Retain serializable task predicate options but remove runtime-only arguments."""
    return {key: value for key, value in params.items() if key not in {"env", "env_id"}}


@REGISTRY.register(conditionals.object_in_container)
def _extract_inside(params: Mapping[str, Any]):
    mode, count = _mode(params)
    destination = params.get("container")
    if not isinstance(destination, str):
        raise UnsupportedTaskSpecError("object_in_container requires string parameter 'container'")
    targets = [
        ManipulationTarget(
            object_name=name,
            destination_name=destination,
            destination_relation="inside",
            verification_predicate="object_in_container",
            verification_params=_verification_params(params),
        )
        for name in _objects(params)
    ]
    return targets, mode, count, False


@REGISTRY.register(conditionals.object_on_top)
def _extract_on_surface(params: Mapping[str, Any]):
    mode, count = _mode(params)
    destination = params.get("reference_object")
    if not isinstance(destination, str):
        raise UnsupportedTaskSpecError("object_on_top requires string parameter 'reference_object'")
    targets = [
        ManipulationTarget(
            object_name=name,
            destination_name=destination,
            destination_relation="on_surface",
            verification_predicate="object_on_top",
            verification_params=_verification_params(params),
        )
        for name in _objects(params)
    ]
    return targets, mode, count, False


@REGISTRY.register(conditionals.object_outside_of_and_on_surface)
def _extract_outside_on_surface(params: Mapping[str, Any]):
    mode, count = _mode(params)
    source = params.get("container")
    destination = params.get("surface")
    if not isinstance(source, str) or not isinstance(destination, str):
        raise UnsupportedTaskSpecError(
            "object_outside_of_and_on_surface requires string 'container' and 'surface' parameters"
        )
    targets = [
        ManipulationTarget(
            object_name=name,
            source_name=source,
            source_relation="inside",
            destination_name=destination,
            destination_relation="outside_container_and_on_surface",
            verification_predicate="object_outside_of_and_on_surface",
            verification_params=_verification_params(params),
        )
        for name in _objects(params)
    ]
    return targets, mode, count, False


@REGISTRY.register(conditionals.object_groups_in_containers)
def _extract_grouped_inside(params: Mapping[str, Any]):
    groups = params.get("groups")
    if not isinstance(groups, list) or not groups:
        raise UnsupportedTaskSpecError("object_groups_in_containers requires a non-empty 'groups' list")
    targets: list[ManipulationTarget] = []
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("container"), str):
            raise UnsupportedTaskSpecError("each grouped placement requires an object and container")
        group_params = dict(group)
        group_params.setdefault("require_gripper_detached", True)
        for name in _objects(group):
            targets.append(
                ManipulationTarget(
                    object_name=name,
                    destination_name=group["container"],
                    destination_relation="inside",
                    verification_predicate="object_in_container",
                    verification_params=group_params,
                )
            )
    return targets, "all", None, False


@REGISTRY.register(conditionals.objects_placed_in_container_in_order)
def _extract_ordered_inside(params: Mapping[str, Any]):
    destination = params.get("container")
    if not isinstance(destination, str):
        raise UnsupportedTaskSpecError("ordered container placement requires string 'container'")
    objects = _objects(params, "objects")
    common = _verification_params(params)
    targets = []
    for stage, name in enumerate(objects):
        per_object = {key: value for key, value in common.items() if key not in {"objects", "strict", "key"}}
        per_object["object"] = name
        targets.append(
            ManipulationTarget(
                object_name=name,
                destination_name=destination,
                destination_relation="inside",
                stage=stage,
                verification_predicate="object_in_container",
                verification_params=per_object,
            )
        )
    return targets, "all", None, True


def _termination_terms(terminations: Any) -> list[tuple[str, Callable, Mapping[str, Any]]]:
    terms = []
    if terminations is None:
        return terms
    for name in dir(terminations):
        if name.startswith("_"):
            continue
        term = getattr(terminations, name)
        func = getattr(term, "func", None)
        params = getattr(term, "params", None)
        if callable(func) and isinstance(params, Mapping) and not getattr(term, "time_out", False):
            terms.append((name, func, params))
    return terms


def _subtask_summary(subtasks: Any) -> list[str]:
    if not subtasks:
        return []
    summary = []
    for index, subtask in enumerate(subtasks):
        terminal_names: list[str] = []
        for conditions_for_group in getattr(subtask, "conditions", {}).values():
            for condition, _score in conditions_for_group:
                func = condition.func if isinstance(condition, partial) else condition
                terminal_names.append(_callable_name(func))
        summary.append(
            f"stage {index}: {getattr(subtask, 'name', type(subtask).__name__)} "
            f"logical={getattr(subtask, 'logical', '?')} predicates={terminal_names}"
        )
    return summary


def extract_task_spec(task_cfg: Any, *, task_name: str | None = None) -> ScriptedTaskSpec:
    """Extract a spec from explicit, registered termination predicates.

    Subtasks are reported in unsupported-task diagnostics and used to cross-check
    ordering, but termination predicates remain authoritative because they define
    actual episode success.
    """
    requested = task_name or getattr(task_cfg, "_task_name", None) or type(task_cfg).__name__
    terms = _termination_terms(getattr(task_cfg, "terminations", None))
    found = [f"{name}:{_callable_name(func)}" for name, func, _ in terms]
    supported = [(name, func, params) for name, func, params in terms if REGISTRY.supports(func)]
    unsupported = [(name, func) for name, func, _ in terms if not REGISTRY.supports(func)]

    if unsupported or len(supported) != 1:
        detail = ", ".join(found) if found else "none"
        subtask_detail = "; ".join(_subtask_summary(getattr(task_cfg, "subtasks", None))) or "none"
        reason = (
            "multiple success predicates are ambiguous" if len(supported) > 1 else
            "no supported authoritative success predicate"
        )
        if unsupported:
            reason = "unsupported termination predicate(s): " + ", ".join(
                f"{name}:{_callable_name(func)}" for name, func in unsupported
            )
        raise UnsupportedTaskSpecError(
            f"Cannot normalize requested task '{requested}': {reason}. "
            f"Termination predicates found: {detail}. Subtask structures found: {subtask_detail}. "
            "Provide --task-spec <path.yaml> with explicit normalized semantics."
        )

    _, predicate, params = supported[0]
    try:
        targets, mode, count, ordered = REGISTRY.extract(predicate, params)
    except UnsupportedTaskSpecError as exc:
        raise UnsupportedTaskSpecError(
            f"Cannot normalize requested task '{requested}': {_callable_name(predicate)}: {exc}. "
            "Provide --task-spec <path.yaml>."
        ) from exc

    return ScriptedTaskSpec(
        targets=tuple(targets),
        completion_mode=mode,
        required_count=count,
        ordered=ordered,
        task_name=requested,
    )


def load_or_extract_task_spec(task_cfg: Any, override: str | None, task_name: str) -> ScriptedTaskSpec:
    """Load an explicit override, otherwise use registered task semantics."""
    if override:
        spec = ScriptedTaskSpec.load(override)
        if spec.task_name is None:
            data = spec.to_dict()
            data["task_name"] = task_name
            spec = ScriptedTaskSpec.from_dict(data)
        return spec
    return extract_task_spec(task_cfg, task_name=task_name)

