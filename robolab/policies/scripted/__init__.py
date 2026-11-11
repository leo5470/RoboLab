"""Config-driven expert policies for common manipulation tasks."""

from .task_spec import ManipulationTarget, ScriptedTaskSpec

__all__ = [
    "ManipulationTarget",
    "ScriptedExpertPolicy",
    "ScriptedPolicyConfig",
    "ScriptedTaskSpec",
    "UnsupportedTaskSpecError",
    "extract_task_spec",
]


def __getattr__(name: str):
    """Keep geometry/spec modules importable without loading Isaac Sim."""
    if name in {"ScriptedExpertPolicy", "ScriptedPolicyConfig"}:
        from .policy import ScriptedExpertPolicy, ScriptedPolicyConfig

        return {
            "ScriptedExpertPolicy": ScriptedExpertPolicy,
            "ScriptedPolicyConfig": ScriptedPolicyConfig,
        }[name]
    if name in {"UnsupportedTaskSpecError", "extract_task_spec"}:
        from .task_spec_extractors import UnsupportedTaskSpecError, extract_task_spec

        return {
            "UnsupportedTaskSpecError": UnsupportedTaskSpecError,
            "extract_task_spec": extract_task_spec,
        }[name]
    raise AttributeError(name)
