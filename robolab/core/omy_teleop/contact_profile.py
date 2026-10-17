"""Dependency-based OMY sensor selection; physical collisions are never removed."""

from .requirements import TaskRequirements


def selected_contact_sensors(requirements, names, profile, *, grippers=("gripper",)):
    """Keep declared pairs and gripper fallbacks; unknown consumers fail closed.

    Task-name strings are intentionally insufficient evidence to prune sensors.
    """
    if profile not in {"full", "required"}:
        raise ValueError(f"Unknown contact sensor profile: {profile}")
    if profile == "full" or not isinstance(requirements, TaskRequirements) or requirements.unknown:
        return "full", set(names)
    available = {tuple(sorted(name.split("__", 1))) for name in names if "__" in name}
    missing = requirements.contacts - available
    if missing:
        requirements.unknown.append(f"required contact sensors are missing: {sorted(missing)}")
        return "full", set(names)
    selected = set()
    for name in names:
        if "__" not in name:
            selected.add(name)
            continue
        first, second = name.split("__", 1)
        if first in grippers or second in grippers or tuple(sorted((first, second))) in requirements.contacts:
            selected.add(name)
    return "required", selected
