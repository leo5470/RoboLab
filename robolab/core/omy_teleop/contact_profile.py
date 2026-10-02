"""Contact-sensor selection for measured OMY task variants."""

# Only tasks whose termination and subtask contact dependencies were audited.
# All gripper-object sensors and the gripper batch sensor remain available.
REQUIRED_OBJECT_PAIRS = {
    "BananaInBowlTask": {frozenset(("banana", "bowl")), frozenset(("banana", "table"))},
    "AnimalsInBinTask": {frozenset((name, "table")) for name in
                         ("lizard_figurine", "lizard_figurine_01", "grey_bin")},
}


def selected_contact_sensors(task: str, names: list[str], profile: str) -> tuple[str, set[str]]:
    """Return effective profile and sensor names; unknown tasks use full mode."""
    if profile not in {"full", "required"}:
        raise ValueError(f"Unknown contact sensor profile: {profile}")
    if profile == "full" or task not in REQUIRED_OBJECT_PAIRS:
        return "full", set(names)
    required_pairs = REQUIRED_OBJECT_PAIRS[task]
    selected = set()
    for name in names:
        if "__" not in name:
            selected.add(name)
            continue
        first, second = name.split("__", 1)
        if first == "gripper" or second == "gripper" or frozenset((first, second)) in required_pairs:
            selected.add(name)
    return "required", selected
