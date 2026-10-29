"""OMY consumer requirements, keyed by callable identity rather than task name.

No simulator imports until the default registry is built. Unknown consumers fail
closed. Inspecting a task never executes its predicates.
"""

import inspect
from dataclasses import dataclass, field
from functools import partial
from itertools import combinations


@dataclass
class TaskRequirements:
    contacts: set[tuple[str, str]] = field(default_factory=set)
    force_contacts: set[tuple[str, str]] = field(default_factory=set)
    poses: set[str] = field(default_factory=set)
    velocities: set[str] = field(default_factory=set)
    joints: set[str] = field(default_factory=set)
    consumers: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    event_tracking: bool = False

    def report(self):
        return {
            "version": 1, "contacts": [list(pair) for pair in sorted(self.contacts)],
            "force_contacts": [list(pair) for pair in sorted(self.force_contacts)],
            "poses": sorted(self.poses), "velocities": sorted(self.velocities),
            "joints": sorted(self.joints), "consumers": self.consumers.copy(),
            "fallback_reasons": list(dict.fromkeys(self.unknown)),
            "event_tracking": self.event_tracking,
        }


def _names(value):
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(name, str) and name for name in value):
        return list(value)
    raise ValueError(f"expected a body name or list of names, got {value!r}")


def _contact(req, first, second, *, force=False):
    for left in _names(first):
        for right in _names(second):
            pair = tuple(sorted((left, right)))
            req.contacts.add(pair)
            if force:
                req.force_contacts.add(pair)


def _spatial(req, p):
    objects = _names(p["object"])
    req.poses.update(objects)
    reference = p.get("container", p.get("reference_object", p.get("surface")))
    for key in ("container", "reference_object", "surface", "reference_obj1", "reference_obj2",
                "frame_of_reference"):
        if key in p and p[key] is not None:
            req.poses.update(_names(p[key]))
    contact = p.get("require_contact_with")
    if contact is True:
        if reference is None:
            raise ValueError("contact=True has no reference body")
        _contact(req, objects, reference)
    elif contact:
        _contact(req, objects, contact)
    if p.get("require_gripper_detached"):
        _contact(req, objects, p.get("gripper_name", "gripper"))
    if p.get("require_stationary"):
        req.velocities.update(objects)


def _supported(req, p):
    _spatial(req, p)
    _contact(req, p["object"], p.get("surface", p.get("reference_object")), force=True)


def _resting_in_container(req, p):
    _spatial(req, {**p, "require_gripper_detached": True})
    req.velocities.update(_names(p["object"]) + _names(p["container"]))
    _contact(req, p["object"], p["container"], force=True)
    if p.get("exclude_contact_with"):
        _contact(req, p["object"], p["exclude_contact_with"])


def _grasp(req, p):
    _contact(req, p["object"], p.get("gripper_name", "gripper"))


def _picked_up(req, p):
    _grasp(req, p)
    req.poses.update(_names(p["object"]) + _names(p["surface"]))


def _groups(req, p):
    for group in p["groups"] or []:
        _spatial(req, {"require_gripper_detached": True, **group})


def _stacked(req, p):
    objects = _names(p["objects"])
    req.poses.update(objects)
    # Order-agnostic sorting can select any neighboring pair at runtime.
    # Union all possibilities up front, not only the initial vertical order.
    for first, second in combinations(objects, 2):
        _contact(req, first, second)


def predicate_registry(conditionals=None):
    """Audit reusable primitives, not arbitrary functions carrying @atomic."""
    if conditionals is None:
        from robolab.core.task import conditionals

    rules = {}
    spatial = (
        "object_in_container", "object_on_bottom", "object_on_center", "object_left_of",
        "object_right_of", "object_in_front_of", "object_behind", "object_next_to",
        "object_below_top", "object_below", "object_enclosed", "object_inside",
        "object_outside_of", "object_upright", "object_at", "object_between",
        "object_center_of", "object_above", "object_above_bottom",
    )
    for name in spatial:
        if hasattr(conditionals, name):
            rules[getattr(conditionals, name)] = _spatial
    for name, rule in {
        "object_on_top": _supported, "object_outside_of_and_on_surface": _supported,
        "object_resting_in_container": _resting_in_container,
        "object_grabbed": _grasp, "object_dropped": _grasp, "object_picked_up": _picked_up,
        "object_in_contact": lambda req, p: _contact(req, p["object1"], p["object2"]),
        "objects_stationary": lambda req, p: req.velocities.update(_names(p["object"])),
        "objects_in_line": lambda req, p: req.poses.update(_names(p["objects"])),
        "object_groups_in_containers": _groups,
        "stacked": _stacked,
    }.items():
        if hasattr(conditionals, name):
            rules[getattr(conditionals, name)] = rule
    # Temporal/stateful primitives require a separate audit, not decorator inference.
    return rules


def default_registry():
    import isaaclab.envs.mdp as mdp

    from robolab.robots import droid
    from . import observations
    from .mimicgen import reset_poses
    from robolab.core.events.reset_pose import reset_pose_uniform

    rules = predicate_registry()
    rules[mdp.time_out] = lambda req, p: None
    rules[mdp.reset_scene_to_default] = lambda req, p: None
    # These reset-only samplers write poses/velocities without stepping physics.
    rules[reset_poses] = lambda req, p: None
    rules[reset_pose_uniform] = lambda req, p: None
    for module in (droid, observations):
        for name in ("arm_joint_pos", "gripper_pos", "ee_pos", "ee_quat", "eef_pos", "eef_quat"):
            if hasattr(module, name):
                rules[getattr(module, name)] = lambda req, p: req.joints.add("robot")
    return rules


def default_recorder_types():
    from robolab.core.events import basic_recorders
    from robolab.core.events.subtask_recorder import SubtaskCompletionRecorderTerm
    from .recording import BoundaryTimeRecorder, LeaderStateRecorder, MimicPoseRecorder

    types = {getattr(basic_recorders, name) for name in (
        "InitialStateRecorder", "PostStepStatesRecorder", "PreStepActionsRecorder",
        "PreStepFlatPolicyObservationsRecorder", "PostStepBBoxRecorder",
        "PostStepEndEffectorPoseRecorder", "PostStepRobotRootPoseRecorder",
    )}
    return types | {SubtaskCompletionRecorderTerm, BoundaryTimeRecorder, LeaderStateRecorder,
                    MimicPoseRecorder}, SubtaskCompletionRecorderTerm


def add_callable(req, func, params, source, registry, *, event=False):
    args, keywords = (), {}
    while isinstance(func, partial):
        args = func.args + args
        keywords = {**(func.keywords or {}), **keywords}
        func = func.func
    name = getattr(func, "__qualname__", type(func).__name__)
    req.consumers.append(f"{source}: {name}")
    try:
        rule = registry.get(func) if callable(func) else None
    except TypeError:
        rule = None
    if rule is None:
        req.unknown.append(f"{source}: unregistered consumer {name}")
        return
    marker = object()
    try:
        # Match partial(env, **params), including positional arguments.
        signature = inspect.signature(func)
        supplied = {**keywords, **(params or {})}
        if event:
            for key in ("env_ids", "global_env_step_count"):
                if key in signature.parameters and key not in supplied:
                    supplied[key] = None
        bound = signature.bind(*args, marker, **supplied)
        bound.apply_defaults()
        if bound.arguments.get("env") is not marker:
            raise ValueError("consumer binds a different environment")
        rule(req, bound.arguments)
    except (TypeError, ValueError, KeyError) as error:
        req.unknown.append(f"{source}: cannot resolve {name}: {error}")


def _terms(cfg):
    if cfg is None:
        return []
    try:
        values = vars(cfg).items() if not isinstance(cfg, dict) else cfg.items()
    except TypeError:
        return [("unknown", cfg)]
    return [(name, term) for name, term in values if not name.startswith("_") and term is not None]


def build_task_requirements(cfg, *, registry=None, recorder_types=None, subtask_recorder_type=None):
    """Union all consumers/stages. Unrecognized configs retain all contacts."""
    registry = default_registry() if registry is None else registry
    if recorder_types is None:
        recorder_types, subtask_recorder_type = default_recorder_types()
    req = TaskRequirements()
    for manager in ("terminations", "rewards", "commands", "curriculum", "events"):
        for name, term in _terms(getattr(cfg, manager, None)):
            if hasattr(term, "func"):
                add_callable(req, term.func, getattr(term, "params", {}), f"{manager}/{name}", registry,
                             event=manager == "events")
            else:
                req.unknown.append(f"{manager}/{name}: unknown term configuration")
    for group, terms in _terms(getattr(cfg, "observations", None)):
        for name, term in _terms(terms):
            if hasattr(term, "func"):
                add_callable(req, term.func, getattr(term, "params", {}), f"observations/{group}/{name}", registry)
                if getattr(term, "modifiers", None):
                    req.unknown.append(f"observations/{group}/{name}: unregistered modifiers")
    for name, term in _terms(getattr(cfg, "recorders", None)):
        if name == "dataset_file_handler_class_type":
            # The writer receives buffered episodes, not the environment. It
            # is a RecorderManager setting, not a per-step RecorderTerm.
            continue
        cls = getattr(term, "class_type", None)
        if cls is None:
            if callable(term):
                req.unknown.append(f"recorders/{name}: unknown consumer")
            continue
        if cls not in recorder_types:
            req.unknown.append(f"recorders/{name}: unregistered recorder {getattr(cls, '__name__', cls)}")
        if cls is subtask_recorder_type and getattr(cfg, "subtasks", None) is not None:
            req.event_tracking = True
    for index, subtask in enumerate(getattr(cfg, "subtasks", None) or []):
        conditions = getattr(subtask, "conditions", None)
        if not isinstance(conditions, dict):
            req.unknown.append(f"subtasks/{index}: unrecognized condition structure")
            continue
        for group, sequence in conditions.items():
            for condition in sequence:
                func = condition[0] if isinstance(condition, tuple) else condition
                add_callable(req, func, {}, f"subtasks/{index}/{group}", registry)
    if req.event_tracking:
        objects = getattr(cfg, "contact_object_list", None) or []
        req.poses.update(name for name in objects if name != "table")
        req.velocities.update(name for name in objects if name != "table")
        req.joints.add("robot")
        for name in objects:
            _contact(req, "gripper", name)
        req.consumers.append("recorders/events: targets, distractors, table and movement transitions")
    declarations = getattr(cfg, "contact_gripper", None) or {}
    def concrete(name):
        target = declarations.get(name)
        return _names(target) if isinstance(target, (list, tuple)) else [name]
    def resolve(pairs):
        return {tuple(sorted((left, right))) for first, second in pairs
                for left in concrete(first) for right in concrete(second)}
    req.contacts = resolve(req.contacts)
    req.force_contacts = resolve(req.force_contacts)
    return req
