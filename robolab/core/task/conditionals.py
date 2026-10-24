# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
conditionals: "Does task condition Y hold for these objects?"

Task-level API with @atomic/@composite decorators.
Takes `env` as first parameter.
Handles logical modes (all/any/choose) and contact requirements.

All @atomic conditionals support env_id parameter:
  env_id=None (default) → returns Tensor(num_envs,) bool (used by IsaacLab TerminationManager)
  env_id=<int>          → returns bool (used by ConditionalsStateMachine)
"""

import math
from functools import partial
from typing import Callable, Literal, Optional, Union

import torch

import robolab.constants
from robolab.core.task.decorators import atomic, composite
from robolab.core.task.predicate_logic import *
from robolab.core.task.predicate_logic import _and, _not, gripper_detached
from robolab.core.task.subtask import Subtask
from robolab.core.world.world_state import get_world

#########################################################
# Composite conditions
#########################################################

@composite
def pick_and_place(
    object: str | list[str],
    container: str,
    logical: Literal["all", "any", "choose"] = "all",
    K: Optional[int] = None,
    score: float = 1.0,
    *,
    placement_func: Callable | None = None,
    placement_params: dict | None = None,
) -> Subtask:
    """
    A composite subtask that picks up object(s) and places them in a container.

    This function creates parallel subtask sequences for each specified object, where each
    object independently progresses through: grab → verify placement.
    The completion logic determines when the entire group is considered complete.

    Args:
        object: Single object name or list of object names to manipulate in parallel
        container: Target container name where objects should be placed
        logical: Completion mode determining when this subtask group succeeds:
            - "all": All objects must complete their subtasks (default)
            - "any": Success when any single object completes
            - "choose": Success when exactly k objects complete (requires k parameter)
        K: Number of objects that must complete when logical="choose"
        score: Overall score weight for this subtask group (default: 1.0)
        placement_func: Atomic placement check. Defaults to object_in_container;
            use object_resting_in_container to wait for an observed settled placement.
        placement_params: Extra parameters passed to the placement check, excluding
            object and container. Use the same parameters as the success termination.
    """
    if isinstance(object, str):
        objects = [object]
    else:
        objects = list(object)

    if placement_func is None:
        placement_func = object_in_container
        params = {"require_contact_with": False, "require_gripper_detached": True}
    else:
        params = {}
    params.update(placement_params or {})

    conditions = {}
    for obj in objects:
        conditions[obj] = [
            (partial(object_grabbed, object=obj), 0.0),
            (partial(placement_func, object=obj, container=container, **params), score),
        ]

    return Subtask(name="pick_and_place", conditions=conditions, logical=logical, score=score, K=K)


@composite
def pick_and_place_grouped(
    groups: list[dict],
    logical: Literal["all", "any", "choose"] = "all",
    K: Optional[int] = None,
    score: float = 1.0,
) -> Subtask:
    """
    A composite subtask where DIFFERENT objects go to DIFFERENT
    containers, all tracked in parallel within a single Subtask.

    Use when a phase has multiple destinations and within-phase order
    doesn't matter — e.g. a swap task where each object's final
    destination is fixed but the policy is free to use any maneuver
    (direct, table-buffered, interleaved).  Encoding the phase as
    multiple sequential ``pick_and_place(..., container=X)`` subtasks
    instead would hard-code one specific maneuver and produce a
    non-monotone score curve under any other policy strategy.

    Each ``groups`` entry is a dict ``{"object": <str | list>,
    "container": <str>}``.  Each named object becomes one parallel
    ladder in the resulting Subtask, with its terminal condition
    pointing at the group's container.  All objects across all groups
    track simultaneously; ``logical`` aggregates over the per-object
    completions just as for ``pick_and_place``.

    Args:
        groups: List of group dicts.  Each ``{"object": str | list[str],
            "container": str}`` defines one or more objects bound to a
            single container.
        logical: Completion mode — same semantics as ``pick_and_place``.
            ``"all"`` (default) requires every object across every group
            to complete its ladder.  ``"any"`` succeeds when any single
            object's ladder completes.  ``"choose"`` requires exactly K
            object-ladders complete.
        K: Required for ``logical="choose"``.
        score: Overall score weight for this Subtask.

    Returns:
        Subtask: with one parallel per-object ladder per (object,
        container) pair across all groups.

    Example:
        # Each fruit → bowl, each can → bin; any order across all four
        pick_and_place_grouped(
            groups=[
                {"object": ["lemon_02", "lime01"], "container": "bowl"},
                {"object": ["tuna_can", "corn_can"], "container": "bin_a01"},
            ],
            logical="all",
            score=1.0,
        )

    Note:
        Use this for subtask tracking only, not for termination
        conditions.  For terminations,
        ``object_groups_in_containers`` consumes the same shape.
    """
    conditions: dict = {}
    for grp in groups:
        cont = grp["container"]
        objs = grp["object"]
        if isinstance(objs, str):
            objs = [objs]
        for obj_name in objs:
            conditions[obj_name] = [
                (partial(object_grabbed, object=obj_name), 0.25),
                (partial(object_above_bottom, object=obj_name,
                         reference_object=cont), 0.25),
                (partial(object_dropped, object=obj_name), 0.25),
                (partial(object_in_container, object=obj_name,
                         container=cont, tolerance=0.01), 0.25),
            ]

    return Subtask(
        name="pick_and_place_grouped",
        conditions=conditions,
        logical=logical,
        score=score,
        K=K,
    )



@composite
def pick_and_place_on_surface(
    object: str | list[str],
    surface: str,
    logical: Literal["all", "any", "choose"] = "all",
    K: Optional[int] = None,
    score: float = 1.0
) -> Subtask:
    """
    A composite subtask that picks up object(s) and places them on a surface.

    Similar to pick_and_place, but verifies stable support on a flat surface using
    contact force cone checking rather than containment checks.

    Args:
        object: Single object name or list of object names to manipulate in parallel
        surface: Target surface name where objects should be placed
        logical: Completion mode - "all", "any", or "choose"
        k: Number of objects that must complete when logical="choose"
        score: Overall score weight for this subtask group (default: 1.0)
    """
    if isinstance(object, str):
        objects = [object]
    else:
        objects = list(object)

    conditions = {}
    for obj in objects:
        conditions[obj] = [
            (partial(object_grabbed, object=obj), 0.0),
            (partial(object_on_top, object=obj, reference_object=surface, require_gripper_detached=True), score),
        ]

    return Subtask(name="pick_and_place_on_surface", conditions=conditions, logical=logical, score=score, K=K)


#########################################################
# Atomic conditions - Contact
#########################################################

@atomic
def object_in_contact(
    env,
    object1: str | list[str],
    object2: str | list[str],
    logical: str = "any",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks contact between objects according to logical."""
    if logical not in ["any", "all", "choose"]:
        raise ValueError(f"Invalid logical: {logical}")

    world = get_world(env)
    result = in_contact(world, object1, object2, force_threshold=0.1, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_in_contact: {object1} and {object2} in contact (logical={logical}) -> {result}")
    return result

@atomic
def object_grabbed(
    env,
    object: str,
    gripper_name: str | list[str] = "gripper",
    env_id: int | None = None,
):
    """Check if an object is currently being grabbed by the gripper (in contact with gripper)."""
    world = get_world(env)
    result = in_contact(world, object, gripper_name, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_grabbed: '{object}' in contact with '{gripper_name}' -> {result}")
    return result

@atomic
def object_dropped(
    env,
    object: str,
    gripper_name: str | list[str] = "gripper",
    env_id: int | None = None,
):
    """Check if an object has been dropped (in contact with none of the given grippers)."""
    world = get_world(env)
    result = gripper_detached(world, object, gripper_name, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_dropped: '{object}' not in contact with '{gripper_name}' -> {result}")
    return result

@atomic
def object_picked_up(
    env,
    object: str,
    surface: str,
    distance: float = 0.05,
    env_id: int | None = None,
):
    """Check if object is grabbed and lifted at least `distance` above the surface."""
    result = _and(
        object_grabbed(env, object, env_id=env_id),
        object_above(env, object=object, reference_object=surface, env_id=env_id, z_margin=distance)
    )
    if robolab.constants.DEBUG:
        print(f"object_picked_up: '{object}' grabbed and lifted {distance}m above '{surface}' -> {result}")
    return result

#########################################################
# Unified Spatial Conditions (New API)
#########################################################
#
# Parameters:
#   require_contact_with: Contact requirement
#       - False: no contact check (default)
#       - True: must be in contact with reference_object
#       - str/list[str]: must be in contact with specified object(s)
#   require_gripper_detached: If True, object must NOT be held by gripper
#

@atomic
def object_in_container(
    env,
    object: str | list[str],
    container: str,
    tolerance: float = 0.01,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    require_stationary: bool = False,
    stationary_threshold: float = 0.05,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are in an open-top container.

    The object's hull-vertex centroid is transformed into the container's local
    frame and tested against its convex hull with the top faces removed. This
    includes the air column above the opening. Optional stationary checking uses
    linear velocity at the current step only; object_resting_in_container also
    checks angular motion, support, release, and an observed settling interval.
    """
    def condition(world, obj, env_id=None):
        result = in_opentop_container(
            world, obj, container,
            tolerance=tolerance,
            env_id=env_id,
        )
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, container, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=env_id))
        if require_stationary:
            result = _and(result, stationary(world, obj, linear_threshold=stationary_threshold, check_angular=False, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_in_container: {object} in '{container}' (tol={tolerance}, logical={logical}) -> {result}")
    return result

def _condition_held_for_time(world, current, *, key, duration_s, positions, orientations,
                             position_tolerance, rotation_tolerance_deg, env_id=None):
    """Require a condition and stable poses over consecutive observed steps.

    Terminations and the subtask recorder may evaluate the same condition in the
    same step (batched or scalar). Episode-step timestamps make these calls
    idempotent: additional evaluations never advance the settling clock. A gap
    in observations restarts the window rather than assuming unobserved rest.
    """
    env = world.env
    required_steps = math.ceil(duration_s / env.step_dt - 1e-9)

    def make_state():
        since = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        last = torch.full_like(since, -1)
        anchor_pos = torch.zeros((env.num_envs, 2, 3), device=env.device)
        anchor_quat = torch.zeros((env.num_envs, 2, 4), device=env.device)

        def reset(env_ids):
            since[env_ids] = -1
            last[env_ids] = -1

        return {"since": since, "last": last, "anchor_pos": anchor_pos,
                "anchor_quat": anchor_quat, "__reset__": reset}

    state = world.get_or_init_predicate_state(key, make_state)
    rows = slice(None) if env_id is None else slice(env_id, env_id + 1)
    step = env.episode_length_buf[rows]
    if not isinstance(current, torch.Tensor):
        current = bool(current)
    current = torch.as_tensor(current, dtype=torch.bool, device=env.device).reshape(-1)
    since, last = state["since"][rows], state["last"][rows]
    positions, orientations = positions.reshape(-1, 2, 3), orientations.reshape(-1, 2, 4)
    anchor_pos, anchor_quat = state["anchor_pos"][rows], state["anchor_quat"][rows]
    position_ok = torch.linalg.vector_norm(positions - anchor_pos, dim=-1) <= position_tolerance
    # q and -q represent the same orientation. Normalize the dot product because
    # physics quaternions can have small norm error; compare to cos(theta/2)
    # rather than taking acos near 1, where rounding amplifies numerical noise.
    quat_norms = torch.linalg.vector_norm(orientations, dim=-1) * torch.linalg.vector_norm(anchor_quat, dim=-1)
    quat_dot = (orientations * anchor_quat).sum(dim=-1).abs() / quat_norms.clamp_min(1e-12)
    rotation_ok = quat_dot >= math.cos(math.radians(rotation_tolerance_deg) / 2)
    pose_ok = (position_ok & rotation_ok).all(dim=-1)
    consecutive = (last == step) | (last == step - 1)
    continuing = (since >= 0) & consecutive & pose_ok
    start = torch.where(continuing, since, step)
    anchor_pos.copy_(torch.where(continuing[:, None, None], anchor_pos, positions))
    anchor_quat.copy_(torch.where(continuing[:, None, None], anchor_quat, orientations))
    since.copy_(torch.where(current, start, -1))
    last.copy_(step)
    result = current & (step - since >= required_steps)
    return bool(result[0]) if env_id is not None else result


@atomic
def object_resting_in_container(
    env,
    object: str | list[str],
    container: str,
    gripper_name: str | list[str] = "gripper",
    exclude_contact_with: str | list[str] | None = None,
    settle_time_s: float = 0.5,
    linear_threshold: float = 0.05,
    angular_threshold: float = 0.5,
    position_tolerance: float = 0.001,
    rotation_tolerance_deg: float = 0.5,
    support_force_threshold: float = 0.01,
    max_container_tilt_deg: float = 15.0,
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Check an observed, released and settled placement supported by the bin.

    The object's hull centroid must lie within the existing open-top containment
    region. Protrusion and stable rim-spanning placements are allowed. The bin
    must be upright, provide upward support, and be stationary. Both linear and
    angular object velocities must stay below their thresholds for settle_time_s
    simulated seconds, with no gripper contact or excluded-surface contact. Both
    object and bin poses must remain within position_tolerance (1 mm) and
    rotation_tolerance_deg (0.5 degrees) of their poses when the interval began.
    This pose requirement rejects slow drift while allowing contact-solver noise
    in instantaneous linear and angular velocity.

    Pass exclude_contact_with="table" when the table must not support the target.
    The window resets on any failed check, missing observation, or episode reset.
    It is shared with identically parameterized subtask/termination checks, so
    full placement credit and episode success occur on the same observed step.
    """
    for name, value in (("settle_time_s", settle_time_s),
                        ("linear_threshold", linear_threshold),
                        ("angular_threshold", angular_threshold),
                        ("position_tolerance", position_tolerance),
                        ("support_force_threshold", support_force_threshold)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(max_container_tilt_deg) or not 0 < max_container_tilt_deg < 90:
        raise ValueError("max_container_tilt_deg must be between 0 and 90 degrees")
    if not math.isfinite(rotation_tolerance_deg) or not 0 < rotation_tolerance_deg < 180:
        raise ValueError("rotation_tolerance_deg must be between 0 and 180 degrees")

    excluded = [exclude_contact_with] if isinstance(exclude_contact_with, str) else list(exclude_contact_with or [])
    grippers = [gripper_name] if isinstance(gripper_name, str) else list(gripper_name)
    settings = (container, tuple(sorted(grippers)), tuple(sorted(excluded)),
                settle_time_s, linear_threshold, angular_threshold, position_tolerance, rotation_tolerance_deg,
                support_force_threshold, max_container_tilt_deg)
    world = get_world(env)
    container_pos, container_quat = world.get_pose(container, env_id=env_id)
    container_ready = _and(
        upright(world, container, tolerance=math.radians(max_container_tilt_deg), env_id=env_id),
        stationary(world, container, linear_threshold, angular_threshold, env_id=env_id),
    )

    def condition(world, obj, env_id=None):
        obj_pos, obj_quat = world.get_pose(obj, env_id=env_id)
        current = in_opentop_container(world, obj, container, env_id=env_id)
        current = _and(current, container_ready)
        current = _and(current, world.is_supported_on_surface(
            obj, container, force_threshold=support_force_threshold, env_id=env_id,
        ))
        current = _and(current, gripper_detached(world, obj, gripper_name, env_id=env_id))
        current = _and(current, stationary(world, obj, linear_threshold, angular_threshold, env_id=env_id))
        for surface in excluded:
            current = _and(current, _not(in_contact(
                world, obj, surface, force_threshold=support_force_threshold, env_id=env_id,
            )))
        return _condition_held_for_time(
            world, current, key=f"resting_in_container::{(obj, settings)!r}",
            duration_s=settle_time_s,
            positions=torch.stack((obj_pos, container_pos), dim=-2),
            orientations=torch.stack((obj_quat, container_quat), dim=-2),
            position_tolerance=position_tolerance, rotation_tolerance_deg=rotation_tolerance_deg,
            env_id=env_id,
        )

    return evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)


def resting_in_container_diagnostics(env, params: dict, env_id: int = 0) -> dict:
    """Read placement requirements and the existing timer without advancing it.

    Intended for collection status and debugging a missed placement. No predicate
    state is created, reset or updated, and no success outcome is overridden.
    """
    import inspect

    bound = inspect.signature(object_resting_in_container).bind(env, **params)
    bound.apply_defaults()
    p = bound.arguments
    obj, container = p["object"], p["container"]
    if not isinstance(obj, str):
        raise ValueError("Placement diagnostics require one object")
    world = get_world(env)
    excluded = p["exclude_contact_with"]
    excluded = [excluded] if isinstance(excluded, str) else list(excluded or [])
    obj_velocity = world.get_velocity(obj, env_id=env_id)
    bin_velocity = world.get_velocity(container, env_id=env_id)
    checks = {
        "inside": bool(in_opentop_container(world, obj, container, env_id=env_id)),
        "bin_upright": bool(upright(world, container, tolerance=math.radians(p["max_container_tilt_deg"]), env_id=env_id)),
        "bin_stationary": bool(stationary(world, container, p["linear_threshold"], p["angular_threshold"], env_id=env_id)),
        "object_stationary": bool(stationary(world, obj, p["linear_threshold"], p["angular_threshold"], env_id=env_id)),
        "bin_support": bool(world.is_supported_on_surface(obj, container, force_threshold=p["support_force_threshold"], env_id=env_id)),
        "gripper_detached": bool(gripper_detached(world, obj, p["gripper_name"], env_id=env_id)),
        "no_excluded_contact": all(not bool(in_contact(world, obj, surface, force_threshold=p["support_force_threshold"], env_id=env_id))
                                   for surface in excluded),
    }
    result = {
        "object": obj, "blocked": [name for name, passed in checks.items() if not passed],
        "checks": checks, "step": int(env.episode_length_buf[env_id]),
        "linear_speed_m_s": {"object": float(torch.linalg.vector_norm(obj_velocity[:3])),
                             "bin": float(torch.linalg.vector_norm(bin_velocity[:3]))},
        "angular_speed_rad_s": {"object": float(torch.linalg.vector_norm(obj_velocity[3:])),
                                "bin": float(torch.linalg.vector_norm(bin_velocity[3:]))},
        "bin_force_n": world.get_contact_force(obj, container, env_id=env_id).tolist(),
        "required_s": p["settle_time_s"], "observed_s": 0.0,
    }
    grippers = [p["gripper_name"]] if isinstance(p["gripper_name"], str) else list(p["gripper_name"])
    settings = (container, tuple(sorted(grippers)), tuple(sorted(excluded)),
                p["settle_time_s"], p["linear_threshold"], p["angular_threshold"],
                p["position_tolerance"], p["rotation_tolerance_deg"],
                p["support_force_threshold"], p["max_container_tilt_deg"])
    state = world._predicate_state.get(f"resting_in_container::{(obj, settings)!r}")
    if state is not None:
        since = int(state["since"][env_id])
        result["observed_s"] = max(0, result["step"] - since) * env.step_dt if since >= 0 else 0.0
        result["evaluated_step"] = int(state["last"][env_id])
        if since >= 0:
            poses = [world.get_pose(name, env_id=env_id) for name in (obj, container)]
            positions = torch.stack([pose[0] for pose in poses])
            quats = torch.stack([pose[1] for pose in poses])
            anchor_q = state["anchor_quat"][env_id]
            dots = (quats * anchor_q).sum(dim=-1).abs()
            norms = torch.linalg.vector_norm(quats, dim=-1) * torch.linalg.vector_norm(anchor_q, dim=-1)
            result["pose_drift_m"] = torch.linalg.vector_norm(positions - state["anchor_pos"][env_id], dim=-1).tolist()
            result["pose_drift_deg"] = torch.rad2deg(2 * torch.acos((dots / norms.clamp_min(1e-12)).clamp(0, 1))).tolist()
    return result


@atomic
def object_on_top(
    env,
    object: str | list[str],
    reference_object: str,
    tolerance: float = 0.01,
    require_contact_with: Union[str, list[str]] = None,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are stably supported on the top surface of reference_object.

    The check is the AND of:
      - is_supported_on_surface: contact-force from surface on obj is non-trivial,
        upward, and within a 45° cone of vertical.
      - centroid_in_footprint: obj's centroid xy lies within the surface's AABB
        (with ``tolerance`` slack). z is intentionally not bounded — concave
        surfaces (plates with wells, tilted/overhanging objects) make any
        all-corners-above-top rule too brittle.
    """
    def condition(world, obj, env_id=None):
        result = world.is_supported_on_surface(obj, reference_object, env_id=env_id)
        result = _and(result, centroid_in_footprint(
            world, obj, reference_object, tolerance=tolerance, env_id=env_id
        ))
        if require_contact_with is not None:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_on_top: {object} on top of '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_on_bottom(
    env,
    object: str | list[str],
    reference_object: str,
    tolerance: float = 0.01,
    z_margin: float = 0.0,
    mode: str = "bbox",
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are positioned above the bottom surface of reference_object."""
    def condition(world, obj, env_id=None):
        result = above_bottom(world, obj, reference_object, tolerance, z_margin, mode, env_id=env_id)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_on_bottom: {object} above bottom of '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_on_center(
    env,
    object: str | list[str],
    reference_object: str,
    tolerance: float = 0.01,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are centered on reference_object (XY alignment)."""
    def condition(world, obj, env_id=None):
        result = center_of(world, obj, reference_object, tolerance, env_id=env_id)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_on_center: {object} centered on '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_left_of(
    env,
    object: str | list[str],
    reference_object: str,
    frame_of_reference: str = "robot",
    mirrored: bool = False,
    cone_deg: int = 45,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are to the left of reference_object."""
    def condition(world, obj, env_id=None):
        result = left_of(world, obj, reference_object, frame_of_reference, mirrored, cone_deg, env_id=env_id)
        if not require_contact_with and not require_gripper_detached:
            result = _and(result, level(world, obj, reference_object, env_id=env_id))
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_left_of: {object} left of '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_right_of(
    env,
    object: str | list[str],
    reference_object: str,
    frame_of_reference: str = "robot",
    mirrored: bool = False,
    cone_deg: int = 45,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are to the right of reference_object."""
    def condition(world, obj, env_id=None):
        result = right_of(world, obj, reference_object, frame_of_reference, mirrored, cone_deg, env_id=env_id)
        if not require_contact_with and not require_gripper_detached:
            result = _and(result, level(world, obj, reference_object, env_id=env_id))
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_right_of: {object} right of '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_in_front_of(
    env,
    object: str | list[str],
    reference_object: str,
    frame_of_reference: str = "robot",
    mirrored: bool = False,
    cone_deg: int = 45,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are in front of reference_object."""
    def condition(world, obj, env_id=None):
        result = in_front_of(world, obj, reference_object, frame_of_reference, mirrored, cone_deg, env_id=env_id)
        if not require_contact_with and not require_gripper_detached:
            result = _and(result, level(world, obj, reference_object, env_id=env_id))
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_in_front_of: {object} in front of '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_behind(
    env,
    object: str | list[str],
    reference_object: str,
    frame_of_reference: str = "robot",
    mirrored: bool = False,
    cone_deg: int = 45,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are behind reference_object."""
    def condition(world, obj, env_id=None):
        result = behind(world, obj, reference_object, frame_of_reference, mirrored, cone_deg, env_id=env_id)
        if not require_contact_with and not require_gripper_detached:
            result = _and(result, level(world, obj, reference_object, env_id=env_id))
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_behind: {object} behind '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_next_to(
    env,
    object: str | list[str],
    reference_object: str,
    dist: float = 0.05,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are within a certain distance of reference_object."""
    def condition(world, obj, env_id=None):
        result = next_to(world, obj, reference_object, dist, env_id=env_id)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_next_to: {object} within {dist}m of '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_below_top(
    env,
    object: str | list[str],
    reference_object: str,
    tolerance: float = 0.01,
    z_margin: float = 0.0,
    mode: str = "bbox",
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are below the top surface of reference_object."""
    def condition(world, obj, env_id=None):
        result = below_top(world, obj, reference_object, tolerance, z_margin, mode, env_id=env_id)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_below_top: {object} below top of '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_below(
    env,
    object: str | list[str],
    reference_object: str,
    tolerance: float = 0.01,
    z_margin: float = 0.0,
    mode: str = "bbox",
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are below the bottom surface of reference_object."""
    def condition(world, obj, env_id=None):
        result = below_bottom(world, obj, reference_object, tolerance, z_margin, mode, env_id=env_id)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_below: {object} below bottom of '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_enclosed(
    env,
    object: str | list[str],
    container: str,
    tolerance: float = 0.01,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects' bounding boxes are fully enclosed inside the container."""
    def condition(world, obj, env_id=None):
        result = enclosed(world, obj, container, tolerance, env_id=env_id)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, container, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_enclosed: {object} enclosed in '{container}' (logical={logical}) -> {result}")
    return result

@atomic
def object_inside(
    env,
    object: str | list[str],
    container: str,
    tolerance: float = 0.01,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects' centroids are inside the container's bounding box."""
    def condition(world, obj, env_id=None):
        result = inside(world, obj, container, tolerance, env_id=env_id)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, container, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_inside: {object} inside '{container}' (logical={logical}) -> {result}")
    return result

@atomic
def object_outside_of(
    env,
    object: str | list[str],
    container: str,
    tolerance: float = 0.01,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are outside the container.

    Symmetric with ``object_in_container``: the object is "outside" iff fewer
    than half of its hull vertices fall in the container's open-top hull
    (``frac_inside < 0.5``). Equivalent to ``not in_opentop_container``.
    """
    def condition(world, obj, env_id=None):
        result = _not(in_opentop_container(world, obj, container, tolerance, env_id=env_id))
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, container, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_outside_of: {object} outside '{container}' (logical={logical}) -> {result}")
    return result

@atomic
def object_upright(
    env,
    object: str | list[str],
    tolerance: float = 0.1,
    up_axis: str = "z",
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are standing upright (oriented correctly)."""
    def condition(world, obj, env_id=None):
        result = upright(world, obj, tolerance, up_axis, env_id=env_id)
        if require_contact_with is True:
            raise ValueError(
                "object_upright(require_contact_with=True) is invalid: object_upright "
                "has no reference_object. Pass a body name (str) or list of body names instead."
            )
        if require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_upright: {object} upright (up_axis={up_axis}, logical={logical}) -> {result}")
    return result

@atomic
def object_at(
    env,
    object: str | list[str],
    position: tuple[float, float, float],
    tolerance: float = 0.02,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Check if objects are at a specific 3D position within tolerance."""
    if logical not in ["any", "all", "choose"]:
        raise ValueError(f"Invalid logical: {logical}")

    world = get_world(env)
    object_list = [object] if isinstance(object, str) else list(object)
    pos_target = torch.tensor(position, dtype=torch.float32, device=world.env.device)

    def check_obj(world, obj, env_id=None):
        pos, _ = world.get_pose(obj, env_id=env_id)
        if env_id is not None:
            at_pos = torch.allclose(pos, pos_target, atol=tolerance)
            if require_gripper_detached:
                at_pos = at_pos and not in_contact(world, obj, gripper_name, env_id=env_id)
            return at_pos
        else:
            # pos: (N, 3)
            diff = torch.abs(pos - pos_target.unsqueeze(0))
            at_pos = (diff <= tolerance).all(dim=1)  # (N,)
            if require_gripper_detached:
                at_pos = at_pos & gripper_detached(world, obj, gripper_name, env_id=None)
            return at_pos

    result = evaluate_spatial_condition(env, object, check_obj, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_at: {object} at {position} (tol={tolerance}, logical={logical}) -> {result}")
    return result

@atomic
def object_between(
    env,
    object: str | list[str],
    reference_obj1: str,
    reference_obj2: str,
    check_alignment: bool = True,
    alignment_tolerance: float = 0.1,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are positioned between two reference objects."""
    def condition(world, obj, env_id=None):
        result = between(world, obj, reference_obj1, reference_obj2, check_alignment, alignment_tolerance, env_id=env_id)
        if require_contact_with and require_contact_with is not True:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_between: {object} between '{reference_obj1}' and '{reference_obj2}' (logical={logical}) -> {result}")
    return result

@atomic
def objects_in_line(
    env,
    objects: list[str],
    axis: str | None = None,
    tolerance: float = 0.05,
    min_spacing: float = 0.02,
    env_id: int | None = None,
):
    """Checks if multiple objects are arranged in a line/row."""
    world = get_world(env)
    result = in_line(world, objects, axis, tolerance, min_spacing, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"objects_in_line: {objects} in a line (axis={axis}) -> {result}")
    return result

@atomic
def objects_stationary(
    env,
    object: str | list[str],
    linear_threshold: float = 0.01,
    angular_threshold: float = 0.1,
    check_angular: bool = True,
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects have stopped moving (velocity near zero)."""
    result = evaluate_spatial_condition(
        env, object,
        lambda world, obj, env_id=None: stationary(world, obj, linear_threshold, angular_threshold, check_angular, env_id=env_id),
        logical, K, env_id=env_id
    )
    if robolab.constants.DEBUG:
        print(f"objects_stationary: {object} stationary (logical={logical}) -> {result}")
    return result

@atomic
def object_center_of(
    env,
    object: str | list[str],
    reference_object: str,
    tolerance: float = 0.01,
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Check if the geometric centers of objects are aligned with reference_object (XY plane only)."""
    def condition(world, obj, env_id=None):
        result = center_of(world, obj, reference_object, tolerance, env_id=env_id)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_center_of: {object} centered on '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_above(
    env,
    object: str | list[str],
    reference_object: str,
    tolerance: float = 0.01,
    z_margin: float = 0.0,
    mode: str = "bbox",
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Check if objects are geometrically positioned above the top surface of reference_object."""
    def condition(world, obj, env_id=None):
        result = above_top(world, obj, reference_object, tolerance, z_margin, mode, env_id=env_id)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, reference_object, env_id=env_id))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=env_id))
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_above: {object} above '{reference_object}' (logical={logical}) -> {result}")
    return result

@atomic
def object_above_bottom(
    env,
    object: str | list[str],
    reference_object: str,
    tolerance: float = 0.01,
    z_margin: float = 0.0,
    mode: str = "bbox",
    require_contact_with: Union[bool, str, list[str]] = False,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Check if objects are positioned above the bottom surface of reference_object."""
    return object_on_bottom(
        env, object, reference_object, tolerance, z_margin, mode,
        require_contact_with, require_gripper_detached, gripper_name,
        logical, K, env_id
    )

#########################################################
# Special compound conditions
#########################################################

@atomic
def object_outside_of_and_on_surface(
    env,
    object: str | list[str],
    container: str,
    surface: str,
    tolerance: float = 0.01,
    require_gripper_detached: bool = False,
    gripper_name: str | list[str] = "gripper",
    logical: str = "all",
    K: int = 1,
    env_id: int | None = None,
):
    """Checks if objects are outside of a container AND stably supported on a surface.

    Symmetric with ``object_in_container``: ``not in_opentop_container``
    (frac_inside < 0.5) for the container check; surface support unchanged.
    """
    def condition(world, obj, env_id=None):
        result = _and(
            _not(in_opentop_container(world, obj, container, tolerance, env_id=env_id)),
            world.is_supported_on_surface(obj, surface, env_id=env_id)
        )
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=env_id))
        return result

    result = evaluate_spatial_condition(env, object, condition, logical, K, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"object_outside_of_and_on_surface: {object} outside '{container}' and on '{surface}' (logical={logical}) -> {result}")
    return result

@atomic
def object_groups_in_containers(
    env,
    groups: list[dict],
    env_id: int | None = None,
):
    """Checks multiple (object(s), container) groups; returns True only if all groups satisfy placement."""
    if groups is None or len(groups) == 0:
        if env_id is not None:
            return False
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    results = []
    for group in groups:
        objects = group.get("object", [])
        container = group.get("container")
        tolerance = group.get("tolerance", 0.01)
        logical = group.get("logical", "all")
        K = group.get("K", 1)
        require_contact_with = group.get("require_contact_with", False)
        require_gripper_detached = group.get("require_gripper_detached", True)

        r = object_in_container(
            env,
            object=objects,
            container=container,
            tolerance=tolerance,
            require_contact_with=require_contact_with,
            require_gripper_detached=require_gripper_detached,
            logical=logical,
            K=K,
            env_id=env_id,
        )
        results.append(r)

    if env_id is not None:
        return all(results)
    else:
        return torch.stack([r if isinstance(r, torch.Tensor) else torch.tensor(r) for r in results]).all(dim=0)


#########################################################
# Not conditions
#########################################################
@atomic
def wrong_object_grabbed(
    env,
    object: str | list[str],
    gripper_name: str | list[str] = "gripper",
    ignore_objects: list[str] = ["table"],
    env_id: int | None = None,
):
    """Check if gripper is holding any object other than the specified target object(s)."""
    if isinstance(object, str):
        intended_set = {object}
    else:
        intended_set = set(object)
    ignore_set = set(ignore_objects)

    candidates = [obj for obj in env.cfg.contact_object_list if obj not in ignore_set]

    # This function returns a list and is inherently per-env
    if env_id is None:
        # Vectorized: check each env
        results = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for eid in range(env.num_envs):
            world = get_world(env)
            objects_in_contact = world.get_objects_in_contact_with(gripper_name, candidates, env_id=eid)
            for obj_name in objects_in_contact:
                if obj_name not in intended_set:
                    results[eid] = True
                    break
        return results
    else:
        world = get_world(env)
        objects_in_contact = world.get_objects_in_contact_with(gripper_name, candidates, env_id=env_id)
        for obj_name in objects_in_contact:
            if obj_name not in intended_set:
                if robolab.constants.DEBUG:
                    print(f"wrong_object_grabbed: Gripper holding '{obj_name}' instead of {object} -> True")
                return True
        if robolab.constants.DEBUG:
            print(f"wrong_object_grabbed: No wrong object grabbed -> False")
        return False


def get_wrong_object_grabbed(
    env,
    intended_objects: str | list[str],
    gripper_name: str | list[str] = "gripper",
    ignore_objects: list[str] = ["table"],
    env_id: int | None = None,
) -> str | None:
    """Get the name of a wrong object that is currently grabbed.
    Note: inherently per-env, defaults to env_id=0 when None."""
    if env_id is None:
        env_id = 0

    if isinstance(intended_objects, str):
        intended_set = {intended_objects}
    else:
        intended_set = set(intended_objects)
    ignore_set = set(ignore_objects)

    candidates = [obj for obj in env.cfg.contact_object_list if obj not in ignore_set]

    world = get_world(env)
    labels = world.resolve_contact_bodies(gripper_name)
    for label in labels:
        # A collision with an open hand is not a grab. Evaluate closure and
        # contact for the same concrete hand instead of combining them across
        # a bimanual alias.
        if not gripper_slightly_closed(env, gripper_name=label, env_id=env_id):
            continue
        objects_in_contact = world.get_objects_in_contact_with(label, candidates, env_id=env_id)
        for obj in objects_in_contact:
            if obj not in intended_set:
                return obj
    return None


def gripper_hit_table(
    env,
    gripper_name: str | list[str] = "gripper",
    table_name: str = "table",
    env_id: int | None = None,
):
    """Check if the gripper is in contact with the table."""
    world = get_world(env)
    result = in_contact(world, gripper_name, table_name, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"gripper_hit_table: '{gripper_name}' in contact with '{table_name}' -> {result}")
    return result


def gripper_fully_closed(
    env,
    robot_name: str = "robot",
    gripper_joint_name: str | None = None,
    closed_threshold: float = 0.75,
    env_id: int | None = None,
    gripper_name: str = "gripper",
):
    """Check whether any selected concrete gripper is sufficiently closed.

    Robots may publish ``env.cfg.gripper_closure_cfg`` as a mapping from concrete
    contact labels to ``(joint_name, open_position, closed_position)``. The
    legacy Droid contract remains the fallback when no declaration is present.
    """
    import math as _math

    world = get_world(env)
    joint_positions = world.get_joint_positions(robot_name, env_id=env_id)
    joint_names = world.get_joint_names(robot_name)
    state_cfg: dict[str, tuple[str, float, float]] = getattr(env.cfg, "gripper_closure_cfg", None) or {}
    labels = world.resolve_contact_bodies(gripper_name)

    # Preserve explicit and pre-declaration callers that use Droid's original
    # single finger joint contract.
    if gripper_joint_name is not None:
        labels = [labels[0] if labels else "gripper"]
        state_cfg = {labels[0]: (gripper_joint_name, 0.0, _math.pi / 4)}
    elif not state_cfg:
        labels = [labels[0] if labels else "gripper"]
        state_cfg = {labels[0]: ("finger_joint", 0.0, _math.pi / 4)}

    closed_states: list[bool | torch.Tensor] = []
    for label in labels:
        cfg = state_cfg.get(label)
        if cfg is None or cfg[0] not in joint_names:
            closed_states.append(
                False
                if env_id is not None
                else torch.zeros(world.env.num_envs, dtype=torch.bool, device=world.env.device)
            )
            continue

        joint_name, open_position, closed_position = cfg
        span = closed_position - open_position
        if abs(span) < 1.0e-9:
            raise ValueError(f"Gripper '{label}' has identical open and closed positions")
        joint_idx = joint_names.index(joint_name)
        gripper_pos = joint_positions[joint_idx] if env_id is not None else joint_positions[:, joint_idx]
        normalized_pos = (gripper_pos - open_position) / span
        state = normalized_pos >= closed_threshold
        closed_states.append(bool(state.item()) if env_id is not None else state)

    if env_id is not None:
        return any(closed_states)
    return torch.stack(closed_states, dim=0).any(dim=0)


def gripper_slightly_closed(
    env,
    robot_name: str = "robot",
    gripper_joint_name: str | None = None,
    closed_threshold: float = 0.30,
    env_id: int | None = None,
    gripper_name: str = "gripper",
):
    """Check if the gripper is slightly closed (at least 30% closed by default)."""
    return gripper_fully_closed(
        env,
        robot_name,
        gripper_joint_name,
        closed_threshold,
        env_id=env_id,
        gripper_name=gripper_name,
    )


#########################################################
# Ordering conditions
#########################################################

@atomic
def stacked(
    env,
    objects: list[str],
    order: str | None = None,
    tolerance: float = 0.01,
    env_id: int | None = None,
):
    """Checks if the objects are stacked in the given order."""
    if order == "None":
        order = None

    if robolab.constants.DEBUG:
        print(f"Checking stacked({objects}, order={order})")

    world = get_world(env)
    result = check_stacked(world, objects, order, tolerance, env_id=env_id)
    if robolab.constants.DEBUG:
        print(f"stacked: {objects} stacked (order={order}, tol={tolerance}) -> {result}")
    return result


@atomic
def objects_placed_in_container_in_order(
    env,
    objects: list[str],
    container: str,
    tolerance: float = 0.01,
    require_contact_with: Union[bool, str, list[str]] = True,
    require_gripper_detached: bool = True,
    gripper_name: str | list[str] = "gripper",
    strict: bool = False,
    key: str | None = None,
    env_id: int | None = None,
):
    """Stateful: checks that ``objects`` ended up in ``container`` in the listed order.

    Latches each object's first-observed placement step per env (using
    ``env.episode_length_buf``). Returns True for an env only when:
      1. Every object currently satisfies the same per-object placement check
         used by ``object_in_container`` (containment + optional contact +
         optional gripper-detached), AND
      2. The latched first-placement steps are monotonic in the given order
         (``<=`` by default; ``<`` if ``strict=True``).

    Latches survive object ejection (first-ever placement is sticky), so a policy
    that places the wrong object first cannot recover by knocking it out and
    re-placing later. Per-env latches are cleared on episode reset by
    ``RobolabEnv._reset_idx`` via ``WorldState.reset_predicate_state``.
    """
    world = get_world(env)
    objects = list(objects)
    state_key = key or f"placed_in_order::{container}::{','.join(objects)}"

    def _make_state():
        N, dev = env.num_envs, env.device
        first = {obj: torch.full((N,), -1, dtype=torch.long, device=dev) for obj in objects}

        def reset(env_ids: torch.Tensor) -> None:
            for t in first.values():
                t[env_ids] = -1

        return {"first_placed_step": first, "__reset__": reset}

    state = world.get_or_init_predicate_state(state_key, _make_state)
    first_placed = state["first_placed_step"]
    ep_step = env.episode_length_buf  # (N,) long

    # Build the same per-object placement predicate as object_in_container.
    def _currently_placed(obj: str, eid: int | None):
        result = in_opentop_container(world, obj, container, tolerance=tolerance, env_id=eid)
        if require_contact_with is True:
            result = _and(result, in_contact(world, obj, container, env_id=eid))
        elif require_contact_with:
            result = _and(result, in_contact(world, obj, require_contact_with, env_id=eid))
        if require_gripper_detached:
            result = _and(result, gripper_detached(world, obj, gripper_name, env_id=eid))
        return result

    if env_id is None:
        # Batched path used by IsaacLab TerminationManager.
        currently = {obj: _currently_placed(obj, None) for obj in objects}

        # Latch first-ever placement per env. torch.where keeps existing latches.
        for obj in objects:
            newly = (first_placed[obj] < 0) & currently[obj]
            if newly.any():
                first_placed[obj] = torch.where(newly, ep_step.to(first_placed[obj].dtype), first_placed[obj])

        all_in = torch.stack([currently[obj] for obj in objects], dim=0).all(dim=0)
        ordered = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for a, b in zip(objects[:-1], objects[1:]):
            both_latched = (first_placed[a] >= 0) & (first_placed[b] >= 0)
            if strict:
                cmp = first_placed[a] < first_placed[b]
            else:
                cmp = first_placed[a] <= first_placed[b]
            ordered &= both_latched & cmp

        result = all_in & ordered
    else:
        currently = {obj: _currently_placed(obj, env_id) for obj in objects}

        for obj in objects:
            if first_placed[obj][env_id].item() < 0 and bool(currently[obj]):
                first_placed[obj][env_id] = int(ep_step[env_id].item())

        all_in = all(bool(currently[obj]) for obj in objects)
        ordered = True
        for a, b in zip(objects[:-1], objects[1:]):
            fa = int(first_placed[a][env_id].item())
            fb = int(first_placed[b][env_id].item())
            if fa < 0 or fb < 0:
                ordered = False
                break
            if strict:
                if not (fa < fb):
                    ordered = False
                    break
            else:
                if not (fa <= fb):
                    ordered = False
                    break
        result = bool(all_in and ordered)

    if robolab.constants.DEBUG:
        print(f"objects_placed_in_container_in_order: {objects} -> '{container}' (strict={strict}) -> {result}")
    return result
