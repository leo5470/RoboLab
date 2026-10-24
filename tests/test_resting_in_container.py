"""Observed placement checks on controlled trajectories, without booting Isaac Sim.

Execute the production function bodies with a small WorldState-shaped input.
Only simulator imports and USD mesh loading are bypassed. Containment, support,
velocity checks, per-env state/reset, and subtask progression use the real code.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import math
from functools import partial, reduce
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

import robolab.constants
from robolab.core.omy_teleop.requirements import TaskRequirements, add_callable, predicate_registry
from robolab.core.task.decorators import atomic, composite
from robolab.core.task.hull_check import build_local_hull, point_in_hull
from robolab.core.task.status import StatusCode
from robolab.core.utils.function_loader import func_as_str, get_callable_info

ROOT = Path(__file__).resolve().parents[1]


def load_definitions(relative, namespace, names=None, *, methods_of=None):
    path = ROOT / relative
    nodes = ast.parse(path.read_text()).body
    if methods_of:
        nodes = next(node.body for node in nodes if isinstance(node, ast.ClassDef) and node.name == methods_of)
    definitions = [node for node in nodes if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                   and (names is None or node.name in names)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])), str(path), "exec"), namespace)


def quat_apply(quat, vector):
    # Independent rotation formula for the CPU-only WorldState input.
    cross = 2 * torch.cross(quat[..., 1:], vector, dim=-1)
    return vector + quat[..., :1] * cross + torch.cross(quat[..., 1:], cross, dim=-1)


def box_hull(dimensions):
    x, y, z = np.array(dimensions) / 2
    return build_local_hull(np.array([[a, b, c] for a in (-x, x) for b in (-y, y) for c in (-z, z)]))


@pytest.fixture
def runtime():
    namespace = {
        "np": np, "torch": torch, "math": math, "partial": partial, "reduce": reduce,
        "robolab": robolab, "DEBUG": False, "atomic": atomic, "composite": composite,
        "get_world": lambda env: env.world, "point_in_hull": point_in_hull,
        "quat_apply": quat_apply,
        "quat_apply_inverse": lambda quat, vector: quat_apply(quat * torch.tensor([1, -1, -1, -1]), vector),
        "StatusCode": StatusCode, "func_as_str": func_as_str, "get_callable_info": get_callable_info,
    }
    load_definitions("robolab/core/task/predicate_logic.py", namespace, {
        "evaluate_logicals", "evaluate_logicals_vectorized", "evaluate_spatial_condition", "_and", "_not",
        "_untransformed_hull", "_obj_centroid_in_container", "in_opentop_container", "stationary", "upright",
        "in_contact", "gripper_detached",
    })
    load_definitions("robolab/core/task/subtask_utils.py", namespace,
                     {"sanitize_subtask_conditions", "normalize_conditions_scores"})
    from dataclasses import dataclass
    from typing import Callable, cast
    namespace.update(dataclass=dataclass, cast=cast, Callable=Callable)
    load_definitions("robolab/core/task/subtask.py", namespace)
    load_definitions("robolab/core/task/conditionals.py", namespace, {
        "pick_and_place", "object_in_container", "object_grabbed", "object_resting_in_container",
        "_condition_held_for_time",
        "resting_in_container_diagnostics",
    })
    load_definitions("robolab/core/task/conditionals_state_machine.py", namespace)
    methods = {}
    load_definitions("robolab/core/world/world_state.py", methods,
                     {"get_or_init_predicate_state", "reset_predicate_state", "is_supported_on_surface"},
                     methods_of="WorldState")
    methods.update(torch=torch, math=math)

    class World:
        def __init__(self, count=3, dt=1 / 15):
            self.env = NS(num_envs=count, device="cpu", step_dt=dt,
                          episode_length_buf=torch.zeros(count, dtype=torch.long), world=self)
            self._predicate_state = {}
            self._untransformed_hull_cache = {"target": box_hull([.047, .047, .047]),
                                             "bin": box_hull([.23, .17, .13])}
            self.positions = {"target": torch.tensor([[0, 0, -.04]]).repeat(count, 1),
                              "bin": torch.zeros(count, 3)}
            self.orientations = {name: torch.tensor([[1., 0, 0, 0]]).repeat(count, 1)
                                 for name in self.positions}
            self.velocities = {name: torch.zeros(count, 6) for name in self.positions}
            self.forces = {frozenset(("target", "bin")): torch.tensor([[0., 0, .5]]).repeat(count, 1)}

        get_or_init_predicate_state = methods["get_or_init_predicate_state"]
        reset_predicate_state = methods["reset_predicate_state"]
        is_supported_on_surface = methods["is_supported_on_surface"]

        def select(self, values, env_id):
            return values if env_id is None else values[env_id]

        def get_pose(self, name, env_id=None):
            return self.select(self.positions[name], env_id), self.select(self.orientations[name], env_id)

        def get_velocity(self, name, env_id=None):
            return self.select(self.velocities[name], env_id)

        def get_contact_force(self, first, second, env_id=None):
            values = self.forces.get(frozenset((first, second)), torch.zeros(self.env.num_envs, 3))
            return self.select(values, env_id)

        def in_contact(self, first, second, force_threshold=.1, env_id=None):
            return (self.get_contact_force(first, second, env_id).abs() > force_threshold).any(dim=-1)

    def check(world, **params):
        return namespace["object_resting_in_container"](
            world.env, object="target", container="bin", exclude_contact_with="table", **params)

    return NS(**namespace, World=World, check=check)


def advance(world, steps=1):
    world.env.episode_length_buf += steps


def test_waits_half_second_and_duplicate_calls_do_not_advance_clock(runtime):
    world = runtime.World()
    for _ in range(8):
        assert not runtime.check(world).any()  # steps 0..7 span less than .5 seconds
        for eid in range(world.env.num_envs):
            assert runtime.check(world, env_id=eid) is False
        advance(world)
    assert runtime.check(world).tolist() == [True] * 3  # first sample at or after .5 s
    for eid in range(3):
        assert runtime.check(world, env_id=eid) is True


@pytest.mark.parametrize("mode", ["falling", "sliding", "spinning", "held", "table", "unsupported",
                                  "side_contact", "outside", "tilted_bin", "moving_bin", "spinning_bin"])
def test_bad_placement_never_succeeds_and_resets_settling_window(runtime, mode):
    world = runtime.World(dt=.1)
    for _ in range(5):
        assert not runtime.check(world).any()
        advance(world)
    if mode in ("falling", "sliding", "spinning"):
        world.velocities["target"][:, {"falling": 2, "sliding": 0, "spinning": 5}[mode]] = 1
    elif mode in ("held", "table"):
        other = "gripper" if mode == "held" else "table"
        world.forces[frozenset(("target", other))] = torch.ones(3, 3)
    elif mode == "unsupported":
        world.forces[frozenset(("target", "bin"))].zero_()
    elif mode == "side_contact":
        world.forces[frozenset(("target", "bin"))][:] = torch.tensor([.5, 0, 0])
    elif mode == "outside":
        world.positions["target"][:, 0] = .3
    elif mode == "tilted_bin":
        world.orientations["bin"][:] = torch.tensor([math.cos(.5), math.sin(.5), 0, 0])
    elif mode in ("moving_bin", "spinning_bin"):
        world.velocities["bin"][:, 0 if mode == "moving_bin" else 5] = 1
    for _ in range(10):
        assert not runtime.check(world).any()
        assert runtime.check(world, env_id=0) is False
        advance(world)
    # Restore the inputs without clearing production state: a fresh half-second
    # must still be observed, even after a previous nearly-complete window.
    reference = runtime.World(dt=.1)
    world.velocities, world.forces = reference.velocities, reference.forces
    world.positions, world.orientations = reference.positions, reference.orientations
    for _ in range(5):
        assert not runtime.check(world).any()
        advance(world)
    assert runtime.check(world).all()


def test_envs_and_partial_resets_are_independent(runtime):
    world = runtime.World(dt=.1)
    for _ in range(6):
        runtime.check(world)
        advance(world)
    assert runtime.check(world).all()
    world.reset_predicate_state([1])
    world.env.episode_length_buf[1] = 0
    assert runtime.check(world).tolist() == [True, False, True]
    for _ in range(5):
        advance(world)
        runtime.check(world)
    assert runtime.check(world).all()
    world.velocities["target"][2, 5] = 1
    advance(world)
    assert runtime.check(world).tolist() == [True, True, False]


def test_unobserved_steps_do_not_count_as_rest(runtime):
    world = runtime.World(dt=.1)
    assert not runtime.check(world).any()
    advance(world, 20)
    assert not runtime.check(world).any()
    for _ in range(5):
        advance(world)
        result = runtime.check(world)
    assert result.all()


def test_scalar_first_and_batched_first_have_identical_timing(runtime):
    world = runtime.World(dt=.1)
    for step in range(6):
        scalar = [runtime.check(world, env_id=eid) for eid in range(3)]
        batch = runtime.check(world).tolist()
        assert scalar == batch == [step >= 5] * 3
        advance(world)


def test_different_thresholds_do_not_share_a_timer(runtime):
    world = runtime.World(dt=.1)
    for _ in range(6):
        runtime.check(world)
        advance(world)
    assert runtime.check(world).all()
    assert not runtime.check(world, settle_time_s=1).any()


@pytest.mark.parametrize("body", ["target", "bin"])
def test_contact_solver_velocity_noise_with_stable_pose_can_settle(runtime, body):
    world = runtime.World(dt=.1)
    for step in range(6):
        world.velocities[body][:, 5] = .15 if step % 2 else -.28
        # The fast OMY solver reported up to .038 m/s for a supported bowl
        # whose pose varied by less than .05 mm across eight seconds.
        world.velocities[body][:, 0] = .038
        world.positions[body][:, 0] = .00004 if step % 2 else 0
        angle = .0001 if step % 2 else 0
        world.orientations[body][:] = torch.tensor([math.cos(angle / 2), 0, 0, math.sin(angle / 2)])
        assert runtime.check(world).tolist() == [step >= 5] * 3
        advance(world)


def test_diagnostics_identify_blocking_checks_without_mutating_timer(runtime):
    world = runtime.World(count=1, dt=.1)
    world.forces[frozenset(("target", "gripper"))] = torch.ones(1, 3)
    assert not runtime.check(world).any()
    state = next(iter(world._predicate_state.values()))
    before = {key: value.clone() for key, value in state.items() if isinstance(value, torch.Tensor)}
    info = runtime.resting_in_container_diagnostics(
        world.env, {"object": "target", "container": "bin", "exclude_contact_with": "table"})
    assert info["blocked"] == ["gripper_detached"]
    assert info["checks"]["inside"] and info["observed_s"] == 0
    assert info["evaluated_step"] == 0
    for key, value in before.items():
        assert torch.equal(state[key], value)
    world.forces[frozenset(("target", "gripper"))].zero_()
    for _ in range(4):
        advance(world)
        runtime.check(world)
    info = runtime.resting_in_container_diagnostics(world.env, {"object": "target", "container": "bin",
                                                              "exclude_contact_with": "table"})
    assert info["blocked"] == []
    assert info["observed_s"] == pytest.approx(.3)
    variant = runtime.resting_in_container_diagnostics(
        world.env, {"object": "target", "container": "bin", "exclude_contact_with": "table", "settle_time_s": 1})
    assert variant["observed_s"] == 0 and "evaluated_step" not in variant
    assert len(world._predicate_state) == 1


@pytest.mark.parametrize("body", ["target", "bin"])
@pytest.mark.parametrize("mode,speed", [("translation", .005), ("translation", .03), ("rotation", .1)])
def test_slow_pose_drift_restarts_window_even_below_velocity_caps(runtime, body, mode, speed):
    world = runtime.World(dt=.1)
    for step in range(20):
        if mode == "translation":
            world.positions[body][:, 0] = step * speed * .1
            world.velocities[body][:, 0] = speed
        else:
            angle = step * .01
            world.orientations[body][:] = torch.tensor([math.cos(angle / 2), 0, 0, math.sin(angle / 2)])
            world.velocities[body][:, 5] = .1
        assert not runtime.check(world).any()
        advance(world)


def test_quaternion_sign_changes_do_not_restart_window(runtime):
    world = runtime.World(dt=.1)
    for step in range(6):
        world.orientations["target"] *= -1
        assert runtime.check(world).tolist() == [step >= 5] * 3
        advance(world)


def test_protruding_and_rim_spanning_objects_can_settle(runtime):
    world = runtime.World(dt=.1)
    # The center is above the rim, with the lower end of a tall object inserted.
    world._untransformed_hull_cache["target"] = box_hull([.05, .05, .28])
    world.positions["target"][:, 2] = .08
    for _ in range(5):
        assert not runtime.check(world).any()
        advance(world)
    assert runtime.check(world).all()
    # A long tool supported across the rim follows the documented same rule.
    world.reset_predicate_state([0, 1, 2])
    world._untransformed_hull_cache["target"] = box_hull([.28, .05, .01])
    for _ in range(5):
        assert not runtime.check(world).any()
        advance(world)
    assert runtime.check(world).all()


def test_light_support_and_gripper_list(runtime):
    world = runtime.World(dt=.1)
    world.forces[frozenset(("target", "bin"))][:, 2] = .03
    world.forces[frozenset(("target", "right"))] = torch.ones(3, 3)
    for _ in range(6):
        assert not runtime.check(world, gripper_name=["left", "right"]).any()
        advance(world)
    world.forces[frozenset(("target", "right"))].zero_()
    for _ in range(5):
        assert not runtime.check(world, gripper_name=["left", "right"]).any()
        advance(world)
    assert runtime.check(world, gripper_name=["left", "right"]).all()


@pytest.mark.parametrize("parameter,value", [("settle_time_s", 0), ("settle_time_s", float("nan")),
                                           ("linear_threshold", -1), ("angular_threshold", float("inf")),
                                           ("support_force_threshold", 0), ("max_container_tilt_deg", 90)])
def test_invalid_parameters_fail_before_creating_state(runtime, parameter, value):
    world = runtime.World()
    with pytest.raises(ValueError):
        runtime.check(world, **{parameter: value})
    assert not world._predicate_state


def test_subtask_credit_and_termination_agree(runtime, monkeypatch):
    monkeypatch.setattr(robolab.constants, "VERBOSE", False)
    world = runtime.World(count=1, dt=.1)
    subtask = runtime.pick_and_place(
        object="target", container="bin", placement_func=runtime.object_resting_in_container,
        placement_params={"exclude_contact_with": "table"})
    sm = runtime.ConditionalsStateMachine(world.env, subtask=subtask)
    for step in range(6):
        terminal = bool(runtime.check(world)[0])
        done, *_ = sm.step()
        assert done == terminal == (step >= 5)
        assert sm.total_score == float(terminal)
        advance(world)
    assert StatusCode.subtask_to_success("object_resting_in_container") == StatusCode.OBJECT_PLACED_IN_CONTAINER_SUCCESS
    assert StatusCode.subtask_to_error("object_resting_in_container") == StatusCode.OBJECT_PLACED_IN_CONTAINER_FAILURE


def test_default_pick_and_place_keeps_legacy_parameters(runtime):
    subtask = runtime.pick_and_place(object="target", container="bin")
    place, _ = subtask.conditions["target"][-1]
    assert place.func is runtime.object_in_container
    assert place.keywords == {"object": "target", "container": "bin", "require_contact_with": False,
                              "require_gripper_detached": True}


def test_requirements_keep_support_table_gripper_and_velocity_queries(runtime):
    req = TaskRequirements()
    add_callable(req, runtime.object_resting_in_container,
                 {"object": "target", "container": "bin", "exclude_contact_with": "table"},
                 "success", predicate_registry(runtime))
    assert not req.unknown
    assert req.poses == {"target", "bin"}
    assert req.velocities == {"target", "bin"}
    assert req.contacts == {("bin", "target"), ("gripper", "target"), ("table", "target")}
    assert req.force_contacts == {("bin", "target")}


@pytest.mark.parametrize("logical,K,expected", [("all", 1, False), ("any", 1, True), ("choose", 1, True),
                                               ("choose", 2, False)])
def test_each_object_has_its_own_settling_window(runtime, logical, K, expected):
    world = runtime.World(count=1, dt=.1)
    world._untransformed_hull_cache["other"] = box_hull([.04, .04, .04])
    world.positions["other"] = torch.zeros(1, 3)
    world.orientations["other"] = torch.tensor([[1., 0, 0, 0]])
    world.velocities["other"] = torch.ones(1, 6)
    world.forces[frozenset(("other", "bin"))] = torch.tensor([[0., 0, .5]])
    for _ in range(6):
        result = runtime.object_resting_in_container(world.env, object=["target", "other"], container="bin",
                                                     logical=logical, K=K)
        advance(world)
    assert result.item() == expected


def load_task_declaration(path, runtime):
    classes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)]
    for node in classes:
        node.decorator_list = []
    namespace = {**vars(runtime), "Task": type("Task", (), {}), "DoneTerm": NS,
                 "mdp": NS(time_out=lambda env: None), "import_scene": lambda *args: None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])), str(path), "exec"), namespace)
    task_cls = next(namespace[node.name] for node in classes
                    if any(isinstance(base, ast.Name) and base.id == "Task" for base in node.bases))
    return task_cls


def test_generated_task_success_and_subtask_use_identical_resting_parameters(runtime, tmp_path):
    spec = importlib.util.spec_from_file_location("simple_pnp_generator", ROOT / "tools/generate_simple_pnp_tasks.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    path = tmp_path / "task.py"
    path.write_text(generator.task_source("pnp_test.usda", "target"))
    task = load_task_declaration(path, runtime)
    termination = task.terminations.success
    placement, _ = task.subtasks[0].conditions["target"][-1]
    assert termination.func is placement.func is runtime.object_resting_in_container
    signature = inspect.signature(termination.func)
    success_args = signature.bind(None, **termination.params)
    subtask_args = signature.bind(None, **placement.keywords)
    success_args.apply_defaults()
    subtask_args.apply_defaults()
    assert success_args.arguments == subtask_args.arguments
    assert success_args.arguments["settle_time_s"] == .5
    assert success_args.arguments["exclude_contact_with"] == "table"


def test_generated_task_migration_preserves_filtering_and_refuses_custom_edits(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("simple_pnp_generator", ROOT / "tools/generate_simple_pnp_tasks.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    scenes, tasks = tmp_path / "scenes", tmp_path / "tasks"
    scenes.mkdir()
    tasks.mkdir()
    records = [{"scene": "pnp_a.usda", "target": {"name": "target"}},
               {"scene": "pnp_b.usda", "target": {"name": "other"}}]
    scene_manifest = scenes / "manifest.json"
    scene_manifest.write_text(json.dumps({"scenes": records}))
    metadata = {"tasks": [], "filtering": {"retained_task_count": 2, "provenance": "kept"}}
    for scene in records:
        name = scene["scene"]
        (scenes / name).write_text("#usda 1.0")
        task = Path(name).stem + "_task.py"
        metadata["tasks"].append({"task": task})
        (tasks / task).write_text(generator.task_source(name, scene["target"]["name"], resting=False))
    (tasks / "manifest.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(generator, "SCENE_DIR", scenes)
    monkeypatch.setattr(generator, "SCENE_MANIFEST", scene_manifest)
    monkeypatch.setattr(generator, "OUT_DIR", tasks)
    import sys
    monkeypatch.setattr(sys, "argv", ["generator", "--update-existing"])
    second = tasks / "pnp_b_task.py"
    second.write_text(second.read_text() + "\n# custom edit\n")
    with pytest.raises(ValueError, match="custom edits"):
        generator.main()
    # Preflight must leave the first task untouched when a later task fails.
    assert "func=object_in_container," in (tasks / "pnp_a_task.py").read_text()
    second.write_text(generator.task_source("pnp_b.usda", "other", resting=False))
    generator.main()
    result = json.loads((tasks / "manifest.json").read_text())
    assert result["filtering"] == metadata["filtering"]
    assert result["task_count"] == 2
    assert result["success_params"]["settle_time_s"] == .5
    assert all("func=object_resting_in_container," in (tasks / row["task"]).read_text()
               for row in result["tasks"])
    before = {path.name: path.read_bytes() for path in tasks.iterdir()}
    generator.main()
    assert before == {path.name: path.read_bytes() for path in tasks.iterdir()}


def test_committed_single_pnp_definitions_use_matching_settled_success_and_credit(runtime):
    root = ROOT / "robolab/tasks/single_pnp"
    manifest = json.loads((root / "manifest.json").read_text())
    rows = manifest["tasks"]
    assert len(rows) == manifest["task_count"]
    assert {row["task"] for row in rows} == {path.name for path in root.glob("*_task.py")}
    signature = inspect.signature(runtime.object_resting_in_container)
    for row in rows:
        task = load_task_declaration(root / row["task"], runtime)
        assert task.__name__ == row["class_name"], row["task"]
        assert task.episode_length_s == row["episode_length_s"] == 50
        assert task.contact_object_list == [row["target"], "grey_bin", "table"]
        assert (ROOT / "assets/scenes" / row["scene"]).is_file()
        termination = task.terminations.success
        placement, _ = task.subtasks[0].conditions[row["target"]][-1]
        assert termination.func is placement.func is runtime.object_resting_in_container
        success_args = signature.bind(None, **termination.params)
        subtask_args = signature.bind(None, **placement.keywords)
        success_args.apply_defaults()
        subtask_args.apply_defaults()
        assert success_args.arguments == subtask_args.arguments, row["task"]
        assert success_args.arguments["object"] == row["target"]
        assert success_args.arguments["container"] == row["destination"] == "grey_bin"
        for key, value in manifest["success_params"].items():
            assert success_args.arguments[key] == value, (row["task"], key)
