"""Production optimization behavior without opening hardware or starting Kit."""

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from robolab.core.omy_teleop.boundary import boundary_queries
from robolab.core.omy_teleop.collection import build_parser, validate_args
from robolab.core.omy_teleop.collection_options import run_contract
from robolab.core.omy_teleop import finger_colliders
from robolab.core.omy_teleop.layout_session import LayoutSession


def parsed(*arguments):
    parser = build_parser()
    args = parser.parse_args(["--desktop", "--source", "tcp", *arguments])
    validate_args(parser, args)
    return args


def test_new_collector_defaults():
    args = parsed()
    assert args.device == "cuda:0"
    assert args.performance_profile == "standard"
    assert (args.solver_position_iterations, args.gpu_max_num_partitions) == (32, 2)
    assert (args.placement_checks, args.finger_colliders) == ("shared", "original")
    assert (args.physics_workers, args.cpu_workers, args.cuda_connections, args.task_checks) == (0, 8, 1, "cpu")
    assert (args.control_hz, args.render_interval, args.stepping_profile, args.recording_profile) == (
        15, 4, "held-target", "compact")


@pytest.mark.parametrize("schema", ("current", "before_latency", "before_partitions", "before_optimizations"))
def test_old_resume_preserves_exact_contract_and_independent_placement(tmp_path, schema):
    args = parsed("--object-id", "034", "--solver-position-iterations", "16",
                  "--gpu-max-num-partitions", "8", "--placement-checks", "independent")
    original = run_contract(args)
    if schema != "current":
        for name in ("task_checks", "cuda_connections", "physics_workers", "cpu_workers"):
            del original["options"][name]
    if schema not in ("current", "before_latency"):
        for name in ("placement_checks", "finger_colliders"):
            del original["options"][name]
    if schema == "before_partitions":
        del original["options"]["gpu_max_num_partitions"]
    LayoutSession.create(tmp_path, args.layout_plan, 1, original).close()
    resumed = parsed("--resume", str(tmp_path))
    assert (resumed.solver_position_iterations, resumed.gpu_max_num_partitions) == (16, 8)
    assert (resumed.placement_checks, resumed.finger_colliders) == ("independent", "original")
    if schema != "current":
        assert (resumed.task_checks, resumed.cuda_connections, resumed.physics_workers, resumed.cpu_workers) == (
            "native", 8, 2, 8)
        for flag, value in (("--task-checks", "cpu"), ("--cuda-connections", "1"), ("--physics-workers", "0")):
            with pytest.raises(SystemExit):
                parsed("--resume", str(tmp_path), flag, value)
    assert run_contract(resumed) == original
    LayoutSession.resume(tmp_path, plan=resumed.layout_plan, run_metadata=run_contract(resumed)).close()
    with pytest.raises(SystemExit):
        parsed("--resume", str(tmp_path), "--placement-checks", "shared")
    with pytest.raises(SystemExit):
        parsed("--resume", str(tmp_path), "--finger-colliders", "hulls")


def test_hull_reference_is_part_of_resume_identity(tmp_path, monkeypatch):
    args = parsed("--object-id", "291", "--finger-colliders", "hulls")
    original = run_contract(args)
    assert original["finger_collider_reference"] == finger_colliders.finger_collider_identity()
    LayoutSession.create(tmp_path, args.layout_plan, 1, original).close()
    assert run_contract(parsed("--resume", str(tmp_path))) == original
    monkeypatch.setattr(finger_colliders, "finger_collider_identity", lambda: {"reference_sha256": "changed"})
    with pytest.raises(SystemExit):
        parsed("--resume", str(tmp_path))


def test_resume_keeps_an_unset_solver_cap(tmp_path):
    args = parsed("--object-id", "034")
    args.solver_position_iterations = None
    LayoutSession.create(tmp_path, args.layout_plan, 1, run_contract(args)).close()
    assert parsed("--resume", str(tmp_path)).solver_position_iterations is None


def test_resume_restores_custom_latency_profile_and_rejects_conflicts(tmp_path):
    args = parsed("--object-id", "034", "--physics-workers", "1", "--cpu-workers", "4",
                  "--cuda-connections", "2", "--task-checks", "native")
    original = run_contract(args)
    LayoutSession.create(tmp_path, args.layout_plan, 1, original).close()
    resumed = parsed("--resume", str(tmp_path))
    assert (resumed.physics_workers, resumed.cpu_workers, resumed.cuda_connections, resumed.task_checks) == (
        1, 4, 2, "native")
    assert run_contract(resumed) == original
    for flag, value in (("--physics-workers", "0"), ("--cpu-workers", "8"),
                        ("--cuda-connections", "1"), ("--task-checks", "cpu")):
        with pytest.raises(SystemExit):
            parsed("--resume", str(tmp_path), flag, value)


def test_hull_flag_rejects_cpu_before_simulator_startup():
    with pytest.raises(SystemExit):
        parsed("--device", "cpu", "--finger-colliders", "hulls")


@pytest.fixture
def runtime():
    path = Path(__file__).with_name("test_resting_in_container.py")
    spec = importlib.util.spec_from_file_location("production_placement_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.runtime.__wrapped__()


@pytest.mark.parametrize("shared", (False, True))
def test_production_cache_uses_one_evaluation_only_when_enabled(runtime, shared):
    world = runtime.World(count=1)
    world.env.cfg = NS(omy_shared_placement_checks=shared)
    calls = []
    original = world.get_or_init_predicate_state

    def count(*args):
        calls.append(args[0])
        return original(*args)

    world.get_or_init_predicate_state = count
    for step in range(10):
        world.env.episode_length_buf[:] = step
        with boundary_queries(world):
            batch = runtime.object_resting_in_container(world.env, object="target", container="bin")
            scalar = runtime.object_resting_in_container(world.env, object="target", container="bin", env_id=0)
            assert bool(batch) == scalar == (step >= 8)
    assert len(calls) == (10 if shared else 20)


def test_production_cache_reset_and_changed_parameters(runtime):
    world = runtime.World(count=1, dt=.1)
    world.env.cfg = NS(omy_shared_placement_checks=True)
    for step in range(6):
        world.env.episode_length_buf[:] = step
        with boundary_queries(world):
            runtime.check(world)
    with boundary_queries(world):
        assert bool(runtime.check(world))
        assert not bool(runtime.check(world, settle_time_s=1))
        world.reset_predicate_state([0])
        world._boundary_query_cache.clear()
        assert not bool(runtime.check(world))


def test_cached_results_cannot_be_mutated_by_a_consumer(runtime):
    world = runtime.World(count=1)
    world.env.cfg = NS(omy_shared_placement_checks=True)
    with boundary_queries(world):
        first = runtime.check(world)
        first[:] = True
        assert not bool(runtime.check(world))


def test_hulls_restore_only_finger_properties_and_default_buffers():
    reference = json.loads(finger_colliders.REFERENCE_PATH.read_text())
    names = ["base", *reference["bodies"]]

    class View:
        def __init__(self):
            self.values = {"masses": torch.full((1, 3), 2.),
                           "coms": torch.zeros(1, 3, 7), "inertias": torch.eye(3).flatten().repeat(1, 3, 1)}

        def __getattr__(self, name):
            operation, key = name.split("_", 1)
            if operation == "get":
                return lambda: self.values[key].clone()
            return lambda value, indices: self.values[key].copy_(value)

    view = View()
    base_before = {key: value[0, 0].clone() for key, value in view.values.items()}
    data = NS(default_mass=torch.zeros(1, 3), default_inertia=torch.zeros(1, 3, 9))
    robot = NS(body_names=names, root_physx_view=view, data=data)
    cfg = NS(omy_finger_colliders="hulls", omy_finger_collider_metadata=finger_colliders.finger_collider_identity())
    env = NS(cfg=cfg, scene={"robot": robot})
    finger_colliders.restore_finger_mass_properties(env)
    for key, values in view.values.items():
        assert torch.equal(values[0, 0], base_before[key])
        for name, body in reference["bodies"].items():
            torch.testing.assert_close(values[0, names.index(name)], torch.tensor(body[key]))
    assert cfg.omy_finger_collider_metadata["mass_properties_applied"]
    assert torch.equal(data.default_mass, view.values["masses"])
    assert torch.equal(data.default_inertia, view.values["inertias"])


@pytest.mark.parametrize("device,count,scale", (("cpu", 1, None), ("cuda:0", 2, None), ("cuda:0", 1, (1, 1, 1))))
def test_hull_profile_rejects_unsupported_models_before_spawning(device, count, scale):
    cfg = NS(sim=NS(device=device), scene=NS(num_envs=count, robot=NS(spawn=NS(scale=scale, mass_props=None))))
    with pytest.raises(ValueError):
        finger_colliders.configure_finger_colliders(cfg, "hulls")


def test_original_profile_is_a_noop_for_standard_robot():
    cfg = NS()
    finger_colliders.configure_finger_colliders(cfg, "original")
    assert cfg.omy_finger_collider_metadata == {"profile": "original"}
    finger_colliders.restore_finger_mass_properties(NS(cfg=cfg))


@pytest.mark.parametrize("damage", ("asset", "version", "mass_override"))
def test_hull_reference_rejects_changed_asset_stack_and_mass_override(tmp_path, monkeypatch, damage):
    asset = tmp_path / "robot.usd"
    asset.write_bytes(b"reference robot")
    reference = json.loads(finger_colliders.REFERENCE_PATH.read_text())
    reference["asset_sha256"] = hashlib.sha256(asset.read_bytes()).hexdigest()
    path = tmp_path / "mass.json"
    path.write_text(json.dumps(reference))
    monkeypatch.setattr(finger_colliders, "REFERENCE_PATH", path)
    monkeypatch.setattr(finger_colliders.importlib.metadata, "version",
                        lambda name: "different" if damage == "version" else reference["stack"][name])
    spawn = NS(scale=None, mass_props=None, usd_path=str(asset), func=lambda *a, **k: None)
    cfg = NS(sim=NS(device="cuda:0"), scene=NS(num_envs=1, robot=NS(spawn=spawn)))
    if damage == "asset":
        asset.write_bytes(b"changed robot")
    if damage == "mass_override":
        spawn.mass_props = NS(mass=1)
    with pytest.raises(ValueError):
        finger_colliders.configure_finger_colliders(cfg, "hulls")
