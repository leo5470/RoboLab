"""Source contracts across kinds: a scripted source is checked against a teleoperation target where the generator
reads it (robot model, action terms, control period, assets, success rule) and every other difference is recorded on
its reference; teleoperation sources keep the exact comparison."""

import copy
import json
from argparse import Namespace

import h5py
import numpy as np
import pytest

from robolab.core.omy_teleop.mimic_sources import source_references
from robolab.core.scripted_demos.writer import STATE_GROUP

ARM = [f"panda_joint{i}" for i in range(1, 8)]
JOINTS = ARM + ["finger_joint", "right_outer_knuckle_joint"]
TASK = "SimplePnp230MilkjugA0103Task"


def _cfg(robot_usd, **changes):
    """A recorded environment configuration in the teleoperation layout."""
    cfg = {
        "sim": {"dt": 1 / 120, "render_interval": 4, "physx": {"gpu_max_num_partitions": 2}},
        "decimation": 8,
        "episode_length_s": 30.0,
        "actions": {
            "body": {"class_type": "isaaclab.envs.mdp.actions.joint_actions:JointPositionAction", "joint_names": ARM,
                     "scale": 1.0, "offset": 0.0, "use_default_offset": False},
            "finger_joint": {"class_type": "isaaclab.envs.mdp.actions.binary_joint_actions:BinaryJointPositionAction",
                             "joint_names": ["finger_joint"], "open_command_expr": {"finger_joint": 0.0}},
        },
        "scene": {
            "robot": {"spawn": {"usd_path": str(robot_usd)}, "actuators": {"gripper": {"velocity_limit_sim": 5.0}}},
            "milkjug_a01": {"spawn": {"usd_path": "assets/objects/milkjug_a01.usd"}, "init_state": {"pos": [0.5, 0, 0]}},
            "grey_bin": {"spawn": {"usd_path": "assets/objects/grey_bin.usd"}, "init_state": {"pos": [0.4, -0.3, 0]}},
        },
        "terminations": {"success": {"func": "tests.no_such_module:success",
                                     "params": {"object": "milkjug_a01", "container": "grey_bin"}}},
        "omy_stepping_profile": "held-target", "omy_recording_profile": "compact", "omy_contact_sensor_profile": "required",
    }
    for path, value in changes.items():
        node = cfg
        *parents, leaf = path.split("__")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return cfg


def _teleop_run(directory, cfg):
    directory.mkdir(parents=True)
    (directory / "env_cfg.json").write_text(json.dumps(cfg))
    meta = {"task": "SimplePnp230MilkjugA0100Task", "action_type": "absolute_joint_position",
            "action_names": [*ARM, "gripper"], "joint_units": "rad", "gripper_encoding": "+1=open,-1=closed",
            "control_hz": 15.0, "performance": {"joint_order": JOINTS}}
    with h5py.File(directory / "omy_demos.hdf5", "w") as stream:
        stream.create_group("data").attrs["robolab_omy_teleop"] = json.dumps(meta)
        stream["data"].create_group("demo_0").attrs["success"] = True
    return directory


def _scripted_family(directory, cfg, with_state=True):
    """A scripted family: pps.hdf5 with one successful episode and RoboLab's env_cfg.json for its layout task."""
    (directory / "robolab_output" / TASK).mkdir(parents=True)
    (directory / "robolab_output" / TASK / "env_cfg.json").write_text(json.dumps(cfg))
    with h5py.File(directory / "pps.hdf5", "w") as stream:
        episode = stream.create_group("data/episode_000000")
        episode.attrs.update({"success": True, "layout_task": TASK})
        if with_state:
            episode[f"{STATE_GROUP}/articulation/robot/joint_names"] = np.array([n.encode() for n in JOINTS])
    return directory / "pps.hdf5"


def _scripted_cfg(robot_usd, **changes):
    """The stock DROID registration's configuration: the differences a real scripted collection shows."""
    cfg = _cfg(robot_usd, **changes)
    cfg["sim"]["render_interval"] = 8
    cfg["sim"]["physx"]["gpu_max_num_partitions"] = 8
    cfg["actions"]["body"]["joint_names"] = ["panda_joint.*"]
    cfg["actions"]["finger_joint"]["class_type"] = "robolab.robots.droid:BinaryJointPositionZeroToOneAction"
    cfg["scene"]["robot"]["actuators"]["gripper"] = {"velocity_limit": 5.0}
    for key in ("omy_stepping_profile", "omy_recording_profile", "omy_contact_sensor_profile"):
        del cfg[key]
    return cfg


@pytest.fixture()
def robot(tmp_path):
    """The same robot file in two checkouts."""
    usd = [tmp_path / "teleop_checkout" / "robot.usd", tmp_path / "scripted_checkout" / "robot.usd"]
    for path in usd:
        path.parent.mkdir()
        path.write_bytes(b"#usda 1.0 franka robotiq")
    return usd


def _references(tmp_path, target, rows):
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"sources": rows}))
    return source_references(Namespace(source_selection=str(manifest), input=[str(target)], episodes=None,
                                       target_input=str(target)))


def _scripted_row(path):
    return {"file": str(path), "episode": "episode_000000", "kind": "scripted"}


def test_scripted_source_is_accepted_with_its_differences_recorded(tmp_path, robot):
    target = _teleop_run(tmp_path / "teleop", _cfg(robot[0]))
    pps = _scripted_family(tmp_path / "scripted", _scripted_cfg(robot[1]))
    [ref] = _references(tmp_path, target, [_scripted_row(pps)])
    assert ref["task"] == TASK and ref["kind"] == "scripted" and len(ref["sha256"]) == 64
    recorded = ref["contract_differences"]
    assert {"sim.render_interval", "sim.physx.gpu_max_num_partitions", "profiles.omy_stepping_profile",
            "robot.actuators.gripper.velocity_limit", "robot.actuators.gripper.velocity_limit_sim",
            "actions.body.joint_names", "actions.finger_joint.class_type",
            "action_metadata.gripper_encoding"} <= set(recorded)
    assert recorded["sim.render_interval"] == {"target": 4, "source": 8}
    # The robot file is the same content in another checkout: neither refused nor recorded.
    assert "robot.spawn.usd_path" not in recorded


@pytest.mark.parametrize("change, blocking", [
    ({"decimation": 4}, "decimation"),
    ({"sim__dt": 1 / 60}, "sim.dt"),
    ({"scene__milkjug_a01__spawn": {"usd_path": "assets/objects/other.usd"}}, "assets"),
    ({"terminations__success__params": {"object": "milkjug_a01", "container": "grey_bin", "tolerance": 9}}, "success"),
    ({"actions__body__scale": 0.5}, "actions.body"),
])
def test_scripted_source_differing_where_the_generator_reads_it_is_refused(tmp_path, robot, change, blocking):
    target = _teleop_run(tmp_path / "teleop", _cfg(robot[0]))
    pps = _scripted_family(tmp_path / "scripted", _scripted_cfg(robot[1], **change))
    with pytest.raises(ValueError, match="Incompatible source") as error:
        _references(tmp_path, target, [_scripted_row(pps)])
    assert repr(blocking) in str(error.value)


def test_scripted_source_with_another_robot_model_is_refused(tmp_path, robot):
    target = _teleop_run(tmp_path / "teleop", _cfg(robot[0]))
    robot[1].write_bytes(b"#usda 1.0 another robot")
    pps = _scripted_family(tmp_path / "scripted", _scripted_cfg(robot[1]))
    with pytest.raises(ValueError, match="robot.spawn.usd_path"):
        _references(tmp_path, target, [_scripted_row(pps)])


def test_scripted_arm_pattern_resolving_to_other_joints_is_refused(tmp_path, robot):
    target = _teleop_run(tmp_path / "teleop", _cfg(robot[0]))
    cfg = _scripted_cfg(robot[1])
    cfg["actions"]["body"]["joint_names"] = ["panda_joint[1-6]"]
    pps = _scripted_family(tmp_path / "scripted", cfg)
    with pytest.raises(ValueError, match="actions.body.joint_names"):
        _references(tmp_path, target, [_scripted_row(pps)])


def test_scripted_episode_without_state_is_refused_before_the_simulator(tmp_path, robot):
    target = _teleop_run(tmp_path / "teleop", _cfg(robot[0]))
    pps = _scripted_family(tmp_path / "scripted", _scripted_cfg(robot[1]), with_state=False)
    with pytest.raises(ValueError, match="collect it again"):
        _references(tmp_path, target, [_scripted_row(pps)])


def test_teleop_sources_keep_the_exact_comparison(tmp_path, robot):
    target = _teleop_run(tmp_path / "teleop", _cfg(robot[0]))
    same = _teleop_run(tmp_path / "same", _cfg(robot[0]))
    [ref] = _references(tmp_path, target, [{"file": str(same / "omy_demos.hdf5"), "episode": "demo_0"}])
    assert "contract_differences" not in ref and ref["task"] == "SimplePnp230MilkjugA0100Task"
    # A render-interval difference a scripted source may have refuses a teleoperation source.
    other = _teleop_run(tmp_path / "other", _cfg(robot[0], sim__render_interval=8))
    with pytest.raises(ValueError, match="sim"):
        _references(tmp_path / "other", target, [{"file": str(other / "omy_demos.hdf5"), "episode": "demo_0"}])


def test_mixed_pool_of_both_kinds(tmp_path, robot):
    target = _teleop_run(tmp_path / "teleop", _cfg(robot[0]))
    pps = _scripted_family(tmp_path / "scripted", _scripted_cfg(robot[1]))
    rows = [{"file": str(target / "omy_demos.hdf5"), "episode": "demo_0"}, _scripted_row(pps)]
    teleop, scripted = _references(tmp_path, target, copy.deepcopy(rows))
    assert "kind" not in teleop and scripted["kind"] == "scripted"
    assert "contract_differences" not in teleop and scripted["contract_differences"]
