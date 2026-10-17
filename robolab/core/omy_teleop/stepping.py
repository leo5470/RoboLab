"""Independent OMY capabilities and a fixed-cadence manager physics loop."""

import importlib.metadata
import math

from .requirements import add_callable, build_task_requirements, default_recorder_types, default_registry


def execution_plan(env):
    """Audit consumers/topology, not task names. Disable only affected features."""
    from isaaclab.actuators import ImplicitActuator
    from isaaclab.envs.mdp import BinaryJointPositionAction, JointPositionAction
    from isaaclab.sensors import ContactSensor, FrameTransformer

    from robolab.robots.droid import arm_joint_pos, gripper_pos

    from .options import ARM_JOINTS
    reasons = []
    if importlib.metadata.version("isaaclab") != "2.2.0":
        reasons.append("manager lifecycle is audited only for Isaac Lab 2.2.0")
    if env.num_envs != 1:
        reasons.append("requires exactly one environment")
    if not str(env.device).startswith("cuda"):
        reasons.append("requires CUDA physics")
    if not math.isclose(env.physics_dt, 1 / 120) or env.cfg.decimation != 8:
        reasons.append("requires eight fixed 120 Hz substeps per native-15 command")
    if env.cfg.sim.render_interval not in (2, 4):
        reasons.append("render interval is not audited")
    target_reasons = []
    actions = env.action_manager
    if actions.total_action_dim != 8 or set(actions.active_terms) != {"body", "finger_joint"}:
        target_reasons.append("requires the seven absolute Franka targets and binary gripper")
    else:
        arm, finger = actions.get_term("body"), actions.get_term("finger_joint")
        if (type(arm) is not JointPositionAction or tuple(arm._joint_names) != ARM_JOINTS
                or arm.cfg.scale != 1.0 or arm.cfg.offset != 0.0 or arm.cfg.use_default_offset
                or type(finger) is not BinaryJointPositionAction or tuple(finger._joint_names) != ("finger_joint",)
                or finger.cfg.open_command_expr != {"finger_joint": 0.0}
                or finger.cfg.close_command_expr != {"finger_joint": math.pi / 4}):
            target_reasons.append("action mapping differs from the audited absolute OMY contract")
    if set(env.scene.articulations) != {"robot"} or env.scene.deformable_objects:
        reasons.append("scene articulation/deformable topology is not audited")
    robot = env.scene["robot"]
    if not robot.actuators or any(type(act) is not ImplicitActuator for act in robot.actuators.values()):
        target_reasons.append("all actuators must be plain implicit actuators")
    target_reasons.extend(runtime_reasons(env))
    if "interval" in env.event_manager.available_modes:
        reasons.append("interval events require ordinary stepping")
    for name in ("reward_manager", "command_manager", "curriculum_manager"):
        if getattr(env, name).active_terms:
            reasons.append(f"{name} consumers have not been audited")
    registry = default_registry()
    requirements = build_task_requirements(env.cfg, registry=registry)
    # Managers copy term configs. Audit their realized callables as well.
    for name in env.termination_manager.active_terms:
        cfg = env.termination_manager.get_term_cfg(name)
        add_callable(requirements, cfg.func, cfg.params, f"termination/{name}", registry)
    boundary_reasons = list(dict.fromkeys(requirements.unknown))
    if getattr(env.cfg, "omy_substep_diagnostics", False):
        boundary_reasons.append("substep diagnostics require ordinary scene updates")
    if not env.cfg.sim.use_fabric:
        boundary_reasons.append("fresh scheduled rendering requires Fabric")
    for name, sensor in env.scene.sensors.items():
        if type(sensor) not in (ContactSensor, FrameTransformer):
            boundary_reasons.append(f"sensor {name} is not audited")
        elif getattr(sensor.cfg, "history_length", 0) or getattr(sensor.cfg, "track_air_time", False):
            boundary_reasons.append(f"sensor {name} needs substep history/air-time updates")
    observation_reasons = []
    observations = env.observation_manager
    if observations.active_terms != {"proprio_obs": ["arm_joint_pos", "gripper_pos"]}:
        observation_reasons.append("observation consumers require the compact arm/gripper contract")
    else:
        for cfg in observations._group_obs_term_cfgs["proprio_obs"]:
            if (cfg.func not in (arm_joint_pos, gripper_pos) or cfg.history_length
                    or cfg.modifiers or getattr(cfg, "delay_min", 0) or getattr(cfg, "delay_max", 0)):
                observation_reasons.append("observation history/modifiers/delays are not audited")
    allowed_recorders, _ = default_recorder_types()
    if any(type(term) not in allowed_recorders for term in env.recorder_manager._terms.values()):
        reasons.append("recorder consumers require ordinary stepping")
    # Full observations or temporal modifiers retain the ordinary update cadence.
    boundary_reasons.extend(observation_reasons)
    requested = getattr(env.cfg, "omy_stepping_profile", "ordinary") == "held-target"
    compact = getattr(env.cfg, "omy_recording_profile", "full") == "compact"
    enabled = requested or compact
    failures = {
        "manager_loop": reasons,
        "held_targets": reasons + target_reasons,
        "boundary_scene_updates": reasons + target_reasons + boundary_reasons,
        "compact_observations": reasons + observation_reasons,
        "boundary_queries": reasons + list(dict.fromkeys(requirements.unknown)),
        "shared_event_queries": reasons + list(dict.fromkeys(requirements.unknown)),
    }
    result = {
        "version": 1,
        "features": {
            "held_targets": requested and not failures["held_targets"],
            "boundary_scene_updates": requested and not failures["boundary_scene_updates"],
            "compact_observations": compact and not failures["compact_observations"],
            "boundary_queries": enabled and not failures["boundary_queries"],
            "shared_event_queries": enabled and requirements.event_tracking and not failures["shared_event_queries"],
        },
        "fallback_reasons": {key: list(dict.fromkeys(values)) for key, values in failures.items()},
        "dependencies": requirements.report(),
    }
    return result


def eligibility_reasons(env):
    """Compatibility diagnostic: union reasons, not an all-or-nothing admission gate."""
    plan = execution_plan(env)
    return list(dict.fromkeys(reason for values in plan["fallback_reasons"].values() for reason in values))


def runtime_reasons(env):
    assets = (*env.scene.articulations.values(), *env.scene.rigid_objects.values())
    return ["external wrench requires submission on every physics substep"] if any(
        getattr(asset, "has_external_wrench", False) for asset in assets
    ) else []


def advance_physics(env, *, held_target, boundary_updates=None):
    """Keep physics/render opportunities; advance lazy scene caches at the boundary."""
    rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    if boundary_updates is None:
        boundary_updates = held_target
    for index in range(env.cfg.decimation):
        env._sim_step_counter += 1
        if not held_target or index == 0:
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if rendering and env._sim_step_counter % env.cfg.sim.render_interval == 0:
            env.sim.render()
        if not boundary_updates:
            env.scene.update(dt=env.physics_dt)
    if boundary_updates:
        env.scene.update(dt=env.step_dt)
