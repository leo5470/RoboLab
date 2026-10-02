"""Eligibility audit and physics loop for the narrowly supported OMY held target."""

import importlib.metadata
import math


def eligibility_reasons(env):
    """Fail closed: new consumers require an explicit audit before admission."""
    from isaaclab.actuators import ImplicitActuator
    from isaaclab.envs.mdp import BinaryJointPositionAction, JointPositionAction, time_out
    from isaaclab.sensors import ContactSensor, FrameTransformer

    from robolab.core.events.basic_recorders import (
        InitialStateRecorder, PostStepStatesRecorder, PreStepActionsRecorder, PreStepFlatPolicyObservationsRecorder,
    )
    from robolab.core.events.subtask_recorder import SubtaskCompletionRecorderTerm
    from robolab.core.task.conditionals import object_in_container
    from robolab.robots.droid import arm_joint_pos, gripper_pos

    from .options import ARM_JOINTS
    from .recording import BoundaryTimeRecorder

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
    if getattr(env.cfg, "_task_name", None) != "BananaInBowlTask":
        reasons.append("task predicates and consumers have not been audited")
    if getattr(env.cfg, "omy_substep_diagnostics", False):
        reasons.append("substep diagnostics require ordinary scene updates")
    if not env.cfg.sim.use_fabric:
        reasons.append("fresh scheduled rendering requires Fabric")
    actions = env.action_manager
    if actions.total_action_dim != 8 or set(actions.active_terms) != {"body", "finger_joint"}:
        reasons.append("requires the seven absolute Franka targets and binary gripper")
    else:
        arm, finger = actions.get_term("body"), actions.get_term("finger_joint")
        if (type(arm) is not JointPositionAction or tuple(arm._joint_names) != ARM_JOINTS
                or arm.cfg.scale != 1.0 or arm.cfg.offset != 0.0 or arm.cfg.use_default_offset
                or type(finger) is not BinaryJointPositionAction or tuple(finger._joint_names) != ("finger_joint",)
                or finger.cfg.open_command_expr != {"finger_joint": 0.0}
                or finger.cfg.close_command_expr != {"finger_joint": math.pi / 4}):
            reasons.append("action mapping differs from the audited absolute OMY contract")
    if set(env.scene.articulations) != {"robot"} or env.scene.deformable_objects:
        reasons.append("scene articulation/deformable topology is not audited")
    robot = env.scene["robot"]
    if not robot.actuators or any(type(act) is not ImplicitActuator for act in robot.actuators.values()):
        reasons.append("all actuators must be plain implicit actuators")
    reasons.extend(runtime_reasons(env))
    if "interval" in env.event_manager.available_modes:
        reasons.append("interval events require ordinary stepping")
    for name in ("reward_manager", "command_manager", "curriculum_manager"):
        if getattr(env, name).active_terms:
            reasons.append(f"{name} consumers have not been audited")
    for name, sensor in env.scene.sensors.items():
        if type(sensor) not in (ContactSensor, FrameTransformer):
            reasons.append(f"sensor {name} is not audited")
        elif getattr(sensor.cfg, "history_length", 0) or getattr(sensor.cfg, "track_air_time", False):
            reasons.append(f"sensor {name} needs substep history/air-time updates")
    observations = env.observation_manager
    if observations.active_terms != {"proprio_obs": ["arm_joint_pos", "gripper_pos"]}:
        reasons.append("observation consumers require the compact arm/gripper contract")
    else:
        for cfg in observations._group_obs_term_cfgs["proprio_obs"]:
            if (cfg.func not in (arm_joint_pos, gripper_pos) or cfg.history_length
                    or cfg.modifiers or getattr(cfg, "delay_min", 0) or getattr(cfg, "delay_max", 0)):
                reasons.append("observation history/modifiers/delays are not audited")
    allowed_recorders = (InitialStateRecorder, PostStepStatesRecorder, PreStepActionsRecorder,
                        PreStepFlatPolicyObservationsRecorder, SubtaskCompletionRecorderTerm, BoundaryTimeRecorder)
    if any(type(term) not in allowed_recorders for term in env.recorder_manager._terms.values()):
        reasons.append("recorder consumers require ordinary stepping")
    for name in env.termination_manager.active_terms:
        cfg = env.termination_manager.get_term_cfg(name)
        if (name, cfg.func) not in (("time_out", time_out), ("success", object_in_container)):
            reasons.append(f"termination {name} is not audited")
    return reasons


def runtime_reasons(env):
    assets = (*env.scene.articulations.values(), *env.scene.rigid_objects.values())
    return ["external wrench requires submission on every physics substep"] if any(
        getattr(asset, "has_external_wrench", False) for asset in assets
    ) else []


def advance_physics(env, *, held_target):
    """Keep physics/render opportunities; advance lazy scene caches at the boundary."""
    rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    for index in range(env.cfg.decimation):
        env._sim_step_counter += 1
        if not held_target or index == 0:
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if rendering and env._sim_step_counter % env.cfg.sim.render_interval == 0:
            env.sim.render()
        if not held_target:
            env.scene.update(dt=env.physics_dt)
    if held_target:
        env.scene.update(dt=env.step_dt)
