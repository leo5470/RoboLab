"""DROID relative-IK Mimic interface for Isaac Lab 2.2.

All Mimic poses use robot-root coordinates, including object poses. The runner
owns resets and episode acceptance: task success is queried explicitly and is
not installed as an auto-resetting termination term. This preserves the final
action/state and allows the demonstrated release/retreat to finish.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.utils import math as pose

from robolab.core.world.world_state import clear_world_cache, get_world
from robolab.datagen.audit import validate_controller


class DroidMimicEnv(ManagerBasedRLMimicEnv):
    eef_name = "droid"

    def __init__(self, cfg, **kwargs):
        validate_controller({"actions": cfg.actions.to_dict()})
        terms = [(name, term) for name, term in vars(cfg.terminations).items()
                 if getattr(term, "func", None) is not None and not getattr(term, "time_out", False)]
        if len(terms) != 1 or terms[0][0] != "success":
            raise ValueError("Mimic requires exactly one explicit success term; ambiguous task needs an override")
        self._success_term = terms[0][1]
        cfg = cfg.copy()
        cfg.terminations = type(cfg.terminations)()
        for name in vars(cfg.terminations):
            setattr(cfg.terminations, name, None)
        # Initial milestone records explicitly in the replay runner. Do not
        # install RoboLab's evaluation exporter or freeze-on-success lifecycle.
        cfg.recorders = None
        super().__init__(cfg, **kwargs)
        self._arm = self.action_manager.get_term("arm_action")
        if self.action_manager.active_terms != ["arm_action", "finger_joint"]:
            raise ValueError("Unexpected action ordering")
        self._robot = self.scene["robot"]
        self._scale = torch.as_tensor(cfg.actions.arm_action.scale, dtype=torch.float32, device=self.device)

    def _check_eef(self, name):
        if name != self.eef_name:
            raise ValueError(f"Unknown EEF {name!r}; expected {self.eef_name!r}")

    def get_robot_eef_pose(self, eef_name, env_ids=None):
        self._check_eef(eef_name)
        # Read the action term's actual controlled pose, not an observation with
        # a different frame-transformer offset or a stale pre-reset buffer.
        position, quaternion = self._arm._compute_frame_pose()
        indices = slice(None) if env_ids is None else env_ids
        return pose.make_pose(position[indices], pose.matrix_from_quat(quaternion[indices]))

    def action_to_target_eef_pose(self, action):
        if action.shape != (self.num_envs, 7) or not torch.isfinite(action).all():
            raise ValueError("Expected finite actions with shape (num_envs, 7)")
        position, quaternion = self._arm._compute_frame_pose()
        target_position, target_quaternion = pose.apply_delta_pose(position, quaternion, action[:, :6] * self._scale)
        return {self.eef_name: pose.make_pose(target_position, pose.matrix_from_quat(target_quaternion))}

    def target_eef_pose_to_action(self, target_eef_pose_dict, gripper_action_dict,
                                  action_noise_dict=None, env_id=0):
        if set(target_eef_pose_dict) != {self.eef_name} or set(gripper_action_dict) != {self.eef_name}:
            raise ValueError("Target and gripper dictionaries must contain only 'droid'")
        target = target_eef_pose_dict[self.eef_name]
        if target.shape != (4, 4) or not torch.isfinite(target).all():
            raise ValueError("Expected a finite 4x4 target pose")
        current = self.get_robot_eef_pose(self.eef_name, [env_id])[0]
        delta_rot = target[:3, :3] @ current[:3, :3].T
        rotation = pose.axis_angle_from_quat(pose.quat_from_matrix(delta_rot))
        arm_action = torch.cat((target[:3, 3] - current[:3, 3], rotation)) / self._scale
        if action_noise_dict is not None:
            noise = float(action_noise_dict[self.eef_name])
            if not 0 <= noise < float("inf"):
                raise ValueError("Action noise must be finite and nonnegative")
            arm_action = arm_action + noise * torch.randn_like(arm_action)
        gripper = torch.as_tensor(gripper_action_dict[self.eef_name], device=self.device, dtype=arm_action.dtype).reshape(-1)
        if gripper.numel() != 1 or not torch.isfinite(gripper).all() or not ((gripper >= 0) & (gripper <= 1)).all():
            raise ValueError("Expected one gripper command in [0, 1]")
        return torch.cat((arm_action, gripper))

    def actions_to_gripper_actions(self, actions):
        if actions.shape[-1] != 7:
            raise ValueError("Expected seven action channels")
        return {self.eef_name: actions[..., 6:7]}

    def get_object_poses(self, env_ids=None):
        indices = slice(None) if env_ids is None else env_ids
        root_position = self._robot.data.root_pos_w[indices]
        root_quaternion = self._robot.data.root_quat_w[indices]
        result = {}
        for name, obj in self.scene.rigid_objects.items():
            position, quaternion = pose.subtract_frame_transforms(
                root_position, root_quaternion, obj.data.root_pos_w[indices], obj.data.root_quat_w[indices]
            )
            result[name] = pose.make_pose(position, pose.matrix_from_quat(quaternion))
        return result

    def get_subtask_term_signals(self, env_ids=None):
        # No heuristic boundary is accepted before the annotation review gate.
        return {}

    def check_success(self):
        return self._success_term.func(self, **self._success_term.params).to(dtype=torch.bool)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        get_world(self).reset_predicate_state(env_ids)

    def close(self):
        super().close()
        clear_world_cache()
