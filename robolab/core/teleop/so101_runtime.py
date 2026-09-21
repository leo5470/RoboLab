"""Isaac-facing adapter for the local leader. Import after the application starts."""

import json
import time

import torch

from .lifecycle import LeaderControls
from .local_controls import LocalControls
from .retarget import PoseRetargeter, validate_action_config
from .so101_controller import LeaderController
from .so101_options import reader_arguments
from .so101_source import LeaderSource, SourceFault, StaleSample


class SO101Teleop:
    def __init__(self, env, args, inset=None):
        self.env = env
        self.args = args
        self.controls = LeaderControls()
        self.input = None
        self.source = None
        self.closed = False
        self.last_message = None
        self.last_pause = None
        self.steps = 0
        self.total_steps = 0
        self.attempt = 0
        try:
            scale = validate_action_config(env.cfg.actions)
            if env.action_manager.total_action_dim != 7:
                raise ValueError("SO-101 requires exactly seven DROID actions")
            self.source = LeaderSource(
                args.leader_python,
                reader_arguments(args),
                timeout=args.retarget_config.sample_timeout_s,
                startup_timeout=args.leader_startup_timeout,
            )
            leader_fk = None
            if not args.leader_mock and args.retarget_config.base_pan_gain != 1:
                from .so101_kinematics import leader_kinematics

                leader_fk = leader_kinematics(
                    args.leader_urdf, signs=args.leader_joint_signs, offsets=args.leader_joint_offsets_deg
                ).pose
            self.controller = LeaderController(
                self.source, PoseRetargeter(args.retarget_config, scale, leader_fk=leader_fk), self.controls
            )
            if not args.smoke_steps:
                callbacks = {
                    "N": self.controls.request_start,
                    "R": self.controls.request_retry,
                    "H": self.controls.request_clutch,
                    "ESC": self.controls.request_stop,
                }
                if inset is not None:
                    callbacks["B"] = inset.toggle
                self.input = LocalControls(callbacks, args.control_device, args.control_source)
        except BaseException:
            self.close()
            raise

    def pose(self):
        from robolab.robots.droid import ee_pos, ee_quat

        return ee_pos(self.env)[0].detach().cpu().numpy(), ee_quat(self.env)[0].detach().cpu().numpy()

    def event(self, event, **details):
        row = {
            "event": event,
            "monotonic_time": time.monotonic(),
            "attempt": self.attempt,
            "step": self.steps,
            **details,
        }
        print(f"SO-101: {event}" + (f" {details}" if details else ""), flush=True)
        if self.args.record:
            with open(f"{self.env.output_dir}/so101_events.jsonl", "a") as stream:
                stream.write(json.dumps(row) + "\n")

    def poll(self):
        if self.input is not None:
            self.input.poll()
        self.source.poll()
        if self.source.buffer.fault:
            raise SourceFault(self.source.buffer.fault)

    def reset(self):
        self.controller.reset()
        self.env.so101_diagnostics = None
        self.last_pause = None
        self.last_message = None
        self.steps = 0
        self.attempt += 1
        if self.input is not None:
            self.input.reset()

    def try_begin(self):
        try:
            self.controller.begin(*self.pose())
        except (StaleSample, ValueError) as exc:
            self.controls.start_requested = False
            if str(exc) != self.last_message:
                self.event("start blocked", reason=str(exc))
                self.last_message = str(exc)
            return False
        self.last_message = None
        if self.args.record:
            metadata = {
                "schema_version": 1,
                **self.source.metadata,
                "mapping": self.args.retarget_config.to_dict(),
                "action_scale": self.controller.mapper.scale,
                "control_source": self.args.control_source,
                "control_device": self.args.control_device,
                "action_frame": "robot-root",
                "controlled_body": "base_link",
                "rotation_encoding": "axis-angle",
                "orientation_mode_encoding": "0=fixed,1=pose,2=wrist-roll",
                "gripper_encoding": "0=open,1=closed",
            }
            self.env.recorder_manager.set_dataset_attrs({"robolab_so101_leader": json.dumps(metadata)})
        self.event("recording" if self.args.record else "commissioning")
        return True

    def advance(self):
        action = self.controller.action(*self.pose())
        paused = self.controls.paused
        if paused != self.last_pause:
            self.event("paused" if paused else "resumed", reason=paused, detail=self.controller.message)
            self.last_pause = paused
        if action is None:
            if self.controller.message and self.controller.message != self.last_message:
                print(self.controller.message, flush=True)
                self.last_message = self.controller.message
            return None
        if self.args.record:
            self.env.so101_diagnostics = {
                key: torch.as_tensor(value.copy(), device=self.env.device).unsqueeze(0)
                for key, value in self.controller.diagnostics.items()
            }
        return torch.as_tensor(action, device=self.env.device)

    def step_completed(self):
        self.steps += 1
        self.total_steps += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.input is not None:
                self.input.close()
        finally:
            if self.source is not None:
                self.source.close()
