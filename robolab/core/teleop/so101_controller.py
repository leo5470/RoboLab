"""Hardware-independent leader controller and episode pause diagnostics."""

import time

import numpy as np

from .retarget import ORIENTATION_MODE_CODES, TrackingError, quaternion
from .so101_source import StaleSample


class LeaderController:
    def __init__(self, source, mapper, controls, clock=time.monotonic):
        self.source = source
        self.mapper = mapper
        self.controls = controls
        self.clock = clock
        self.reset()

    def reset(self):
        self.mapper.reset()
        self.pause_count = 0
        self.paused_seconds = 0.0
        self.pause_started = None
        self.last_wall_step = None
        self.diagnostics = None
        self.message = None
        self.resumed = False

    def begin(self, robot_pos, robot_quat):
        sample = self.source.current()
        self.mapper.anchor(sample, robot_pos, robot_quat)
        self.controls.begin()
        self.last_wall_step = self.clock()

    def action(self, robot_pos, robot_quat):
        self.resumed = False
        sample = None
        try:
            sample = self.source.current()
        except StaleSample as exc:
            self.controls.pause("stale input")
            self.message = str(exc)
        if self.controls.paused:
            if self.pause_started is None:
                self.pause_started = self.clock()
                self.pause_count += 1
            if not self.controls.resume_requested:
                return None
            if sample is None:
                self.controls.resume_requested = False
                return None
            try:
                self.mapper.anchor(sample, robot_pos, robot_quat)
            except ValueError as exc:
                self.message = str(exc)
                self.controls.resume_requested = False
                return None
            self.controls.resume()
            self.paused_seconds += self.clock() - self.pause_started
            self.pause_started = None
            self.resumed = True
            self.message = None
        try:
            action = self.mapper.command(sample, robot_pos, robot_quat)
        except TrackingError as exc:
            self.controls.pause("tracking error")
            self.message = str(exc)
            return None
        now = self.clock()
        self.diagnostics = {
            "joints_deg": sample.joints_deg.copy(),
            "gripper": np.array([sample.gripper]),
            "leader_position": sample.position.copy(),
            "leader_quaternion": sample.quaternion.copy(),
            "sequence": np.array([sample.sequence], dtype=np.int64),
            "read_start": np.array([sample.read_start]),
            "read_end": np.array([sample.read_end]),
            "sample_age_s": np.array([now - sample.read_start]),
            "target_position": self.mapper.target_pos.copy(),
            "target_quaternion": quaternion(self.mapper.target_rot),
            "orientation_mode": np.array([ORIENTATION_MODE_CODES[self.mapper.cfg.orientation_mode]], dtype=np.int64),
            "reference_generation": np.array([self.mapper.generation], dtype=np.int64),
            "pause_count": np.array([self.pause_count], dtype=np.int64),
            "paused_seconds": np.array([self.paused_seconds]),
            "wall_step_interval_s": np.array([now - self.last_wall_step]),
        }
        self.last_wall_step = now
        return action
