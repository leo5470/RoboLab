"""Home-pose adapter; input, joint mapping and limiting stay upstream.

Import after Isaac Sim starts. Upstream defaults to its cube demo's Panda home;
RoboLab supplies the selected DROID environment's actual reset joint positions.
"""

import math
import time
from dataclasses import dataclass

import numpy as np
from isaaclab.devices import teleop_device_factory
from omy_leader_isaaclab.device import OmyLeaderCfg, OmyLeaderDevice
from omy_leader_isaaclab.franka_config import FRANKA_HOME
from omy_leader_isaaclab.franka_retarget import OmyToFrankaRetarget

from .calibration import recenter_direct, validate_home


@dataclass
class RoboLabOmyLeaderCfg(OmyLeaderCfg):
    franka_initial_q: tuple[float, ...] = FRANKA_HOME

    def __post_init__(self):
        if self.class_type is None:
            self.class_type = RoboLabOmyLeaderDevice
        super().__post_init__()


class RoboLabOmyLeaderDevice(OmyLeaderDevice):
    def _open_source(self):
        super()._open_source()
        latest = self._latest
        self.last_input_sample = None

        def capture():
            # The serial reader replaces the tuple atomically. Read it once so
            # its reading and timestamp belong to the same upstream sample.
            if self.cfg.source == "serial":
                reading, received_at = self._buf
                age = time.time() - received_at
            else:
                reading, age = latest()
                received_at = time.time() - age
            observed_at = time.monotonic_ns()
            self.last_input_sample = {
                "read_monotonic_ns": observed_at,
                "source_wall_ns": int(received_at * 1e9) if reading is not None and math.isfinite(age) else None,
                "age_s": float(age) if reading is not None and math.isfinite(age) else None,
                "stale": reading is None or not math.isfinite(age) or age < 0 or age > self.cfg.stale_s,
            }
            if self.last_input_sample["stale"]:
                return None, math.inf
            return reading, age

        self._latest = capture

    def advance(self):
        self.last_input_sample = None
        return super().advance()

    def _make_mapper(self):
        if self.cfg.target != "franka":
            raise ValueError("The RoboLab home adapter requires target='franka'")
        mapper = super()._make_mapper()
        mapper.initial_q = validate_home(mapper.config, self.cfg.franka_initial_q)
        if self.cfg.auto_zero and mapper.config.wrist_mode != "direct":
            # Fail before opening the source or configuring any leader motors.
            raise ValueError("RoboLab --auto-zero requires direct wrist mode")
        mapper.reset()
        return mapper

    def _mapper_home(self):
        return np.array([*self.cfg.franka_initial_q, 1.0], dtype=float)

    def reset(self):
        super().reset()
        # The environment has just reset to home. Do not send the previous
        # episode's command if the leader happens to be stale during this reset.
        self._last = self._mapper_home()

    def _auto_zero(self, reading):
        config = recenter_direct(self._mapper.config, reading, self.cfg.franka_initial_q)
        self._mapper = OmyToFrankaRetarget(config, initial_q=self.cfg.franka_initial_q)
        print("[omy_leader] DROID home auto-zero:",
              {joint: round(value, 3) for joint, value in config.omy_zero_rad.items()})

    def enable_input_diagnostics(self):
        """Observe the exact sample consumed by upstream advance(), without another read."""
        from .diagnostics import enable_input_diagnostics

        enable_input_diagnostics(self)

    def __str__(self):
        home = [round(math.degrees(value), 2) for value in self.cfg.franka_initial_q]
        return (f"OMY-L100 leader ({self.cfg.source}) -> DROID (upstream Franka mapper)\n"
                f"  reset J1..J7 (deg): {home}; gripper open\n"
                f"  wrist: {self._mapper.config.wrist_mode}; calibrated directions/offsets retained")


# Isaac Lab 2.x uses exact config types; newer versions use cfg.class_type.
if hasattr(teleop_device_factory, "DEVICE_MAP"):
    teleop_device_factory.DEVICE_MAP[RoboLabOmyLeaderCfg] = RoboLabOmyLeaderDevice
