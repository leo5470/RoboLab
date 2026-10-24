"""Neutral-pose offsets for the upstream direct joint mapper (no simulator)."""

import math
from dataclasses import replace

from omy_leader_isaaclab.franka_config import FRANKA_LOWER, FRANKA_UPPER
from omy_leader_isaaclab.franka_wrist import K4, K5, K7


class DirectWristTurnReference:
    """Choose the wrist encoder's whole-turn reference once per connection.

    Extended-position joint_6 can report the calibrated pose plus whole turns.
    Select an equivalent initial J7 target inside its calibrated joint limits,
    then keep that source offset fixed. Never wrap subsequent motion or reset
    the reference at episode boundaries: that would introduce command jumps.
    """

    def __init__(self, config):
        self.config = config
        self.offset_rad = None

    def normalize(self, reading):
        raw = float(reading["joint_6.pos"])
        if not math.isfinite(raw):
            raise ValueError("Nonfinite L100 wrist reading")
        if self.offset_rad is None:
            c = self.config
            gain = K7 * c.omy_sign.get("joint_6", 1.0)
            self.offset_rad = 0.0
            if c.wrist_mode == "direct" and abs(gain) == 1.0:
                target = c.j7_offset_rad + gain * (raw - c.omy_zero_rad.get("joint_6", 0.0))
                turns = round(target / math.tau)
                equivalent = target - turns * math.tau
                if FRANKA_LOWER[6] + c.limit_margin_rad <= equivalent <= FRANKA_UPPER[6] - c.limit_margin_rad:
                    self.offset_rad = turns * math.tau / gain
        if not self.offset_rad:
            return reading
        return {**reading, "joint_6.pos": raw - self.offset_rad}


def validate_home(config, target):
    target = tuple(float(value) for value in target)
    if len(target) != 7 or not all(math.isfinite(value) for value in target):
        raise ValueError("Franka home must contain seven finite joint angles")
    if not math.isclose(target[2], config.j3_fixed_rad, abs_tol=1e-9):
        raise ValueError("Franka home J3 must match the mapper's fixed J3")
    if any(not lo + config.limit_margin_rad <= q <= hi - config.limit_margin_rad
           for q, lo, hi in zip(target, FRANKA_LOWER, FRANKA_UPPER)):
        raise ValueError("Franka home is outside the upstream margined joint limits")
    return target


def recenter_direct(config, reading, target):
    """Return offsets making this raw READY reading map to the requested home.

    Inverts the existing affine arm/direct-wrist maps. Directions, scales,
    limits, rates, wrist constants and gripper endpoints remain unchanged.
    These software offsets are not motor homing/EEPROM settings.
    """
    target = validate_home(config, target)
    if config.wrist_mode != "direct":
        raise ValueError("READY recentering requires direct wrist mode; ZYZ auto-zero is not supported")
    zero = dict(config.omy_zero_rad)

    def offset(joint, value):
        raw = float(reading[f"{joint}.pos"])
        sign = config.omy_sign.get(joint, 1.0)
        if not math.isfinite(raw) or not math.isfinite(sign) or sign == 0:
            raise ValueError(f"Invalid reading or direction for {joint}")
        zero[joint] = raw - value / sign

    for mapping in config.arm_map:
        if not math.isfinite(mapping.scale) or mapping.scale == 0:
            raise ValueError("Arm mapping scale must be finite and nonzero")
        offset(mapping.omy_joint, (target[mapping.franka_index] - mapping.offset_rad) / mapping.scale)
    offset("joint_4", (target[5] - config.j6_reset_rad) / K4)
    offset("joint_5", target[4] / K5)
    offset("joint_6", (target[6] - config.j7_offset_rad) / K7)
    return replace(config, omy_zero_rad=zero)
