"""Switchable motion repairs for MimicGen pick-and-place generation, without Isaac Sim.

Each repair is off by default; with all of them off, generation is unchanged.

- Pre-grasp approach (``approach_m``): the grasp segment starts at the last
  step before the gripper closes at which the hand is still ``approach_m``
  from its closing pose, instead of at the first recorded step. The reset
  pose connects to that pre-grasp pose; the human's path from reset is not
  replayed relative to the object.
- Raised lift end (``lift_clearance_m``): the grasp segment ends once the
  closed gripper has raised the object's bottom ``lift_clearance_m`` above
  the container rim, instead of at 2 cm.
- Over-bin placement start (``over_bin``): the placement segment starts at
  the first step, still raised and closed, at which the object's footprint is
  over the container opening (at most ``over_bin_overhang_m`` past the outer
  walls). The recorded carry between the two cuts is not replayed: the
  connection runs directly between two raised poses, with the gripper closed.
- Adaptive timing (``connection_speed_m_s``/``connection_rate_rad_s``): each
  connection takes long enough to stay within these speeds, and never less
  than the configured fixed interpolation time.
- Symmetric object yaw (``symmetric_yaw``): for an object that is rotationally
  symmetric about its vertical axis, its recorded heading carries no
  information, so on both the source and the target side the reference pose's
  heading is replaced by the object's bearing from the robot base (tilt kept).
  A grasp segment then turns with the change in bearing, which roughly
  preserves the arm's posture, instead of with an arbitrary object yaw that
  can force large hand rotations while carrying. (Dropping the heading
  entirely kept the source hand's heading, which the arm often cannot hold at
  distant targets.) Pilot 2026-11-12, source 02 on five targets: neither
  fixed rule is uniformly better than the raw yaw (raw 7/10 accepted, bearing
  6/10, dropped heading 4/10), so the best heading is target-specific.
  ``object_yaw_offset_deg`` therefore sets it per job instead: the source
  reference heading is removed and the target's becomes this angle, so the
  grasp segment turns by exactly this much about the object. GPU pilot
  2026-11-12: headings ranked best by ``path_screen`` were accepted 4/10
  against 7/10 for the recorded yaw (turning the grasp raises tracking error
  on the simulated arm, e.g. 1.8 -> 5 cm on the source's own layout), so the
  production profile keeps this repair off.

Reference measurements on the pail sources (2026-11-12): every source has
both cuts with ``lift_clearance_m=0.03`` and a 1 cm overhang; human hand
speed while carrying is 0.23 m/s median, 0.31 m/s at the 90th percentile
(rotation 0.37 rad/s at the 90th percentile).
"""

import math
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RepairSettings:
    approach_m: float | None = None
    lift_clearance_m: float | None = None
    over_bin: bool = False
    over_bin_overhang_m: float = 0.01
    connection_speed_m_s: float | None = None
    connection_rate_rad_s: float = 0.5
    symmetric_yaw: bool = False
    object_yaw_offset_deg: float | None = None

    def __post_init__(self):
        for name in ("approach_m", "lift_clearance_m", "connection_speed_m_s"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be positive")
        if not (math.isfinite(self.over_bin_overhang_m) and self.over_bin_overhang_m >= 0):
            raise ValueError("over_bin_overhang_m must be nonnegative")
        if not (math.isfinite(self.connection_rate_rad_s) and self.connection_rate_rad_s > 0):
            raise ValueError("connection_rate_rad_s must be positive")
        if self.object_yaw_offset_deg is not None and not (
                self.symmetric_yaw and math.isfinite(self.object_yaw_offset_deg)):
            raise ValueError("object_yaw_offset_deg must be finite and requires symmetric_yaw")
        if self.over_bin and self.lift_clearance_m is None:
            raise ValueError("over_bin requires lift_clearance_m: the connection must start raised")

    @property
    def boundaries(self):
        """Whether any repair moves a segment boundary."""
        return self.approach_m is not None or self.lift_clearance_m is not None or self.over_bin

    @property
    def active(self):
        return self.boundaries or self.connection_speed_m_s is not None or self.symmetric_yaw

    def metadata(self):
        return asdict(self)


def add_arguments(parser):
    group = parser.add_argument_group("motion repairs (all off by default)")
    group.add_argument("--approach-m", type=float,
                       help="Start the grasp segment where the hand is this far from its closing pose "
                            "(default: replay from the first recorded step)")
    group.add_argument("--lift-clearance-m", type=float,
                       help="End the grasp segment once the object's bottom is this far above the container rim "
                            "(default: at the 2 cm lift signal)")
    group.add_argument("--over-bin", action="store_true",
                       help="Start the placement segment with the object over the container opening; "
                            "requires --lift-clearance-m")
    group.add_argument("--over-bin-overhang-m", type=float, default=0.01,
                       help="Largest footprint overhang past the container's outer walls at the over-bin cut")
    group.add_argument("--connection-speed-m-s", type=float,
                       help="Lengthen connections to stay within this hand speed (default: fixed --interpolation-s)")
    group.add_argument("--connection-rate-rad-s", type=float, default=0.5,
                       help="With --connection-speed-m-s, also stay within this hand rotation rate")
    group.add_argument("--symmetric-object-yaw", action="store_true",
                       help="Treat the grasped object as symmetric about its vertical axis")
    group.add_argument("--object-yaw-offset-deg", type=float,
                       help="With --symmetric-object-yaw, turn the grasp by this angle about the object relative to "
                            "the source (default: follow the change in bearing from the robot base)")


def settings_from_args(args):
    return RepairSettings(approach_m=args.approach_m, lift_clearance_m=args.lift_clearance_m, over_bin=args.over_bin,
                          over_bin_overhang_m=args.over_bin_overhang_m,
                          connection_speed_m_s=args.connection_speed_m_s,
                          connection_rate_rad_s=args.connection_rate_rad_s, symmetric_yaw=args.symmetric_object_yaw,
                          object_yaw_offset_deg=args.object_yaw_offset_deg)


def placement_geometry(object_bounds, container_bounds):
    """Object footprint radius, container rim height and opening from local bounds.

    Each bounds argument is ``{"lower", "upper", "scale"}`` in the asset's root
    frame, as v1 preparation reads them. The container is assumed to rest on
    the same support as the object starts on.
    """
    lo, hi, scale = (np.asarray(object_bounds[k], dtype=float) for k in ("lower", "upper", "scale"))
    radius = float(np.max(np.abs(np.r_[lo[:2], hi[:2]] * np.r_[scale[:2], scale[:2]])))
    clo, chi, cscale = (np.asarray(container_bounds[k], dtype=float) for k in ("lower", "upper", "scale"))
    return {"object_radius_m": radius, "rim_height_m": float((chi[2] - clo[2]) * cscale[2]),
            "opening_center_m": ((chi[:2] + clo[:2]) / 2 * cscale[:2]).tolist(),
            "opening_half_m": ((chi[:2] - clo[:2]) / 2 * cscale[:2]).tolist()}


def _yaw(quat):
    w, x, y, z = quat
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def remove_yaw(quats):
    """Quaternions [w, x, y, z] with their heading about world z removed (``Rz(-yaw) R``, ZYX yaw)."""
    quats = np.asarray(quats, dtype=float)
    out = []
    for q in np.atleast_2d(quats):
        q = q / np.linalg.norm(q)
        yaw = _yaw(q)
        c, s = math.cos(-yaw / 2), math.sin(-yaw / 2)
        w, x, y, z = q
        # (c, 0, 0, s) * q
        out.append([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w])
    return np.array(out).reshape(quats.shape)


def heading_reference(poses, heading_deg):
    """Root poses [.., 7] with the heading about world z set to ``heading_deg`` (tilt kept)."""
    poses = np.array(poses, dtype=float, copy=True)
    flat = poses.reshape(-1, 7)
    c, s = math.cos(math.radians(heading_deg) / 2), math.sin(math.radians(heading_deg) / 2)
    for row, q in zip(flat, remove_yaw(flat[:, 3:7]).reshape(-1, 4)):
        w, x, y, z = q
        row[3:7] = [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w]
    return flat.reshape(poses.shape)


def bearing_reference(poses):
    """Root poses [.., 7] with the heading set to the bearing of the position from the robot base (origin)."""
    poses = np.array(poses, dtype=float, copy=True)
    flat = poses.reshape(-1, 7)
    tilt = remove_yaw(flat[:, 3:7])
    for row, q in zip(flat, tilt):
        bearing = math.atan2(row[1], row[0])
        c, s = math.cos(bearing / 2), math.sin(bearing / 2)
        w, x, y, z = q
        row[3:7] = [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w]
    return flat.reshape(poses.shape)


def source_cuts(actions, hand_xyz, object_pose, container_pose, grasp_end, geometry, settings):
    """Segment boundaries of one source under ``settings``.

    ``object_pose``/``container_pose`` are pre-action root poses [T, 7];
    ``hand_xyz`` the achieved hand positions [T, 3]; ``grasp_end`` the pool's
    unrepaired grasp segment end. Returns ``{"approach", "lift", "over_bin",
    "release"}`` (None where the repair is off); raises if a needed cut is absent.
    """
    actions, hand_xyz = np.asarray(actions), np.asarray(hand_xyz, dtype=float)
    object_pose, container_pose = np.asarray(object_pose, dtype=float), np.asarray(container_pose, dtype=float)
    steps = len(actions)
    closed = actions[:, -1] < 0
    grasp = grasp_end - 1  # the step whose lift raised the grasp signal
    opened = np.flatnonzero(~closed[grasp:])
    release = grasp + int(opened[0]) if len(opened) else steps
    cuts = {"approach": None, "lift": None, "over_bin": None, "release": int(release)}
    if settings.approach_m is not None:
        open_before = np.flatnonzero(~closed[:grasp + 1])
        close = int(open_before[-1]) + 1 if len(open_before) else 0
        far = np.flatnonzero(np.linalg.norm(hand_xyz[:close] - hand_xyz[close], axis=1) >= settings.approach_m)
        cuts["approach"] = int(far[-1]) if len(far) else 0
    if settings.lift_clearance_m is not None:
        lift = object_pose[:, 2] - object_pose[0, 2]
        high = closed & (lift >= geometry["rim_height_m"] + settings.lift_clearance_m)
        high[:grasp] = False
        high[release:] = False
        if not high.any():
            raise ValueError(f"Object is never raised {settings.lift_clearance_m} m above the rim before release")
        cuts["lift"] = int(np.argmax(high))
        if settings.over_bin:
            center, half = np.asarray(geometry["opening_center_m"]), np.asarray(geometry["opening_half_m"])
            inside = np.zeros(steps, bool)
            for t in range(steps):
                yaw = _yaw(container_pose[t, 3:])
                c, s = math.cos(yaw), math.sin(yaw)
                d = object_pose[t, :2] - container_pose[t, :2]
                local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]]) - center
                inside[t] = np.all(np.abs(local) + geometry["object_radius_m"] <= half + settings.over_bin_overhang_m)
            over = high & inside
            over[:cuts["lift"]] = False
            if not over.any():
                raise ValueError("Object is never over the container opening while raised and held")
            cuts["over_bin"] = int(np.argmax(over))
    return cuts


def edit_boundaries(randomized, base, cuts, settings):
    """Apply source cuts to the generator's randomized boundaries ``[N, 2, 2]`` (grasp, placement).

    ``base`` holds the unrandomized boundaries, so a randomized offset of the
    grasp end is kept relative to the lift cut. The placement may start after
    the grasp segment ends (the over-bin gap), never before.
    """
    out = np.array(randomized, copy=True)
    base = np.asarray(base)
    if out.shape != base.shape or out.shape[1:] != (2, 2) or len(cuts) != len(out):
        raise ValueError("Boundaries must be [sources, 2 subtasks, 2] with one cut record per source")
    for n, cut in enumerate(cuts):
        offset = int(out[n, 0, 1] - base[n, 0, 1])
        if cut["approach"] is not None:
            out[n, 0, 0] = cut["approach"]
        if cut["lift"] is not None:
            out[n, 0, 1] = cut["lift"] + offset
        out[n, 1, 0] = out[n, 0, 1] if cut["over_bin"] is None else max(cut["over_bin"], out[n, 0, 1])
        start, end, placement, final = out[n, 0, 0], out[n, 0, 1], out[n, 1, 0], out[n, 1, 1]
        if not (0 <= start < end <= placement < final):
            raise ValueError(f"Source {n}: repaired boundaries {out[n].tolist()} are not ordered and nonempty")
        if cut["over_bin"] is not None and placement >= cut["release"]:
            raise ValueError(f"Source {n}: placement starts at {placement}, after the release at {cut['release']}")
    return out


def connection_steps(start, target, minimum, speed_m_s, rate_rad_s, hz):
    """Interpolation steps for a connection between two 4x4 poses."""
    start, target = np.asarray(start, dtype=float), np.asarray(target, dtype=float)
    distance = float(np.linalg.norm(target[:3, 3] - start[:3, 3]))
    angle = float(np.arccos(np.clip((np.trace(target[:3, :3] @ start[:3, :3].T) - 1) / 2, -1, 1)))
    return max(int(minimum), math.ceil(distance / speed_m_s * hz - 1e-9), math.ceil(angle / rate_rad_s * hz - 1e-9))


def retime(waypoints, start_pose, current_steps, new_steps, interpolate):
    """Replace a segment's leading connection with one of ``new_steps`` interpolation points.

    Isaac Lab emits ``current_steps + 1`` waypoints for the connection, the
    last being the segment's first target. ``interpolate(start, end, n)``
    returns ``n + 2`` poses including both ends, as Isaac Lab's does.
    """
    if new_steps == current_steps:
        return list(waypoints)
    first = waypoints[current_steps]
    poses = interpolate(start_pose, first.pose, new_steps)
    lead = [type(first)(pose=pose, gripper_action=first.gripper_action, noise=waypoints[0].noise)
            for pose in poses[1:-1]]
    return lead + list(waypoints[current_steps:])
