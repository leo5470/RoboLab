#!/usr/bin/env python3
"""Standalone SO-101 USB reader. Never imports RoboLab or Isaac Sim.

Run --help without LeRobot installed. --mock exercises IPC without hardware.
Only forward kinematics is needed, so meshes and an IK solver are unnecessary.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
PROTOCOL_VERSION = 1


class UrdfKinematics:
    """Evaluate the fixed/revolute chain from base_link to a fixed tool frame."""

    def __init__(self, path, target="gripper_frame_link", signs=None, offsets=None):
        self.signs = np.asarray(signs if signs is not None else [1] * 5, dtype=float)
        self.offsets = np.asarray(offsets if offsets is not None else [0] * 5, dtype=float)
        if (
            self.signs.shape != (5,)
            or self.offsets.shape != (5,)
            or not np.isin(self.signs, [-1, 1]).all()
            or not np.isfinite(self.offsets).all()
        ):
            raise ValueError("FK requires five signs (+1/-1) and five finite offsets in degrees")
        content = Path(path).read_bytes()
        self.checksum = hashlib.sha256(content).hexdigest()
        root = ET.fromstring(content)
        joints = {joint.find("child").attrib["link"]: joint for joint in root.findall("joint")}
        chain, visited = [], set()
        link = target
        while link != "base_link":
            if link in visited or link not in joints:
                raise ValueError(f"No valid URDF chain from base_link to {target}")
            visited.add(link)
            joint = joints[link]
            kind, name = joint.attrib["type"], joint.attrib["name"]
            if kind not in ("fixed", "revolute", "continuous") or joint.find("mimic") is not None:
                raise ValueError(f"Unsupported URDF joint {name}: {kind}")
            origin = joint.find("origin")
            xyz = self._xyz(origin, "xyz", "0 0 0")
            rpy = self._xyz(origin, "rpy", "0 0 0")
            transform = np.eye(4)
            transform[:3, 3] = xyz
            transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            axis = self._xyz(joint.find("axis"), "xyz", "1 0 0")
            if kind != "fixed" and (name not in JOINT_NAMES or np.linalg.norm(axis) < 1e-8):
                raise ValueError(f"Tool chain must use only the five arm joints, found {name}")
            chain.append((name, kind, transform, axis / max(np.linalg.norm(axis), 1e-8)))
            link = joint.find("parent").attrib["link"]
        self.chain = list(reversed(chain))
        names = [name for name, kind, _, _ in self.chain if kind != "fixed"]
        if len(names) != 5 or set(names) != set(JOINT_NAMES):
            raise ValueError("URDF tool chain must contain each of the five SO-101 arm joints exactly once")

    @staticmethod
    def _xyz(element, attribute, default):
        result = np.fromstring(element.get(attribute, default) if element is not None else default, sep=" ")
        if result.shape != (3,) or not np.isfinite(result).all():
            raise ValueError(f"Malformed URDF {attribute}")
        return result

    def pose(self, degrees):
        values = np.asarray(degrees, dtype=float)
        if values.shape != (5,) or not np.isfinite(values).all():
            raise ValueError("Expected five finite arm angles in degrees")
        angles = dict(zip(JOINT_NAMES, np.deg2rad(values * self.signs + self.offsets)))
        transform = np.eye(4)
        for name, kind, origin, axis in self.chain:
            transform = transform @ origin
            if kind != "fixed":
                movement = np.eye(4)
                movement[:3, :3] = Rotation.from_rotvec(axis * angles[name]).as_matrix()
                transform = transform @ movement
        quat = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
        return transform[:3, 3].tolist(), quat.tolist()


def write_status(path, data):
    if path:
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, allow_nan=False))
        os.replace(temporary, path)


def connect_leader(args):
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    leader = SO101Leader(
        SO101LeaderConfig(
            port=args.port,
            id=args.id,
            use_degrees=True,
            calibration_dir=Path(args.calibration_dir) if args.calibration_dir else None,
        )
    )
    if not leader.calibration:
        raise ValueError(f"No calibration for {args.id}; run lerobot-calibrate separately before collection")
    try:
        leader.connect(calibrate=False)
        if not leader.is_calibrated:
            raise ValueError("Device calibration does not match the saved calibration; calibrate separately")
    except BaseException:
        # A failed connect may still have opened the bus.
        if leader.is_connected:
            leader.disconnect()
        raise
    return leader


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port")
    parser.add_argument("--id")
    parser.add_argument("--calibration-dir")
    parser.add_argument("--urdf")
    parser.add_argument("--target-frame", default="gripper_frame_link")
    parser.add_argument("--joint-signs", type=float, nargs=5, default=[1] * 5)
    parser.add_argument("--joint-offsets-deg", type=float, nargs=5, default=[0] * 5)
    parser.add_argument("--read-hz", type=float, default=60)
    parser.add_argument("--read-retries", type=int, default=2)
    parser.add_argument("--endpoint", help="Optional local ipc:// endpoint; omit for terminal diagnostics")
    parser.add_argument("--status-file", help="Atomic startup/error status for the parent collector")
    parser.add_argument("--mock", action="store_true", help="Synthetic pose source; never opens USB")
    parser.add_argument("--duration", type=float, default=0, help="Stop after this many seconds; 0 runs until stopped")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.read_hz) or not 0 < args.read_hz <= 500:
        parser.error("--read-hz must be in (0, 500]")
    if not math.isfinite(args.duration) or args.duration < 0 or not 0 <= args.read_retries <= 10:
        parser.error("invalid duration or retry count")
    if args.endpoint and not args.endpoint.startswith("ipc://"):
        parser.error("Only local ipc:// transport is supported")
    if not args.mock and not all((args.port, args.id, args.urdf)):
        parser.error("hardware mode requires --port, --id and --urdf")

    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    leader = context = socket = None
    session_id = str(uuid.uuid4())
    try:
        kinematics = None
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "session_id": session_id,
            "mock": args.mock,
            "joint_names": list(JOINT_NAMES),
            "angle_units": "degrees",
            "quaternion_order": "wxyz",
            "position_units": "meters",
            "read_hz": args.read_hz,
            "read_retries": args.read_retries,
            "target_frame": args.target_frame,
            "joint_signs": args.joint_signs,
            "joint_offsets_deg": args.joint_offsets_deg,
        }
        if not args.mock:
            kinematics = UrdfKinematics(args.urdf, args.target_frame, args.joint_signs, args.joint_offsets_deg)
            leader = connect_leader(args)
            calibration = Path(leader.calibration_fpath).read_bytes()
            metadata.update(
                leader_id=args.id,
                port=args.port,
                urdf_sha256=kinematics.checksum,
                calibration_sha256=hashlib.sha256(calibration).hexdigest(),
                lerobot_version=importlib.metadata.version("lerobot"),
            )
        if args.endpoint:
            import zmq

            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.CONFLATE, 1)
            socket.connect(args.endpoint)
        write_status(args.status_file, {"state": "ready", "metadata": metadata})
        print(json.dumps({"state": "ready", "metadata": metadata}), flush=True)
        start = time.monotonic()
        sequence = 0
        while not stopped and (not args.duration or time.monotonic() - start < args.duration):
            read_start = time.monotonic()
            if args.mock:
                elapsed = read_start - start
                joints = [0.0] * 5
                position = [0.20 + 0.015 * math.sin(elapsed), 0.0, 0.20]
                quat, gripper = [1.0, 0.0, 0.0, 0.0], 100.0
            else:
                for attempt in range(args.read_retries + 1):
                    try:
                        reading = leader.get_action()
                        break
                    except (ConnectionError, OSError):
                        if attempt == args.read_retries:
                            raise
                joints = [float(reading[f"{name}.pos"]) for name in JOINT_NAMES]
                gripper = float(reading["gripper.pos"])
                position, quat = kinematics.pose(joints)
            sample = {
                "version": PROTOCOL_VERSION,
                "session_id": session_id,
                "sequence": sequence,
                "read_start": read_start,
                "read_end": time.monotonic(),
                "joint_names": list(JOINT_NAMES),
                "joints_deg": joints,
                "gripper": gripper,
                "position": position,
                "quaternion": quat,
            }
            encoded = json.dumps(sample, allow_nan=False).encode()
            if socket is not None:
                try:
                    socket.send(encoded, flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
            elif sequence % max(1, round(args.read_hz / 5)) == 0:
                print(encoded.decode(), flush=True)
            sequence += 1
            time.sleep(max(0.0, 1 / args.read_hz - (time.monotonic() - read_start)))
        write_status(args.status_file, {"state": "stopped", "metadata": metadata})
        return 0
    except Exception as exc:
        write_status(args.status_file, {"state": "error", "message": str(exc), "session_id": session_id})
        print(f"SO-101 reader failed: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        try:
            if leader is not None and leader.is_connected:
                leader.disconnect()
        finally:
            if socket is not None:
                socket.close(linger=0)
            if context is not None:
                context.term()


if __name__ == "__main__":
    raise SystemExit(main())
