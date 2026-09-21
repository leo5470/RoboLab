"""Local SO-101 helper supervision and validated, latest-only IPC input."""

import json
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import zmq

from .retarget import rotation, vector

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
MAX_MESSAGE_BYTES = 16384


class SourceFault(RuntimeError):
    """The current attempt must be rejected, rather than resumed."""


class StaleSample(RuntimeError):
    """Pause stepping until explicitly re-anchored with a fresh sample."""


@dataclass(frozen=True)
class LeaderSample:
    session_id: str
    sequence: int
    read_start: float
    read_end: float
    joints_deg: object
    gripper: float
    position: object
    quaternion: object

    @classmethod
    def decode(cls, payload, now):
        if len(payload) > MAX_MESSAGE_BYTES:
            raise ValueError("Leader message is too large")
        data = json.loads(payload)
        if data.get("version") != 1 or data.get("joint_names") != list(JOINT_NAMES):
            raise ValueError("Unsupported leader protocol or joint ordering")
        session = data["session_id"]
        sequence = data["sequence"]
        start, end = data["read_start"], data["read_end"]
        if not isinstance(session, str) or not 1 <= len(session) <= 128:
            raise ValueError("Invalid reader session")
        if type(sequence) is not int or not 0 <= sequence < 2**63:
            raise ValueError("Invalid reader sequence")
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end <= now + 0.01):
            raise ValueError("Invalid monotonic read timestamps")
        gripper = float(data["gripper"])
        if not math.isfinite(gripper) or not -1 <= gripper <= 101:
            raise ValueError("Invalid normalized gripper reading")
        quat = vector(data["quaternion"], 4, "leader quaternion")
        rotation(quat)
        return cls(
            session,
            sequence,
            start,
            end,
            vector(data["joints_deg"], 5, "joints_deg"),
            gripper,
            vector(data["position"], 3, "leader position"),
            quat,
        )


class SampleBuffer:
    def __init__(self, timeout=0.2):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Sample timeout must be finite and positive")
        self.timeout = timeout
        self.latest = None
        self.fault = None

    def accept(self, payload, now):
        try:
            sample = LeaderSample.decode(payload, now)
            if self.latest is not None:
                if sample.session_id != self.latest.session_id:
                    raise ValueError("Leader reader session changed")
                if sample.sequence <= self.latest.sequence:
                    return False
                if sample.read_start < self.latest.read_start or sample.read_end < self.latest.read_end:
                    raise ValueError("Leader timestamps moved backwards")
            self.latest = sample
            return True
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
            self.fault = str(exc)
            return False

    def current(self, now):
        if self.fault:
            raise SourceFault(self.fault)
        # Age from read start includes serial read latency, not just IPC latency.
        if self.latest is None or now - self.latest.read_start > self.timeout:
            raise StaleSample("No fresh leader sample; motion is paused")
        return self.latest


class LeaderSource:
    def __init__(self, python, reader_args, timeout=0.2, startup_timeout=20.0, clock=time.monotonic):
        self.clock = clock
        self.buffer = SampleBuffer(timeout)
        self.started = clock()
        self.startup_timeout = startup_timeout
        self.metadata = None
        self.process = self.context = self.socket = None
        self.directory = tempfile.TemporaryDirectory(prefix="robolab-so101-")
        self.status_path = Path(self.directory.name) / "status.json"
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PULL)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.CONFLATE, 1)
            self.socket.setsockopt(zmq.MAXMSGSIZE, MAX_MESSAGE_BYTES)
            endpoint = f"ipc://{self.directory.name}/samples.sock"
            self.socket.bind(endpoint)
            reader = Path(__file__).resolve().parents[3] / "scripts/so101_leader_reader.py"
            # Kit mutates Python/library search paths; the helper uses its own environment.
            environment = dict(os.environ)
            for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
                environment.pop(key, None)
            self.process = subprocess.Popen(
                [
                    str(python),
                    "-u",
                    str(reader),
                    *reader_args,
                    "--endpoint",
                    endpoint,
                    "--status-file",
                    str(self.status_path),
                ],
                stdin=subprocess.DEVNULL,
                env=environment,
            )
        except BaseException:
            self.close()
            raise

    def poll(self):
        now = self.clock()
        if self.metadata is None and self.status_path.exists():
            try:
                status = json.loads(self.status_path.read_text())
                if status["state"] == "ready":
                    self.metadata = status["metadata"]
                elif status["state"] == "error":
                    self.buffer.fault = status["message"]
            except (ValueError, KeyError, OSError) as exc:
                self.buffer.fault = f"Invalid reader status: {exc}"
        if self.process.poll() is not None:
            detail = ""
            if self.status_path.exists():
                try:
                    detail = json.loads(self.status_path.read_text()).get("message", "")
                except (ValueError, OSError):
                    pass
            self.buffer.fault = f"Leader reader exited ({self.process.returncode}). {detail}"
        # CONFLATE means a single receive gets the most recent pending message.
        try:
            payload = self.socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            payload = None
        if payload is not None:
            self.buffer.accept(payload, self.clock())
        if self.metadata and self.buffer.latest and self.metadata["session_id"] != self.buffer.latest.session_id:
            self.buffer.fault = "Reader metadata/session mismatch"
        if self.buffer.latest is None and now - self.started > self.startup_timeout:
            self.buffer.fault = self.buffer.fault or "Leader reader startup timed out"

    def current(self):
        self.poll()
        sample = self.buffer.current(self.clock())
        if self.metadata is None:
            raise StaleSample("Waiting for leader metadata")
        return sample

    def close(self):
        try:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
        finally:
            if self.socket is not None:
                self.socket.close(linger=0)
                self.socket = None
            if self.context is not None:
                self.context.term()
                self.context = None
            self.directory.cleanup()
