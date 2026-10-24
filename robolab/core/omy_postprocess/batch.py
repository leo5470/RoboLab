"""A lazy, isolated Isaac worker with per-job acknowledgments and GPU ownership."""

import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from robolab.core.omy_teleop.collection_handoff import replace_report


class BatchRenderer:
    def __init__(self, output, device, *, startup_timeout_s=300, job_timeout_s=900):
        self.output, self.device = Path(output), device
        self.startup_timeout_s, self.job_timeout_s = startup_timeout_s, job_timeout_s
        self.process = self.queue = self.slot = None
        self.job_number = 0
        self.metrics = {"worker_starts": 0, "worker_crashes": 0, "completed_jobs": 0,
                        "environment_builds": 0, "environment_reuses": 0, "queue_wait_s": 0.0}
        self.signature = None
        self.ready_recorded = None

    def _stop(self, signum=signal.SIGTERM):
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signum)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=10)
        self.process = None

    def run(self, command, *, cwd, stdout, stderr, check, env):
        script_index = next(i for i, arg in enumerate(command) if str(arg).endswith("/scripts/materialize_demo_images.py"))
        signature = (tuple(command[:script_index + 1]), env.get("CUDA_VISIBLE_DEVICES"))
        if self.signature is not None and self.signature != signature:
            raise ValueError("A batch worker cannot change its isolated GPU or interpreter")
        self.signature = signature
        if self.slot is None:
            from .render import _gpu_slot

            self.slot = _gpu_slot(self.device)
            started = time.perf_counter()
            self.slot.__enter__()
            self.metrics["queue_wait_s"] += time.perf_counter() - started
        if self.process is None:
            if self.metrics["worker_starts"] >= 2:
                raise RuntimeError("Isaac batch worker restart limit reached; retry the retained outputs")
            directory = self.output / "workers"
            directory.mkdir(parents=True, exist_ok=True)
            self.queue = Path(tempfile.mkdtemp(prefix="isaac_", dir=directory))
            self.job_number = 0
        number = self.job_number
        self.job_number += 1
        destination = command[command.index("--output") + 1]
        request = {"schema_version": 1, "id": number,
                   "argv": command[script_index + 1:], "log": str(Path(stdout.name).resolve())}
        replace_report(self.queue / f"job_{number:06d}.json", request)
        if self.process is None:
            worker_command = [*command, "--batch-queue", str(self.queue.resolve())]
            kit = worker_command.index("--kit_args") + 1
            worker_command[kit] += " --/plugins/carb.tasking.plugin/threadCount=4 --/plugins/omni.tbb.globalcontrol/maxThreadCount=4"
            worker_env = {**env, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
            with (self.queue / "worker.log").open("ab") as log:
                self.metrics["worker_starts"] += 1
                self.process = subprocess.Popen(worker_command, cwd=cwd, env=worker_env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.perf_counter()
        result_path = self.queue / f"result_{number:06d}.json"
        try:
            while not result_path.exists():
                status = self.process.poll()
                if status is not None:
                    self.metrics["worker_crashes"] += 1
                    self._stop()
                    raise RuntimeError(f"Isaac worker exited ({status}); see {self.queue / 'worker.log'}")
                limit = self.job_timeout_s if (self.queue / "ready.json").exists() else self.startup_timeout_s
                if time.perf_counter() - started > limit:
                    self.metrics["worker_crashes"] += 1
                    self._stop()
                    raise RuntimeError(f"Isaac worker timed out; see {self.queue / 'worker.log'}")
                time.sleep(0.05)
            result = json.loads(result_path.read_text())
            if result.get("id") != number or result.get("output") != destination:
                self._stop()
                raise ValueError("Worker acknowledgment does not match this render job")
            if result["status"] != "complete":
                raise RuntimeError(f"Isaac render failed: {result.get('error')}; see {stdout.name}")
            self.metrics["completed_jobs"] += 1
            if self.ready_recorded != self.queue:
                ready = json.loads((self.queue / "ready.json").read_text())
                self.metrics["startup_s"] = self.metrics.get("startup_s", 0) + ready.get("startup_s", 0)
                self.ready_recorded = self.queue
            self.metrics["environment_setup_s"] = self.metrics.get("environment_setup_s", 0) + result.get("environment_setup_s", 0)
            self.metrics["environment_reuses" if result["reused_environment"] else "environment_builds"] += 1
            return subprocess.CompletedProcess(command, 0)
        except KeyboardInterrupt:
            self._stop(signal.SIGINT)
            raise

    def close(self):
        try:
            if self.process is not None:
                status = self.process.poll()
                if status is None:
                    replace_report(self.queue / "stop.json", {"stop": True})
                    try:
                        status = self.process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        raise RuntimeError(f"Isaac worker shutdown timed out; see {self.queue / 'worker.log'}")
                if status:
                    raise RuntimeError(f"Isaac worker shutdown failed ({status}); see {self.queue / 'worker.log'}")
        finally:
            try:
                self._stop()
            finally:
                if self.slot is not None:
                    self.slot.__exit__(None, None, None)
                    self.slot = None
