"""Run sequential materialization jobs inside one already-started Kit app."""

from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import sys
import time
import traceback

from robolab.core.omy_teleop.collection_handoff import replace_report
from .io import sha256_file


class EnvironmentCache:
    def __init__(self):
        self.key = self.value = None
        self.reused = False
        self.setup_s = 0.0

    def get(self, args, metadata, build):
        started = time.perf_counter()
        key = (sha256_file(Path(args.input).parent / "env_cfg.json"), metadata.get("task"),
               json.dumps(metadata.get("task_source"), sort_keys=True), args.renderer,
               args.camera_scale, args.camera_width, args.camera_height, args.record_viewport_camera)
        self.reused = self.value is not None and key == self.key
        if not self.reused:
            self.close()
            self.value = build()
            self.key = key
        self.setup_s = time.perf_counter() - started
        return self.value

    def close(self):
        if self.value is not None:
            self.value[0].close()
        self.key = self.value = None


@contextmanager
def job_log(path):
    """Redirect Python and native Kit stdout, then detach before acknowledgment."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    try:
        with open(path, "ab", buffering=0) as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        ctypes.CDLL(None).fflush(None)
        for target, fd in zip((1, 2), saved):
            os.dup2(fd, target)
            os.close(fd)


def serve(runtime, parser, app):
    queue = Path(runtime["args_cli"].batch_queue).resolve()
    cache = EnvironmentCache()
    number = 0
    replace_report(queue / "ready.json", {"pid": os.getpid(), "ready": True,
                                          "startup_s": time.perf_counter() - runtime["_BOOT_STARTED"]})
    try:
        while app.is_running() and not (queue / "stop.json").exists():
            path = queue / f"job_{number:06d}.json"
            if not path.exists():
                time.sleep(0.05)
                continue
            request = json.loads(path.read_text())
            if request.get("schema_version") != 1 or request.get("id") != number:
                raise ValueError("Invalid batch job identity")
            started = time.perf_counter()
            result = {"id": number, "status": "failed", "reused_environment": False}
            with job_log(request["log"]):
                try:
                    cli = parser.parse_args(request["argv"])
                    result["output"] = cli.output
                    if not cli.headless or cli.device != "cuda:0" or cli.batch_queue:
                        raise ValueError("Batch jobs require the selected isolated headless CUDA device")
                    metadata, validation, resolution = runtime["_preflight"](cli)
                    runtime.update(args_cli=cli, recording_metadata=metadata,
                                   omy_validation=validation, omy_task_resolution=resolution)
                    if runtime["main"](env_cache=cache) != 0:
                        raise RuntimeError("Materializer returned a nonzero status")
                    result.update(status="complete", reused_environment=cache.reused, environment_setup_s=cache.setup_s)
                except (Exception, SystemExit) as error:
                    result["error"] = f"{type(error).__name__}: {error}"
                    traceback.print_exc()
                    cache.close()
            result["elapsed_s"] = time.perf_counter() - started
            if result["status"] == "complete":
                replace_report(Path(result["output"]).parent / "materialization_metrics.json", result)
            replace_report(queue / f"result_{number:06d}.json", result)
            number += 1
        return 0
    finally:
        cache.close()
