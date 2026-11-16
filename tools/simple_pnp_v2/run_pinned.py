#!/usr/bin/env python3
"""Run an Isaac script pinned to one host GPU *without* device isolation.

Fallback for hosts where bubblewrap cannot create user namespaces (for example
Ubuntu 24.04's kernel.apparmor_restrict_unprivileged_userns=1). Unlike
run_isolated.py, the child can still enumerate every GPU through Vulkan, so
this only *pins* it: CUDA sees one GPU (by UUID), and Kit's renderer is told to
use the matching single GPU with multi-GPU rendering disabled. Check
``nvidia-smi`` to confirm which GPUs the process actually touches; the
launch metadata (written next to --output when given) records the pinning.

Usage (from the repository root):
    .venv/bin/python tools/simple_pnp_v2/run_pinned.py cuda:1 tools/simple_pnp_v2/<script>.py [args...]
"""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robolab.core.omy_postprocess.gpu_isolation import _query_gpu  # noqa: E402


def main():
    if len(sys.argv) < 3 or not sys.argv[1].startswith("cuda:"):
        raise SystemExit(__doc__)
    index = int(sys.argv[1].split(":", 1)[1])
    uuid, minor = _query_gpu(index)
    args = sys.argv[2:]
    kit = f"--/renderer/multiGpu/enabled=false --/renderer/activeGpu={index}"
    if "--kit_args" in args:
        i = args.index("--kit_args")
        args[i + 1] = f"{kit} {args[i + 1]}"
    else:
        args += ["--kit_args", kit]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)
    env.pop("DISPLAY", None); env.pop("WAYLAND_DISPLAY", None)
    meta = {"isolation": "none (pinned)", "host_device": f"cuda:{index}", "gpu_uuid": uuid,
            "device_node": f"/dev/nvidia{minor}", "child_device": "cuda:0", "kit_pinning": kit}
    print("run_pinned:", json.dumps(meta), flush=True)
    command = [sys.executable, "-B", *args, "--device", "cuda:0"]
    # Replace this process so a caller's timeout/signal reaches Isaac directly
    # (a wrapper child would be orphaned and keep its GPU memory).
    os.chdir(ROOT)
    os.execve(sys.executable, command, env)


if __name__ == "__main__":
    main()
