#!/usr/bin/env python3
"""Run an Isaac diagnostic on one host GPU, isolated with isolated_gpu_launch.

Usage (from the repository root):
    .venv/bin/python tools/simple_pnp_v2/run_isolated.py cuda:0 tools/simple_pnp_v2/<script>.py [args...]

The child sees only the selected GPU as cuda:0; ``--device cuda:0`` is appended.
Requires bubblewrap (bwrap). Isaac often exits 139 at shutdown after writing its
results: judge outcomes from the written reports, not from this exit code.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robolab.core.omy_postprocess.gpu_isolation import isolated_gpu_launch


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    gpu = isolated_gpu_launch(sys.argv[1])
    command = [*gpu.prefix, sys.executable, "-B", *sys.argv[2:], "--device", gpu.child_device]
    raise SystemExit(subprocess.call(command, env=gpu.env, cwd=ROOT))


if __name__ == "__main__":
    main()
