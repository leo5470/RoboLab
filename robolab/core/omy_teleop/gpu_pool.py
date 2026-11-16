"""Resolve the pipeline's GPU pool from CUDA_VISIBLE_DEVICES to host GPUs.

The isolated launchers (``isolated_gpu_launch``, ``_gpu_slot``) address host
GPUs by nvidia-smi index. ``CUDA_VISIBLE_DEVICES`` instead uses CUDA's own
enumeration, which may differ in order and may name devices by UUID. The CUDA
driver is therefore asked which devices the current environment exposes, and
their UUIDs are matched to nvidia-smi indices. When the variable is unset,
the pool is host GPU 0 only.
"""

import json
import os
import re
import subprocess
import sys

DEFAULT_HOST_GPUS = (0,)

_PROBE = r"""
import ctypes, json
cuda = ctypes.CDLL("libcuda.so.1")
status = cuda.cuInit(0)
if status == 100:  # CUDA_ERROR_NO_DEVICE: the environment exposes nothing
    print("[]")
    raise SystemExit(0)
if status != 0:
    raise SystemExit(f"cuInit failed with CUDA error {status}")
count = ctypes.c_int()
if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
    raise SystemExit("cuDeviceGetCount failed")
get_uuid = getattr(cuda, "cuDeviceGetUuid_v2", None) or cuda.cuDeviceGetUuid
uuids = []
for ordinal in range(count.value):
    device = ctypes.c_int()
    raw = (ctypes.c_ubyte * 16)()
    if cuda.cuDeviceGet(ctypes.byref(device), ordinal) != 0 or get_uuid(raw, device) != 0:
        raise SystemExit(f"Cannot read the UUID of CUDA device {ordinal}")
    h = bytes(raw).hex()
    uuids.append(f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}")
print(json.dumps(uuids))
"""


def host_gpus(runner=subprocess.run):
    """Return {uuid: host index} in nvidia-smi order."""
    result = runner(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                    capture_output=True, text=True, check=True, timeout=15)
    mapping = {}
    for line in result.stdout.splitlines():
        if line.strip():
            index, uuid = (part.strip() for part in line.split(","))
            mapping[uuid] = int(index)
    return mapping


def visible_uuids(env, runner=subprocess.run):
    """UUIDs CUDA exposes under ``env``, in CUDA ordinal order."""
    result = runner([sys.executable, "-I", "-c", _PROBE], capture_output=True, text=True,
                    timeout=60, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Cannot enumerate CUDA devices: {result.stderr.strip() or result.stdout.strip()}")
    return json.loads(result.stdout)


def resolve_gpu_pool(env=None, runner=subprocess.run):
    """Host GPUs for the pipeline: [{"host_index": N, "uuid": "GPU-..."}, ...]."""
    env = dict(os.environ if env is None else env)
    hosts = host_gpus(runner)
    by_index = {index: uuid for uuid, index in hosts.items()}
    if "CUDA_VISIBLE_DEVICES" not in env:
        missing = [i for i in DEFAULT_HOST_GPUS if i not in by_index]
        if missing:
            raise RuntimeError(f"Default host GPU(s) {missing} not present")
        return [{"host_index": i, "uuid": by_index[i]} for i in DEFAULT_HOST_GPUS]
    uuids = visible_uuids(env, runner)
    if not uuids:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']!r} exposes no GPU")
    pool = []
    for uuid in uuids:
        if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", uuid) or uuid not in hosts:
            raise RuntimeError(f"Visible device {uuid} is not a whole host GPU listed by nvidia-smi")
        pool.append({"host_index": hosts[uuid], "uuid": uuid})
    return pool
