"""Expose only the selected host GPU to an offline renderer on Linux."""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class GpuLaunch:
    prefix: list[str]
    env: dict[str, str]
    child_device: str
    metadata: dict


def _query_gpu(index):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id", str(index), "--query", "--xml-format"],
            capture_output=True, text=True, check=True, timeout=15,
        )
        devices = ET.fromstring(result.stdout).findall("gpu")
        if len(devices) != 1:
            raise ValueError("Expected exactly one selected GPU")
        uuid = devices[0].findtext("uuid", "").strip()
        minor = devices[0].findtext("minor_number", "").strip()
        if (not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", uuid)
                or not re.fullmatch(r"[0-9]+", minor)):
            raise ValueError("GPU UUID or device minor number is missing or invalid")
        return uuid, int(minor)
    except (OSError, ValueError, ET.ParseError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"Cannot resolve host GPU {index} for isolated rendering: {error}") from error


def _device_paths(minor, *, dev_root=Path("/dev")):
    required = [dev_root / f"nvidia{minor}", dev_root / "nvidiactl", dev_root / "nvidia-uvm"]
    for path in required:
        if not path.exists():
            raise RuntimeError(f"Required GPU device is unavailable: {path}")
    tools = dev_root / "nvidia-uvm-tools"
    return [*required, *([tools] if tools.exists() else [])]


def isolated_gpu_launch(device):
    """Prepare a device namespace; no fallback may expose another GPU.

    The requested index is in the host's nvidia-smi order. CUDA and Kit see
    only one GPU inside the child, so both must use local index zero. UUID
    selection is retained to avoid depending on CUDA's device enumeration.
    """
    match = re.fullmatch(r"cuda(?::([0-9]+))?", device)
    if match is None and device != "cpu":
        raise ValueError("Rendering device must be cpu, cuda or cuda:N")
    index = int(match.group(1) or 0) if match else 0
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError("Isolated OMY rendering requires bubblewrap (bwrap); renderer not started")
    uuid, minor = _query_gpu(index)
    # The filesystem stays available, but /dev is private. No other GPU or
    # DRM render node is mounted into it. These shared NVIDIA control nodes
    # are required to operate the one selected GPU device.
    prefix = [bwrap, "--die-with-parent", "--bind", "/", "/", "--dev", "/dev"]
    for path in _device_paths(minor):
        prefix.extend(["--dev-bind", str(path), str(path)])
    prefix.append("--")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = uuid
    # Headless processing must not connect to the collection's display server.
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    child_device = "cpu" if device == "cpu" else "cuda:0"
    return GpuLaunch(prefix, env, child_device, {
        "isolation": "device_namespace", "host_device": f"cuda:{index}",
        "gpu_uuid": uuid, "device_node": f"/dev/nvidia{minor}", "child_device": child_device,
    })
