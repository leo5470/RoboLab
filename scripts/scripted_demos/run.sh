#!/usr/bin/env bash
# Run a scripted-demo script with a RoboLab venv, importing robolab from this checkout. Nothing is installed into the
# venv and no bytecode is written, so another user's venv can be borrowed read-only (ROBOLAB_VENV).
#
#   SIM_GPU=N scripts/scripted_demos/run.sh SCRIPT.py [args...]   Isaac Sim / RoboLab, on host GPU N
#   scripts/scripted_demos/run.sh --pxr SCRIPT.py [args...]       USD (pxr) only: no Isaac app, no GPU
#   scripts/scripted_demos/run.sh --cpu SCRIPT.py [args...]       plain Python (numpy tools, the demo viewer)
#
# Simulator mode: the renderer enumerates GPUs through Vulkan, which ignores CUDA_VISIBLE_DEVICES, so the process runs
# under bubblewrap with a private /dev holding only the chosen GPU, as robolab/core/omy_postprocess/gpu_isolation.py
# does; inside, that GPU is cuda:0. SIM_GPU has no default (pick a GPU no other simulator is using: two simulators on
# a 10 GB card make PhysX fail to allocate and the state silently turns to NaN). SIM_GPU=none skips the isolation.
#
# pxr mode: the USD libraries bundled with the venv's Isaac Sim (as tools/simple_pnp_v2/usd_python.sh does), plus
# matplotlib and the rest of the venv.
#
# ROBOLAB_VENV: the venv to use (default: this checkout's .venv).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV=${ROBOLAB_VENV:-$REPO/.venv}
PY="$VENV/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if [ "${1:-}" = --pxr ]; then
    shift
    shopt -s nullglob
    libs=("$VENV"/lib/python3.*/site-packages/isaacsim/extscache/omni.usd.libs-*)
    [ ${#libs[@]} -eq 1 ] || { echo "run.sh: expected one omni.usd.libs-* in $VENV, found ${#libs[@]}" >&2; exit 2; }
    export PYTHONPATH="${libs[0]}:$PYTHONPATH"
    export LD_LIBRARY_PATH="${libs[0]}/bin:$(dirname "$(readlink -f "$PY")")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    exec "$PY" -u "$@"
fi
if [ "${1:-}" = --cpu ]; then
    shift
    exec "$PY" -u "$@"
fi

export OMNI_KIT_ACCEPT_EULA=YES
unset DISPLAY WAYLAND_DISPLAY
SIM_GPU=${SIM_GPU:?set SIM_GPU to the host GPU index for the simulator (or none)}
[ "$SIM_GPU" = none ] && exec "$PY" -u "$@"
uuid=$(nvidia-smi --id="$SIM_GPU" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
minor=$(nvidia-smi --id="$SIM_GPU" -q | awk -F: '/Minor Number/ && !m {gsub(/ /, "", $2); print $2; m=1}')
[ -n "$uuid" ] && [ -n "$minor" ] || { echo "run.sh: cannot resolve GPU $SIM_GPU" >&2; exit 2; }
devs=(--dev-bind "/dev/nvidia$minor" "/dev/nvidia$minor" --dev-bind /dev/nvidiactl /dev/nvidiactl
      --dev-bind /dev/nvidia-uvm /dev/nvidia-uvm)
[ -e /dev/nvidia-uvm-tools ] && devs+=(--dev-bind /dev/nvidia-uvm-tools /dev/nvidia-uvm-tools)
export CUDA_VISIBLE_DEVICES="$uuid"
exec bwrap --die-with-parent --bind / / --dev /dev "${devs[@]}" -- "$PY" -u "$@"
