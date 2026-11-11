#!/usr/bin/env bash
# Evaluate Pi0.5 on the single-stage surface-placement task set.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROBOLAB_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
REMOTE_HOST="${REMOTE_HOST:-localhost}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
NUM_ENVS="${NUM_ENVS:-1}"
NUM_RUNS="${NUM_RUNS:-1}"
OUTPUT_FOLDER_NAME="${OUTPUT_FOLDER_NAME:-pi05_single_stage_surface}"

if [[ ! -x "${ISAAC_PYTHON}" ]]; then
  echo "Isaac Sim Python launcher is not executable: ${ISAAC_PYTHON}" >&2
  echo "Set ISAAC_PYTHON if the container uses a different path." >&2
  exit 1
fi

cd "${ROBOLAB_DIR}"

exec "${ISAAC_PYTHON}" policies/pi0_family/run.py \
  --policy pi05 \
  --remote-host "${REMOTE_HOST}" \
  --remote-port "${REMOTE_PORT}" \
  --headless \
  --num-envs "${NUM_ENVS}" \
  --num-runs "${NUM_RUNS}" \
  --output-folder-name "${OUTPUT_FOLDER_NAME}" \
  --task \
    FruitsOnionToPlateTask \
    KeyboardOutOfBinTask \
    TakeSpatulaOffShelfTask \
    ToolsPickingHammerTask \
    ToolsPickingDrillTask \
    OneBottleOnShelfTask \
  "$@"
