#!/usr/bin/env bash
# Evaluate pi05_droid_jointpos through the PPS server on the single-stage surface-placement task set.
#
# Counterpart of ../pi05_single_stage/run_surface_placement.sh, task-for-task and flag-for-flag,
# so the numbers are comparable with output/pi05_single_stage_5eps. Two differences, both
# deliberate:
#
#   * policies/pps/run.py instead of policies/pi0_family/run.py — same env registration and
#     the same eval loop, plus the steering keys. With --mode none it is the frozen base.
#   * NUM_ENVS defaults to 5 rather than 1, which is what the 5eps baseline actually ran.
#
# Set MODE/GAMMA/ARM to sweep a steered arm over the same task set; the output folder and
# every result row carry the arm, so two gamma points cannot be confused after the fact.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROBOLAB_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# Local checkout by default; the pi05 scripts assume the Docker image's launcher instead.
ISAAC_PYTHON="${ISAAC_PYTHON:-${ROBOLAB_DIR}/.venv/bin/python}"
REMOTE_HOST="${REMOTE_HOST:-localhost}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
NUM_ENVS="${NUM_ENVS:-5}"
NUM_RUNS="${NUM_RUNS:-1}"
MODE="${MODE:-none}"
GAMMA="${GAMMA:-0.0}"
ARM="${ARM:-}"
# The 5eps baseline wrote viewport videos only; match it so wall time is comparable too.
VIDEO_MODE="${VIDEO_MODE:-viewport}"
# The default folder encodes the arm. run_evaluation resumes by SKIPPING tasks already
# complete in the folder, so a second sweep sharing a name would report the first sweep's
# numbers rather than run. Unsteered keeps the historical name so existing results resume.
if [[ "${MODE}" == "none" ]]; then
  DEFAULT_OUTPUT="pps_jointpos_surface_placement_5eps"
else
  DEFAULT_OUTPUT="pps_jointpos_surface_placement_${MODE}_g${GAMMA}${ARM:+_${ARM%%=*}}_5eps"
fi
OUTPUT_FOLDER_NAME="${OUTPUT_FOLDER_NAME:-${DEFAULT_OUTPUT}}"

if [[ ! -x "${ISAAC_PYTHON}" ]]; then
  echo "Isaac Sim Python launcher is not executable: ${ISAAC_PYTHON}" >&2
  echo "Set ISAAC_PYTHON if this checkout uses a different path." >&2
  exit 1
fi

cd "${ROBOLAB_DIR}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"

args=(
  --remote-host "${REMOTE_HOST}"
  --remote-port "${REMOTE_PORT}"
  --mode "${MODE}"
  --gamma "${GAMMA}"
  --headless
  --num-envs "${NUM_ENVS}"
  --num-runs "${NUM_RUNS}"
  --video-mode "${VIDEO_MODE}"
  --output-folder-name "${OUTPUT_FOLDER_NAME}"
)
[[ -n "${ARM}" ]] && args+=(--arm "${ARM}")

exec "${ISAAC_PYTHON}" policies/pps/run.py "${args[@]}" \
  --task \
    FruitsOnionToPlateTask \
    KeyboardOutOfBinTask \
    TakeSpatulaOffShelfTask \
    ToolsPickingHammerTask \
    ToolsPickingDrillTask \
    OneBottleOnShelfTask \
  "$@"
