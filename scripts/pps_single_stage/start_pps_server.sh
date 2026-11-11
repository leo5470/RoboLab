#!/usr/bin/env bash
# Start the PPS steering server used by the jointpos evaluation scripts.
#
# Counterpart of ../pi05_single_stage/start_pi05_server.sh. Same model —
# pi05_droid_jointpos — but served from the PPS repo's PyTorch copy rather than
# openpi's JAX server, because PPS adds gamma*(v_task - v_ref) *inside* the
# denoising loop and the JAX sampler is a jitted lax.while_loop with no hook.
#
# With no --arm and MODE=none this is the frozen base: the wrapper computes
# v + 0.0, which is bitwise v (verified by pps/scripts/wire_identity_droid.py),
# so the base arm and every steered arm are measured through one code path.

set -euo pipefail

PPS_DIR="${PPS_DIR:-/tmp2/leocheng/pps}"
CKPT="${CKPT:-${PPS_DIR}/runs/ckpt/pi05_droid_jointpos_pt}"
PORT="${PORT:-8000}"
# IsaacSim takes cuda:0 by default; keep the 3B base off that card.
SERVER_GPU="${SERVER_GPU:-1}"
MODE="${MODE:-none}"
GAMMA="${GAMMA:-0.0}"
ARM="${ARM:-}"
TASK_LABEL="${TASK_LABEL:-}"
LOG_DIR="${LOG_DIR:-}"

if [[ ! -f "${PPS_DIR}/scripts/serve_pps.py" ]]; then
  echo "PPS was not found at: ${PPS_DIR}" >&2
  echo "Set PPS_DIR to the root of your PPS checkout." >&2
  exit 1
fi
if [[ ! -f "${CKPT}/model.safetensors" ]]; then
  echo "No converted checkpoint at: ${CKPT}" >&2
  echo "Run, in the PPS repo:" >&2
  echo "  uv run python scripts/convert_checkpoint.py \\" >&2
  echo "      --source gs://openpi-assets-simeval/pi05_droid_jointpos \\" >&2
  echo "      --config-name pi05_droid --out ${CKPT}" >&2
  exit 1
fi

cd "${PPS_DIR}"
set -a; . ./.env; set +a
# shellcheck disable=SC1091
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="${SERVER_GPU}"

args=(--ckpt "${CKPT}" --port "${PORT}" --mode "${MODE}" --gamma "${GAMMA}")
if [[ -n "${ARM}" ]]; then
  # ARM is NAME=REF.pt,TASK.pt. Take the text before the FIRST '=' — `##*=` strips through the
  # last one and yields the paths, which then fail the server's "arm is not loaded" check.
  ARM_NAME="${ARM%%=*}"
  if [[ "${ARM_NAME}" == "${ARM}" || -z "${ARM_NAME}" ]]; then
    echo "ARM must look like NAME=REF.pt,TASK.pt (got: ${ARM})" >&2
    exit 1
  fi
  args+=(--arm "${ARM}" --default-arm "${ARM_NAME}")
fi
[[ -n "${TASK_LABEL}" ]] && args+=(--task "${TASK_LABEL}")
[[ -n "${LOG_DIR}" ]]    && args+=(--log-dir "${LOG_DIR}")

exec python scripts/serve_pps.py "${args[@]}" "$@"
