#!/usr/bin/env bash
# 100 episodes per task on BOTH backends, unattended.
#
# The 5-episode comparison could not separate the JAX and PyTorch bases (pooled p = 0.20) even
# though eight of the nine tasks that moved moved down. This runs the same 22 tasks at 100
# episodes each so that question is settled, and so every task's base rate is known to about
# +/-10 points instead of +/-40 — which is what picking a steering target actually needs.
#
# Both arms are the frozen base. Neither steers: the PyTorch arm runs --mode none, and the JAX
# arm is openpi's own server, exactly as the published baseline ran it.
#
#   arm       server                                   client
#   pytorch   pps/scripts/serve_pps.py (--mode none)   policies/pps/run.py
#   jax       openpi scripts/serve_policy.py           policies/pi0_family/run.py --policy pi05
#
# Resumable: both arms write to fixed output folders and RoboLab skips runs it finds already
# complete, so re-running this script after a crash continues where it stopped. Videos are off
# (~1 MB/episode of HDF5 instead of ~4 MB with them), and nothing else about the protocol changes.
#
#   ./scripts/pps_single_stage/run_100ep_both_backends.sh
#   PARALLEL=0 ./scripts/pps_single_stage/run_100ep_both_backends.sh   # one arm at a time
#   ARMS=pytorch ./scripts/pps_single_stage/run_100ep_both_backends.sh # just one arm

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROBOLAB_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PPS_DIR="${PPS_DIR:-/tmp2/leocheng/pps}"
# The vendored RoboLab/.openpi checkout is root-owned and its venv is unusable for a normal
# user; the standalone fork clone is the one RoboLab's README actually tells you to serve from.
OPENPI_DIR="${OPENPI_DIR:-/tmp2/leocheng/forks/openpi}"
LOG_DIR="${LOG_DIR:-${ROBOLAB_DIR}/output/logs_100ep}"

# Episodes per task = NUM_ENVS * NUM_RUNS. Both must divide 100 exactly, or the arms end up with
# different trial counts than the label claims.
# MEASURED, not guessed (scaling probe, MustardInLeftBinTask):
#   envs   env_step   policy    s/episode
#      5    181 ms     91 ms      24.5
#     15    243 ms    214 ms      13.7
#     30    379 ms    479 ms      12.9
# IsaacSim batches physics and tiled rendering well (6x the envs costs 2.1x the step), but policy
# inference is serial — one request per env per replan — so it scales LINEARLY and overtakes the
# simulator by 30 envs. That is why per-episode cost saturates at ~1.9x and why more envs stop
# helping. The fix is a batched server, which is real work and not something to start unattended.
#
# VRAM (same probe): 10.4 GB at 5 envs, 15.4 at 15, 22.4 at 30 — about 8 GB fixed plus 0.48 GB
# per env, and the heaviest tasks here (the 70-env class in docs/env_vram_size_guide.md) want
# ~1.4x that. With a 8 GB server sharing the card, 10 envs is the ceiling that still leaves
# headroom on the worst task.
NUM_ENVS="${NUM_ENVS:-10}"
# Waves. Each is a COMPLETE, balanced comparison at that depth, and RoboLab resumes by skipping
# runs already recorded — so this can be stopped at any wave boundary and the data is still valid.
# 20 episodes/task lands in ~4-5 h, 100 in ~20-23 h.
WAVES="${WAVES:-2 6 10}"

ARMS="${ARMS:-pytorch jax}"
PARALLEL="${PARALLEL:-1}"       # 1 = one arm per GPU, concurrently
PYTORCH_GPU="${PYTORCH_GPU:-0}"
JAX_GPU="${JAX_GPU:-1}"
PYTORCH_PORT="${PYTORCH_PORT:-8000}"
JAX_PORT="${JAX_PORT:-8001}"
# JAX preallocates. It shares its card with IsaacSim here, so cap it well below half.
XLA_MEMORY_FRACTION="${XLA_MEMORY_FRACTION:-0.30}"

TASKS=(
  # 70-env class in docs/env_vram_size_guide.md — the memory-hungriest, so they run first
  CoffeePotInBinTask ThrowAwayAppleTask ToolsPickingHammerTask ToolsPickingDrillTask
  OneBottleOnShelfTask
  # 80-env class
  BigPumpkinInBinTask SmallPumpkinInBinTask ToyInBinTask YogurtInBowlTask MarkerInMugTask
  SpoonInMugTask FruitsOnionTask FruitsOnionToPlateTask KeyboardOutOfBinTask
  TakeSpatulaOffShelfTask
  # 100-env class
  SauceBottlesCrateTask MustardInLeftBinTask MustardInRightBinTask FoodPacking1BoxesTask
  FoodPacking1CansTask SmallerObjectButterInBinTask LargerObjectRaisinBoxInBinTask
)

mkdir -p "${LOG_DIR}"
SERVER_PIDS=()
cleanup() { for pid in "${SERVER_PIDS[@]:-}"; do kill "${pid}" 2>/dev/null; done; }
trap cleanup EXIT INT TERM

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG_DIR}/driver.log"; }

wait_for_server() {  # $1 log file, $2 label
  for _ in $(seq 1 120); do
    grep -q "server listening on" "$1" 2>/dev/null && return 0
    sleep 5
  done
  say "TIMEOUT waiting for the $2 server; see $1"
  return 1
}

start_pytorch_server() {
  say "starting PyTorch (PPS) server on GPU ${PYTORCH_GPU}, port ${PYTORCH_PORT}"
  ( cd "${PPS_DIR}" && set -a && . ./.env && set +a \
    && source .venv/bin/activate \
    && CUDA_VISIBLE_DEVICES="${PYTORCH_GPU}" python scripts/serve_pps.py \
         --port "${PYTORCH_PORT}" --mode none ) \
    > "${LOG_DIR}/server_pytorch.log" 2>&1 &
  SERVER_PIDS+=($!)
  wait_for_server "${LOG_DIR}/server_pytorch.log" "PyTorch"
}

start_jax_server() {
  say "starting JAX (openpi) server on GPU ${JAX_GPU}, port ${JAX_PORT}"
  ( cd "${OPENPI_DIR}" \
    && CUDA_VISIBLE_DEVICES="${JAX_GPU}" \
       XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEMORY_FRACTION}" \
       uv run scripts/serve_policy.py \
         --port "${JAX_PORT}" \
         policy:checkpoint \
         --policy.config=pi05_droid_jointpos \
         --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos ) \
    > "${LOG_DIR}/server_jax.log" 2>&1 &
  SERVER_PIDS+=($!)
  wait_for_server "${LOG_DIR}/server_jax.log" "JAX"
}

# One arm, all tasks. Per task rather than one long invocation so a single failing task cannot
# take the other twenty-one with it, and so progress is visible in the driver log.
run_arm() {  # $1 arm, $2 gpu, $3 port, $4 num_runs
  local arm="$1" gpu="$2" port="$3" runs="$4"
  local folder="pps100_${arm}"
  local runner=(policies/pps/run.py --mode none)
  [[ "${arm}" == "jax" ]] && runner=(policies/pi0_family/run.py --policy pi05)

  for task in "${TASKS[@]}"; do
    local log="${LOG_DIR}/${arm}_${task}.log"
    for attempt in 1 2; do
      say "${arm}: ${task} (${runs} runs x ${NUM_ENVS} envs, attempt ${attempt})"
      ( cd "${ROBOLAB_DIR}" \
        && OMNI_KIT_ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES="${gpu}" \
           "${ROBOLAB_DIR}/.venv/bin/python" "${runner[@]}" \
             --remote-host localhost --remote-port "${port}" \
             --headless --num-envs "${NUM_ENVS}" --num-runs "${runs}" \
             --video-mode none \
             --output-folder-name "${folder}" \
             --task "${task}" ) >> "${log}" 2>&1
      local rc=$?
      if [[ ${rc} -eq 0 ]]; then break; fi
      say "${arm}: ${task} exited ${rc} on attempt ${attempt}; see ${log}"
    done
  done
  say "${arm}: wave of ${runs} runs done"
}

say "=== both-backend run: ${NUM_ENVS} envs, waves at runs = ${WAVES} ==="
say "arms: ${ARMS} | parallel: ${PARALLEL}"

for arm in ${ARMS}; do
  if [[ "${arm}" == "pytorch" ]] && ! start_pytorch_server; then
    say "FATAL: the PyTorch server did not come up — aborting rather than failing every task"
    exit 1
  fi
  if [[ "${arm}" == "jax" ]] && ! start_jax_server; then
    say "FATAL: the JAX server did not come up — aborting rather than failing every task"
    exit 1
  fi
done

compare() {
  ( cd "${ROBOLAB_DIR}" && source "${PPS_DIR}/.venv/bin/activate" \
    && python scripts/pps_single_stage/compare_to_baseline.py \
         --baseline output/pps100_jax --arm output/pps100_pytorch \
         --markdown output/pps100_pytorch/vs_jax_100ep.md ) \
    > "${LOG_DIR}/comparison_$1ep.log" 2>&1
  say "wave $1: comparison written to ${LOG_DIR}/comparison_$1ep.log"
}

for runs in ${WAVES}; do
  depth=$((runs * NUM_ENVS))
  say "--- wave: ${depth} episodes/task ---"
  if [[ "${PARALLEL}" == "1" && "${ARMS}" == *pytorch* && "${ARMS}" == *jax* ]]; then
    run_arm pytorch "${PYTORCH_GPU}" "${PYTORCH_PORT}" "${runs}" &
    A=$!
    run_arm jax "${JAX_GPU}" "${JAX_PORT}" "${runs}" &
    B=$!
    wait "${A}" "${B}"
  else
    for arm in ${ARMS}; do
      [[ "${arm}" == "pytorch" ]] && run_arm pytorch "${PYTORCH_GPU}" "${PYTORCH_PORT}" "${runs}"
      [[ "${arm}" == "jax" ]] && run_arm jax "${JAX_GPU}" "${JAX_PORT}" "${runs}"
    done
  fi
  compare "${depth}"
done

cleanup
say "=== both arms finished ==="
say "=== done ==="
