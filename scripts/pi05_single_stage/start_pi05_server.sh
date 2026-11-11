#!/usr/bin/env bash
# Start the OpenPI Pi0.5 policy server used by the RoboLab evaluation scripts.

set -euo pipefail

OPENPI_DIR="${OPENPI_DIR:-/workspace/openpi}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_droid_jointpos}"
POLICY_DIR="${POLICY_DIR:-gs://openpi-assets-simeval/pi05_droid_jointpos}"
XLA_MEMORY_FRACTION="${XLA_MEMORY_FRACTION:-0.5}"

if [[ ! -f "${OPENPI_DIR}/scripts/serve_policy.py" ]]; then
  echo "OpenPI was not found at: ${OPENPI_DIR}" >&2
  echo "Set OPENPI_DIR to the root of your OpenPI checkout." >&2
  exit 1
fi

cd "${OPENPI_DIR}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEMORY_FRACTION}"

exec uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config="${POLICY_CONFIG}" \
  --policy.dir="${POLICY_DIR}" \
  "$@"
