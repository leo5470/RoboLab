#!/usr/bin/env bash
# Run a Python script with pxr (USD) outside Isaac, using the USD libraries
# bundled with the project's Isaac Sim install. Plain .venv/bin/python cannot
# import pxr; this avoids adding usd-core (and a uv.lock change).
#
# Usage: tools/simple_pnp_v2/usd_python.sh script.py [args...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
shopt -s nullglob
libs=("$ROOT"/.venv/lib/python3.*/site-packages/isaacsim/extscache/omni.usd.libs-*)
if [ ${#libs[@]} -ne 1 ]; then
    echo "usd_python.sh: expected exactly one omni.usd.libs-* under .venv isaacsim extscache, found ${#libs[@]}" >&2
    exit 2
fi
USD="${libs[0]}"
PYLIB="$(dirname "$(readlink -f "$PY")")/../lib"   # libpython for pxr's compiled modules
export PYTHONPATH="$USD${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$USD/bin:$PYLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$ROOT"
exec "$PY" "$@"
