#!/usr/bin/env bash
# PPS's own reader check on a scripted-demo file (Phase 5 gate): the pps repository's
# `prepare_jointpos.py --validate-only`, run with a pps interpreter (read-only; nothing is installed).
#
#   PPS_REPO=<pps checkout> [PPS_PYTHON=<python>] scripts/scripted_demos/validate_pps.sh FILE TASK [OUT_DIR]
#
# PPS_PYTHON defaults to $PPS_REPO/.venv/bin/python; OUT_DIR to <file dir>/pps_validation.
# The scripted files were validated against pps d9c50fd (branch RoboLab of leo5470/pps).
set -euo pipefail
FILE=${1:?usage: validate_pps.sh FILE TASK [OUT_DIR]}
TASK=${2:?usage: validate_pps.sh FILE TASK [OUT_DIR]}
OUT=${3:-$(dirname "$(readlink -f "$FILE")")/pps_validation}
PPS_REPO=${PPS_REPO:?set PPS_REPO to a pps checkout}
PPS_PYTHON=${PPS_PYTHON:-$PPS_REPO/.venv/bin/python}
export PYTHONDONTWRITEBYTECODE=1
exec "$PPS_PYTHON" "$PPS_REPO/scripts/prepare_jointpos.py" \
    --source omy-hdf5 --layout episodes --input "$FILE" --task "$TASK" --validate-only --out "$OUT"
