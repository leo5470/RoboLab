#!/usr/bin/env bash
# Phase 2 step 1: GraspDataGen grasps for the batch objects (robotiq_2f_85).
#
#   CUDA_VISIBLE_DEVICES=N scripts/scripted_demos/grasps/run_graspdatagen.sh [INDEX ...]
#   (default: all seven batch objects; ROBOLAB_VENV defaults to this checkout's .venv; GRASPDATAGEN_DIR to the
#   third_party/GraspDataGen submodule)
#
# Per object: one grasp_guess run, then two grasp_sim runs on the same guesses: with force_magnitude equal to the
# object's simulated mass in kg (GraspDataGen's force_magnitude is in g on its 1 kg test body, grasp_sim.py:1015-1018,
# so this reproduces the object's own weight), and with the default 1.0. This is what datagen.py does (guess + sim),
# split so both validations see identical guesses.
#
# Needs <grasp_data>/meshes (export_meshes.py) and <grasp_data>/masses.json (scripts/scripted_demos/probe_objects.py).
# Outputs: <grasp_data>/gdg/{guess,sim_mass,sim_1g}/grasp_*_data/robotiq_2f_85/<idx>_<name>.yaml and logs.
# Re-running skips objects whose outputs exist (OVERWRITE=1 to redo).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV=${ROBOLAB_VENV:-$REPO/.venv}
DATA=${GRASP_DATA_DIR:-$REPO/output/grasp_data}
GDG=${GRASPDATAGEN_DIR:-$REPO/third_party/GraspDataGen}
[ -d "$GDG/scripts/graspgen" ] || { echo "run_graspdatagen.sh: no GraspDataGen checkout at $GDG; run" \
    "git submodule update --init third_party/GraspDataGen or set GRASPDATAGEN_DIR" >&2; exit 2; }
SEED=${SEED:-20261107}
OVERWRITE=${OVERWRITE:-0}

# Isaac Sim 5.0's extensions need its bundled warp 1.7.1; the venv's pip warp-lang (1.19 dev) lacks wp.types.array
# and GraspDataGen imports warp before the app starts, so the bundled copy goes first on the path.
WARP=$(ls -d "$VENV"/lib/python3.*/site-packages/isaacsim/extscache/omni.warp.core-*)
export PYTHONPATH="$WARP:$GDG/scripts/graspgen"
export OMNI_KIT_ACCEPT_EULA=YES PYTHONDONTWRITEBYTECODE=1
export KIT_PORTABLE_ROOT=${KIT_PORTABLE_ROOT:-$REPO/output/kit_portable}   # see gdg_run.py
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the GPU for GraspDataGen}
GRIPPER="$DATA/bots/robotiq_2f_85.usd"   # a copy, so the gripper cache (.npz) lands in our tree
mkdir -p "$DATA/bots" "$DATA/gdg/logs"
[ -f "$GRIPPER" ] || cp "$GDG/bots/robotiq_2f_85.usd" "$GRIPPER"

if [ $# -gt 0 ]; then INDICES=("$@"); else INDICES=(303 301 283 284 230 231 294); fi
cd "$DATA"   # GraspDataGen resolves some defaults relative to the working directory

for idx in "${INDICES[@]}"; do
    obj=$(ls "$DATA/meshes/${idx}_"*.obj | head -1)
    key=$(basename "$obj" .obj)
    mass=$(python3 -c "import json,sys; print(json.load(open('$DATA/masses.json'))['$key'])")
    num=2048; case $idx in 301|303) num=4096;; esac
    guess="$DATA/gdg/guess/grasp_guess_data/robotiq_2f_85/$key.yaml"
    if [ "$OVERWRITE" = 1 ] || [ ! -f "$guess" ]; then
        echo "[$(date '+%F %T')] $key: grasp_guess num_grasps=$num seed=$SEED"
        GRASP_DATASET_DIR="$DATA/gdg/guess" "$VENV/bin/python" "$REPO/scripts/scripted_demos/grasps/gdg_run.py" "$GDG/scripts/graspgen/grasp_guess.py" \
            --gripper_config robotiq_2f_85 --gripper_file "$GRIPPER" --object_file "$obj" --object_scale 1.0 \
            --num_grasps "$num" --seed "$SEED" --headless > "$DATA/gdg/logs/${key}_guess.log" 2>&1
    fi
    for run in sim_mass sim_1g; do
        force=$mass; [ $run = sim_1g ] && force=1.0
        out="$DATA/gdg/$run/grasp_sim_data/robotiq_2f_85/$key.yaml"
        if [ "$OVERWRITE" = 1 ] || [ ! -f "$out" ]; then
            echo "[$(date '+%F %T')] $key: grasp_sim $run force_magnitude=$force"
            GRASP_DATASET_DIR="$DATA/gdg/$run" "$VENV/bin/python" "$REPO/scripts/scripted_demos/grasps/gdg_run.py" "$GDG/scripts/graspgen/grasp_sim.py" \
                --gripper_config robotiq_2f_85 --gripper_file "$GRIPPER" --object_file "$obj" --object_scale 1.0 \
                --grasp_file "$guess" --force_magnitude "$force" --headless > "$DATA/gdg/logs/${key}_${run}.log" 2>&1
        fi
        grep -o "created [0-9]* successes and [0-9]* fails" "$DATA/gdg/logs/${key}_${run}.log" | tail -1 | sed "s/^/    $run: /"
    done
done
