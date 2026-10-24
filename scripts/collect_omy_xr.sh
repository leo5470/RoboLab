#!/usr/bin/env bash
# Optimized L100 + Quest collection with per-task and object-layout modes.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/collect_omy_xr.sh TASK [--num-episodes N] [OPTIONS]
       scripts/collect_omy_xr.sh --object NAME [--object-id ID] [--num-episodes N] [--episodes-per-layout N] [OPTIONS]
       scripts/collect_omy_xr.sh --object-id ID [--num-episodes N] [--episodes-per-layout N] [OPTIONS]
       scripts/collect_omy_xr.sh --resume SESSION_DIR [OPTIONS]

Collect one task, or all retained layouts for one single_pnp object, using the
optimized collection profile. Object mode keeps the environment and XR session alive.
Run from the host desktop with SteamVR/ALVR running and the leader connected.

  --object NAME     Collect all retained layouts, e.g. canned_mushrooms.
  --object-id ID    Catalog ID, e.g. 034; can disambiguate --object NAME.
  --episodes-per-layout N
                    Successful episodes per layout in object mode (default: 1).
  --resume DIR      Resume pending layouts using the stored quota and profile.
  --num-episodes N   Total successful episodes (TASK default: 10; object default: all layouts).
  --output-dir DIR  Explicit output directory for a new run.
  --reset-key KEY   READY/retry key (default: F9).
  --next-layout-key KEY
                    Skip to next layout in object mode (default: F12).
  --device cpu|cuda|cuda:N
                    Physics backend (new runs: cuda:0; resume: stored).
                    CPU is an opt-in backend experiment and uses ordinary stepping.
  --solver-position-iterations N
                    Physics solver cap: 16, 32, or 64 (new runs: 16; resume: stored).
  --gpu-max-num-partitions N
                    GPU solver partition limit: 1, 2, 4, 8, 16, 32 (default: 8).
                    Try 1 or 2 with 32 iterations for the single-robot candidate.
  --stats-interval S
                    Seconds between periodic reports (default: 1; 0 disables the whole block).
  --log             Also save console output to a unique /tmp/omy_collect_fast_*.log.
  --no-log          Console output only (default).
  --no-postprocess  Skip automatic validation, packaging and camera rendering.
  --postprocess-device cuda:N
                    Isolated host GPU (default: cuda:1; collection defaults to cuda:0).
  --postprocess-skip-failed
                    Package/render only successful attempts; raw failures are saved.
  --postprocess-preview-videos
                    Encode success clips and a combined review reel on CPU.
  --scene-settle-min-s S / --scene-settle-stable-s S / --scene-settle-timeout-s S
                    Preparation durations in simulated seconds (0.5 / 0.3 / 5).
  --scene-settle-linear-m-s V / --scene-settle-angular-rad-s V
                    Maximum prepared speeds (0.01 m/s / 0.5 rad/s), plus pose stability.
  -h, --help        Show this help.

Demonstrations and collection metadata are saved in both modes under
output/omy_teleop/YYYYMMDD_HHMMSS_TASKNAME/ using the host's local time.
Failed/reset attempts are saved but do not count toward the success quota.
New runs allow 120 simulated seconds per attempt (1,800 commands at 15 Hz).
In the focused Isaac Sim window, press F9 at READY, then move to start.
Press F9 during recording to save the attempt and retry the same layout.
In object mode, success advances after the quota; F12 skips the layout.
Skipped layouts remain pending for a later --resume. Every change returns to READY.
After collection ends or Ctrl+C completes cleanup, postprocessing starts automatically.
Postprocessing uses GPU 1; start another collection on GPU 0 in a separate terminal.
The absolute processed dataset, manifest and report locations are printed when done.

Examples:
  scripts/collect_omy_xr.sh SimplePnp034CannedMushrooms24Task --log
  scripts/collect_omy_xr.sh BananaInBowlTask --num-episodes 25 --no-log
  scripts/collect_omy_xr.sh --object canned_mushrooms --episodes-per-layout 10
  scripts/collect_omy_xr.sh --object milkjug_a01 --num-episodes 10
  scripts/collect_omy_xr.sh --object-id 034
  scripts/collect_omy_xr.sh --resume output/omy_teleop/SESSION
EOF
}

require_value() {
  if [[ -z "${2:-}" || "${2:-}" == -* ]]; then
    printf '%s requires a value.\n' "$1" >&2
    exit 2
  fi
}

task=""
object_name=""
object_id=""
resume_dir=""
output_dir=""
reset_key=""
next_layout_key=""
log_console=false
postprocess_args=()
postprocess_device="cuda:1"
preparation_args=()
num_episodes=10
num_episodes_set=false
episodes_per_layout=1
episodes_per_layout_set=false
solver_position_iterations=16
solver_position_iterations_set=false
physics_device="cuda:0"
physics_device_set=false
gpu_partitions=""
stats_interval=1
while (( $# )); do
  argument="$1"
  case "$argument" in
    --object)
      require_value "$argument" "${2:-}"
      object_name="$2"
      shift
      ;;
    --object-id)
      if [[ ! "${2:-}" =~ ^[0-9]+$ ]]; then
        printf '%s\n' '--object-id requires a nonnegative catalog ID.' >&2
        exit 2
      fi
      object_id="$2"
      shift
      ;;
    --resume|--output-dir|--reset-key|--next-layout-key|--stats-interval)
      require_value "$argument" "${2:-}"
      case "$argument" in
        --resume) resume_dir="$2" ;;
        --output-dir) output_dir="$2" ;;
        --reset-key) reset_key="$2" ;;
        --next-layout-key) next_layout_key="$2" ;;
        --stats-interval) stats_interval="$2" ;;
      esac
      shift
      ;;
    --num-episodes)
      if [[ ! "${2:-}" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s\n' '--num-episodes requires a positive integer.' >&2
        exit 2
      fi
      num_episodes="$2"
      num_episodes_set=true
      shift
      ;;
    --episodes-per-layout)
      if [[ ! "${2:-}" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s\n' '--episodes-per-layout requires a positive integer.' >&2
        exit 2
      fi
      episodes_per_layout="$2"
      episodes_per_layout_set=true
      shift
      ;;
    --device)
      if [[ "${2:-}" != cpu && "${2:-}" != cuda && ! "${2:-}" =~ ^cuda:[0-9]+$ ]]; then
        printf '%s\n' '--device requires cpu, cuda, or cuda:N.' >&2
        exit 2
      fi
      physics_device="$2"
      physics_device_set=true
      shift
      ;;
    --solver-position-iterations)
      case "${2:-}" in
        16|32|64) solver_position_iterations="$2" ;;
        *) printf '%s\n' '--solver-position-iterations requires 16, 32, or 64.' >&2; exit 2 ;;
      esac
      solver_position_iterations_set=true
      shift
      ;;
    --gpu-max-num-partitions)
      case "${2:-}" in
        1|2|4|8|16|32) gpu_partitions="$2" ;;
        *) printf '%s\n' '--gpu-max-num-partitions requires 1, 2, 4, 8, 16, or 32.' >&2; exit 2 ;;
      esac
      shift
      ;;
    --log) log_console=true ;;
    --no-log) log_console=false ;;
    --no-postprocess|--postprocess-skip-failed|--postprocess-preview-videos) postprocess_args+=("$argument") ;;
    --scene-settle-min-s|--scene-settle-stable-s|--scene-settle-timeout-s|--scene-settle-linear-m-s|--scene-settle-angular-rad-s)
      require_value "$argument" "${2:-}"
      preparation_args+=("$argument" "$2")
      shift
      ;;
    --postprocess-device)
      if [[ ! "${2:-}" =~ ^cuda:[0-9]+$ ]]; then
        printf '%s\n' '--postprocess-device requires cuda:N.' >&2
        exit 2
      fi
      postprocess_device="$2"
      shift
      ;;
    -h|--help) usage; exit 0 ;;
    -*) printf 'Unknown option: %s\n' "$argument" >&2; usage >&2; exit 2 ;;
    *)
      if [[ -n "$task" ]]; then
        printf 'Expected one task; unexpected argument: %s\n' "$argument" >&2
        usage >&2
        exit 2
      fi
      task="$argument"
      ;;
  esac
  shift
done

selection_count=0
[[ -z "$task" ]] || selection_count=$((selection_count + 1))
[[ -z "$object_name" && -z "$object_id" ]] || selection_count=$((selection_count + 1))
[[ -z "$resume_dir" ]] || selection_count=$((selection_count + 1))
if (( selection_count != 1 )); then
  printf '%s\n' 'Select exactly one TASK, object (--object and/or --object-id), or --resume session.' >&2
  usage >&2
  exit 2
fi

selection_args=()
if [[ -n "$task" ]]; then
  if [[ "$episodes_per_layout_set" == true ]]; then
    printf '%s\n' '--episodes-per-layout requires object mode.' >&2
    exit 2
  fi
  selection_args=(--task "$task" --num-episodes "$num_episodes")
elif [[ -n "$resume_dir" ]]; then
  if [[ "$num_episodes_set" == true || "$episodes_per_layout_set" == true || -n "$output_dir" ]]; then
    printf '%s\n' '--resume uses the stored quota and output directory; omit quota and --output-dir options.' >&2
    exit 2
  fi
  [[ "$resume_dir" == /* ]] || resume_dir="$PWD/$resume_dir"
  selection_args=(--resume "$resume_dir")
else
  [[ "$num_episodes_set" != true ]] || selection_args+=(--num-episodes "$num_episodes")
  [[ -z "$object_name" ]] || selection_args+=(--object "$object_name")
  [[ -z "$object_id" ]] || selection_args+=(--object-id "$object_id")
  selection_args+=(--episodes-per-layout "$episodes_per_layout")
fi

if [[ -n "$output_dir" ]]; then
  [[ "$output_dir" == /* ]] || output_dir="$PWD/$output_dir"
  selection_args+=(--output-dir "$output_dir")
fi
[[ -z "$reset_key" ]] || selection_args+=(--reset-key "$reset_key")
[[ -z "$next_layout_key" ]] || selection_args+=(--next-layout-key "$next_layout_key")

robolab_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$robolab_root"
export PYTHONPATH="$robolab_root${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$robolab_root/.venv/bin/python" ]]; then
  printf 'Python environment not found: %s/.venv/bin/python\n' "$robolab_root" >&2
  exit 1
fi

profile_args=()
if [[ -z "$resume_dir" ]]; then
  profile_args=(
    --source serial --calib "$robolab_root/omy_calib_robolab.json"
    --device "$physics_device" --control-hz 15
    --solver-position-iterations "$solver_position_iterations"
    --stepping-profile held-target --recording-profile compact
    --contact-sensors required --contact-history-length 0
    --render-interval 4 --vel-scale 2 --episode-length-s 120
    --max-attempt-seconds 120 --max-attempt-steps 1800
  )
else
  # Defaults must not override the saved run contract. An explicit selection
  # remains visible to Python so conflicting resume settings are rejected.
  if [[ "$solver_position_iterations_set" == true ]]; then
    profile_args+=(--solver-position-iterations "$solver_position_iterations")
  fi
  if [[ "$physics_device_set" == true ]]; then
    profile_args+=(--device "$physics_device")
  fi
fi

[[ -z "$gpu_partitions" ]] || profile_args+=(--gpu-max-num-partitions "$gpu_partitions")

collector_command=(
  "$robolab_root/.venv/bin/python" -u "$robolab_root/examples/collect_omy_xr.py"
  --task-dirs single_pnp benchmark
  "${selection_args[@]}"
  "${profile_args[@]}"
  "${preparation_args[@]}"
  --port /dev/robotis_left --render-gpu 0
  --cpu-workers 8 --physics-workers 2
  --rendering_mode performance
  --xr-resolution-multiplier 0.8
  --stats-interval "$stats_interval"
)

supervisor_args=("${postprocess_args[@]}" --postprocess-device "$postprocess_device")
if [[ "$log_console" == true ]]; then
  log_path="$(mktemp "/tmp/omy_collect_fast_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
  supervisor_args+=(--log "$log_path")
fi
# The supervisor remains alive through collector shutdown and postprocessing.
# It owns logging so Ctrl+C cannot terminate a tee pipeline during finalization.
exec "$robolab_root/.venv/bin/python" -u "$robolab_root/scripts/supervise_omy_collection.py" \
  "${supervisor_args[@]}" -- "${collector_command[@]}"
