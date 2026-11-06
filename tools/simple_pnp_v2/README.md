# Simple-PnP v2 diagnostics

Planning and investigation tools for [the v2 plan](../../docs/simple_pnp_v2_plan.md).
Diagnostics plus the v2 pipeline (freeze → generate → task drafts → qualify →
visibility). Run everything from the repository root. Write outputs under
`output/simple_pnp_v2_validation/`, which is gitignored.

| Tool | Needs | Purpose |
| --- | --- | --- |
| `usd_python.sh` | project `.venv` with Isaac Sim | Run a script with `pxr` outside Isaac, using Isaac's bundled USD libraries (plain `.venv/bin/python` cannot import `pxr`). |
| `handle_directions.py` | `pxr` | Preliminary handle directions from mesh outlines, in each asset's composed frame. Confirm with renders before freezing. |
| `collision_bottoms.py` | `pxr` | Collision-shape vs visual bottoms, collision approximations/offsets, and root rotations scenes must preserve. Exit 1 if any collision shape lies below its visual bottom. |
| `reach_map.py` | `pxr` | Kinematic top-down reach map for the DROID Franka + Robotiq 2F-85. FK is checked against the authored reset pose. Ignores collisions. About 3 minutes for three heights. |
| `capacity_estimate.py` | numpy | Rough Monte Carlo estimate of per-entry layout capacity under the table, clearance, strict-novelty, and optional reach constraints. |
| `run_isolated.py` | `bwrap`, a GPU | Run an Isaac script on one host GPU via `isolated_gpu_launch`. |
| `start_probe.py` | Isaac, a GPU | Reset, prepare, and hold scenes in the OMY collection profile across seeds and clearances, optionally with a ground-plane candidate copy. Records every rigid body, including the table. |
| `analyze_start_probe.py` | numpy | Summarise probes against the table's current top. Compare cases within a run and, optionally, across two runs. |
| `run_pinned.py` | a GPU | Fallback for `run_isolated.py` when `bwrap` cannot create user namespaces (Ubuntu 24.04 `kernel.apparmor_restrict_unprivileged_userns=1`): CUDA limited to one GPU by UUID and Kit's renderer pinned to it. Not device isolation; check `nvidia-smi`. |
| `freeze_inputs.py` | `pxr` | Step 1: allowlist, payload/dependency hashes, bin pin, exact payload geometry and root rotations, table/mount/ground, and every v1 layout (current + removed) as footprint centres. |
| `handle_regions.json` | — | Handle classes, directions and regions confirmed from three-view point plots (replaces the outline estimates, which were wrong for most class-A entries). |
| `handle_records.py` | `pxr` | Handle point sets in the v2 rest frame from `handle_regions.json`. |
| `plot_handles.py` | `pxr`, matplotlib | Three-view plots of each handle record for confirmation. |
| `generate_scenes.py` | `pxr` | Step 3: feasible-set sampling (support, clearance, reach, v1 novelty, v2 diversity, handle yaw + bin-occlusion screen), USD authoring with root rotations preserved and contact starts, independent re-audit, one record per slot. |
| `generate_tasks.py` | — | Step 5: task drafts in `robolab/tasks/single_pnp_v2_drafts/` from authored records; with `--qualified gate_status.json`, the promoted tasks in `robolab/tasks/single_pnp_v2/` (refuses unless every slot passed every gate). |
| `gate_status.py` | — | Per-slot gate status verified against each scene hash (`production/gate_status.json`), input to `generate_tasks.py --qualified`. |
| `success_check.py` | Isaac | Task goal semantics: success never fires in the untouched scene and fires after the target is dropped into the bin. |
| `qualify.py` | Isaac, a GPU | Step 4 physics: fresh OMY env per case, nominal + seeded perturbed runs, per-physics-step hold trace, limits, flushed per-case reports. |
| `visibility.py` | Isaac, a GPU | Step 4 visibility: depth copy of OverShoulderLeft, handle points projected from the simulated final pose; area-based rule (≥ 100 px², ≥ 0.5 × the entry's distance-normalised best case, class-A angle ≤ 60°). `--yaw-offsets` runs a reference sweep. Cameras cannot be re-created in one Kit app, so run **one task per process**. |
| `visibility_reference.py` | — | Per-entry best-case visible area from `visibility.py` sweeps (input to the gate's `--reference`). |

Examples:

```bash
tools/simple_pnp_v2/usd_python.sh tools/simple_pnp_v2/collision_bottoms.py --retained \
    --report output/simple_pnp_v2_validation/collision_bottoms.json
tools/simple_pnp_v2/usd_python.sh tools/simple_pnp_v2/reach_map.py \
    --output output/simple_pnp_v2_validation/reach_map.json
.venv/bin/python tools/simple_pnp_v2/capacity_estimate.py \
    --reach-map output/simple_pnp_v2_validation/reach_map.json

# Reproduction check on a new machine (reset determinism, then the ground-plane fix):
.venv/bin/python tools/simple_pnp_v2/run_isolated.py cuda:0 tools/simple_pnp_v2/start_probe.py \
    --task SimplePnp230MilkjugA0100Task SimplePnp064ComputerMouse02Task --seeds 0 1 2 0 \
    --output output/simple_pnp_v2_validation/probes/determinism_gpu0
.venv/bin/python tools/simple_pnp_v2/run_isolated.py cuda:0 tools/simple_pnp_v2/start_probe.py \
    --task SimplePnp230MilkjugA0100Task --clearances 0.002 0.00001 --ground-z -0.700 \
    --output output/simple_pnp_v2_validation/probes/ground_gpu0
.venv/bin/python tools/simple_pnp_v2/analyze_start_probe.py output/simple_pnp_v2_validation/probes/determinism_gpu0
```

Isaac processes often exit 139 at shutdown after writing results. `start_probe.py`
writes each task's JSON before teardown, so judge outcomes from those files.

## Results to reproduce (2026-11-02, RTX 3090, Isaac Sim 5.0.0 / Isaac Lab 2.2.0)

- **Reset determinism:**
  - post-reset states identical across seeds;
  - positions identical across separate processes;
  - successive in-process resets drift 0.2–0.5 mm, even with the same seed;
  - the mouse layout amplifies that to 3–4 mm / 7–8°.
- **Upstream ground plane:**
  - the table root rises from z −0.350 to −0.347 in the first step;
  - with `--ground-z -0.700` the table does not move;
  - contact starts (1e-5) move objects 0.01–0.22 mm in total;
  - 2 mm starts drop about 2 mm.
- **Collision bottoms:** none below visual bottoms, maximum +0.83 mm. Only `computer_mouse.usd` has an up-tilting root rotation.
- **Reach and capacity:** reach and capacity numbers as recorded in the plan.

## Pipeline (2026-11-02)

```bash
I=output/simple_pnp_v2_validation/inputs
tools/simple_pnp_v2/usd_python.sh tools/simple_pnp_v2/reach_map.py --output $I/reach_map.json
tools/simple_pnp_v2/usd_python.sh tools/simple_pnp_v2/freeze_inputs.py --reach-map $I/reach_map.json --output $I/inputs.json
tools/simple_pnp_v2/usd_python.sh tools/simple_pnp_v2/handle_records.py --inputs $I/inputs.json --output $I/handle_records.json
tools/simple_pnp_v2/usd_python.sh tools/simple_pnp_v2/generate_scenes.py --inputs $I/inputs.json --handles $I/handle_records.json --layouts 1
.venv/bin/python tools/simple_pnp_v2/generate_tasks.py --record-dir output/simple_pnp_v2_validation/generation
.venv/bin/python tools/simple_pnp_v2/run_pinned.py cuda:0 tools/simple_pnp_v2/qualify.py --tasks <classes> --runs 5 \
    --output output/simple_pnp_v2_validation/physics/pilot
.venv/bin/python tools/simple_pnp_v2/run_pinned.py cuda:1 tools/simple_pnp_v2/visibility.py --tasks <one class> \
    --handles $I/handle_records.json --inputs $I/inputs.json --yaw-offsets -180 -150 -120 -90 -60 -30 0 30 60 90 120 150 \
    --output output/simple_pnp_v2_validation/pilot3/sweep
python3 tools/simple_pnp_v2/visibility_reference.py --sweeps output/simple_pnp_v2_validation/pilot3/sweep \
    --output $I/visibility_reference.json
.venv/bin/python tools/simple_pnp_v2/run_pinned.py cuda:1 tools/simple_pnp_v2/visibility.py --tasks <one class> \
    --handles $I/handle_records.json --inputs $I/inputs.json --reference $I/visibility_reference.json \
    --output output/simple_pnp_v2_validation/pilot3/visibility
```

Bounds: `BBoxCache.ComputeUntransformedBound` inflates rotated prims in composed
scenes (up to ~2x). The shared audit, the freeze, and v2 `ScenePreparation` use
`robolab.core.scenes.placement.prim_local_bounds` (geometry points) instead.
