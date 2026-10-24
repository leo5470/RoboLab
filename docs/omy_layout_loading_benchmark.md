# Continuous layout loading benchmark

Measured on 2026-10-20 with all 50 retained `canned_mushrooms` layouts
(`SimplePnp034CannedMushrooms00Task` through `49Task`).
The prototype reduced total loading time by approximately 97.5% with rendering.
These measurements are from the standalone prototype. Object-level collection
is now implemented in the production CLI and passed the integrated simulator/GPU
check on 2026-10-21. Its separate measurements and validation are reported below;
the prototype timings are not measurements of the integrated collector.

## Prototype results (2026-10-20)

| Measurement | State-only headless | Headless with 640×480 RGB camera |
| --- | ---: | ---: |
| Fresh process to first ready layout, median of 3 launches | 8.103 s | 22.941 s |
| Initial startup of the persistent session | 8.198 s | 23.813 s |
| Switch to the next layout, median of 49 switches | 14.499 ms | 94.226 ms |
| Slowest switch | 16.629 ms | 99.258 ms |
| Total loading across all 50 layouts, measured persistent session | 8.916 s | 28.423 s |
| Total loading for 50 fresh launches, estimated from median | 405.127 s | 1,147.059 s |
| Estimated reduction in total loading | 97.80% | 97.52% |

Fresh-launch samples were layouts 00, 24, and 49. The state-only samples were
8.152, 8.103, and 8.063 seconds; the rendered samples were 27.648, 22.714,
and 22.941 seconds. Each mode ran in this order: fresh layout 00, one persistent
session visiting all 50 layouts, fresh layout 24, fresh layout 49. Disk and
shader caches were not cleared.

The 50-launch figures are extrapolations, not measurements of 50 separate
launches. Persistent totals are the measured initial startup plus all 49
measured transitions. Demonstration duration, previous-episode export, process
shutdown, operator readiness, and XR connection time are excluded from both
loading totals.

## Prototype method and checks

The diagnostic uses the current OMY environment with CUDA physics, 15 Hz
commands, 120 Hz physics, solver position cap 16, compact state recording,
required contact sensors, zero contact history, and render interval 4.
It runs three synthetic open-gripper/home commands per layout. All diagnostic
HDF5 files are marked `training_ready=false`; no leader port or XR session
is opened.

The persistent session creates one simulator and one environment. It reads
authored target/bin XYZ and WXYZ from each USD layer, updates the existing
rigid bodies' default root states, zeros their velocities, and calls the
ordinary evaluation-state reset and environment reset. Scene geometry and task
managers remain instantiated.

Every measured switch includes:

- Creating the layout's output directory, rotating its HDF5 recording, and
  writing its environment configuration and diagnostic task metadata.
- Updating default poses, clearing episode state, resetting, calling render,
  and synchronizing CUDA.
- In rendered mode, retrieving the new operator-camera RGB frame on the CPU.

Assertions passed for all 50 layouts in each completed run:

- Task definitions match apart from layout identifiers and scene filenames.
- Scene layers differ only in the target/bin translation and orientation.
- Initial fresh-scene defaults agree with the authored poses, allowing
  equivalent quaternion signs.
- Both rigid bodies return to the requested positions/orientations with zero
  root velocity. Maximum measured reset error was zero at tensor precision.
- The recorder's initial state and the reopened HDF5 initial state match the
  selected layout.
- Every output file contains the correct task label and exactly one
  three-command diagnostic episode; states remain finite after stepping.
- All 50 rendered frames are nonblank and distinct. Saved images for layouts
  00 and 49 were also inspected to confirm visible bin relocation.

The extra camera is an offscreen rendering check, not the production XR
viewport. It triggers the existing fallback from optimized boundary scene
updates for an unaudited sensor; fresh and persistent runs use the same
fallback. These results validate layout loading, reset, rendering, and basic
recording, not successful manipulation or end-to-end headset latency.

Hardware/software: NVIDIA GeForce RTX 3090, Intel Xeon W-2255, Isaac Lab 2.2.0,
Isaac Sim 5.0.0.0, PyTorch 2.7.0+cu128. Host VR processes remained running;
the benchmark did not connect to or control them.

## Integrated collector check (2026-10-21)

The production `collection.run` completed all 50 CannedMushrooms layouts on the
RTX 3090 with a 640×480 offscreen RGB camera. It created one environment,
performed 50 resets, opened 50 recordings, and created one synthetic leader
device. Each layout recorded three fixed-home/open-gripper commands, with a
synthetic three-command success condition driving automatic advancement.

| Integrated measurement | Result |
| --- | ---: |
| Script start to first ready layout | 24.8353 s |
| Recording-open to ready, median of 49 subsequent layouts | 46.170 ms |
| Slowest recording-open to ready | 54.973 ms |
| Sum across those 49 transitions | 2.2902 s |
| Complete diagnostic elapsed, including 150 commands, exports and QC | 58.685 s |

The transition timer starts at `LayoutRecording.open`, after the session has
created the next part directory. It includes recording/configuration rotation,
pose application, reset, rendering, RGB readback and CUDA synchronization. It
excludes preceding episode export, session manifest advancement, part-directory
creation/persistence, hardware/XR connection and operator waits. These boundaries
differ from the prototype measurements; this integrated run does not establish a
new percentage reduction against fresh launches.

All 50 layouts passed reset-pose/zero-velocity and saved initial-state checks,
with zero maximum reset error at tensor precision. All 50 recordings had the
correct task and configuration, and a separate full validation pass accepted
their structure and native-15 timing while confirming explicit exclusion from
training. Rendered frames were nonblank and distinct across all layouts.
Inspection of the noisy single frames for layouts 00 and 49 confirmed visible
bin relocation.

Real Carbonite provider-event checks passed for F9 repeat suppression, F12 event
delivery, Shift+F9 rejection, and status-widget creation/update. The collection
loop itself used synthetic controls, READY confirmation, leader actions and XR
lifecycle. Physical keyboard delivery, leader operation, grasp success and live
headset behavior remain unverified by this diagnostic. No serial/TCP leader
connection or XR session was opened. Every recording is marked
`training_ready=false` and `diagnostic_success=true`, and training selection and
export reject these diagnostic successes.

Held-target commands remained enabled. The added camera and synthetic success
consumer required fallbacks for boundary scene updates and query sharing,
recorded in each part's performance configuration. The camera checks offscreen
rendering rather than the production XR viewport.

See the [integrated report](../output/omy_layout_collection_integration_20261021_r2/README.md),
[raw measurements](../output/omy_layout_collection_integration_20261021_r2/summary.json),
and [validation of all 50 recordings](../output/omy_layout_collection_integration_20261021_r2/all_layouts_validation.json).
Reproduce with a new output directory:

```bash
.venv/bin/python -u scripts/check_omy_layout_collection.py \
  --object canned_mushrooms --render-camera \
  --output-dir output/layout_collection_integration
```

## Reproduce and inspect the prototype

Run from the repository root with new output directories:

```bash
.venv/bin/python -u scripts/benchmark_omy_layout_loading.py \
  --object canned_mushrooms \
  --output-dir output/layout_loading_state

.venv/bin/python -u scripts/benchmark_omy_layout_loading.py \
  --object canned_mushrooms --render-camera \
  --output-dir output/layout_loading_rendered
```

Each command launches four sequential simulator processes. The persistent
process visits all 50 layouts. Results include per-layout timings, task and
scene identifiers, expected poses, validation errors, HDF5 diagnostics, and
rendered sample images when enabled.

Measured artifacts:

- [State-only summary](../output/omy_layout_loading_20261020_state/summary.json)
- [Rendered summary](../output/omy_layout_loading_20261020_rendered/summary.json)
- [State-only per-layout measurements](../output/omy_layout_loading_20261020_state/reuse_all/result.json)
- [Rendered per-layout measurements](../output/omy_layout_loading_20261020_rendered/reuse_all/result.json)
- [Benchmark source](../scripts/benchmark_omy_layout_loading.py)
