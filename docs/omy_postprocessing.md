# Offline OMY postprocessing: implemented usage

The entry point is [`scripts/postprocess_omy.py`](../scripts/postprocess_omy.py). Indexing, packaging and numerical mapping use only the existing HDF5/NumPy environment. Rendering starts a separate headless Isaac Lab process. Nothing here opens the L100 port, starts XR, changes live collection, or modifies standard RoboLab evaluation.

The design and outstanding training decisions remain in [postprocess.md](postprocess.md). This implementation preserves raw trajectories; it does not repair bad grasps or remove manipulation pauses.

The [`collect_omy_xr.sh`](../scripts/collect_omy_xr.sh) launcher now runs validation,
packaging and policy-camera rendering automatically after quota completion or a
graceful Ctrl+C. It waits for recording and simulator shutdown, then prints the
absolute postprocessed dataset, manifest and report locations. Outputs live in
`SESSION/postprocess/RUN_ID/`; `dataset.json` lists completed HDF5 camera episodes,
and `summary.json` retains processing failures and collection status. Rendering
uses configured camera resolution, realtime `cuda:1` and two renders per frame.
Failed/interrupted valid attempts and quality flags are retained, with
`training_ready: false`. Numerical mapping and recipe-gated export are separate.

Collection uses GPU 0. While processing runs on GPU 1 in the original terminal,
start the next collection in a separate terminal. Override this with the
launcher's `--postprocess-device cuda:N`, or `--device cuda:N` on the session
postprocessor. These indices use the host's `nvidia-smi` ordering. Each renderer
uses bubblewrap to expose only the selected GPU in a private device namespace;
CUDA and Vulkan see that GPU as local index 0. This prevents the extra CUDA and
graphics allocations previously observed on host GPU 0. CPU and storage remain
shared. Isolation requires Linux with `bwrap`, `nvidia-smi` and working
user/mount namespaces; failure stops processing without an unisolated fallback.
The render log and manifest's `gpu_execution` field record the host GPU UUID
and child mapping. Existing verified camera caches remain reusable.
OMY renderer processes for the same user and GPU take turns through a file lock,
including jobs started by later collections. A persistent worker holds the lock
for its entire lifetime and releases it after simulator shutdown.

Use `--no-postprocess` to opt out. Retry a finalized invocation with
`.venv/bin/python -u scripts/postprocess_omy_session.py --handoff SESSION/collection_runs/RUN_ID.json`.
Matching archives/caches are verified and reused. Each resume processes only
new recording parts in its own invocation directory. The
[collection guide](omy_xr_collection.md#automatic-postprocessing-and-dataset-location)
documents interrupts, logging, output layout and exit statuses.

Older object-layout sessions without handoff files can be processed with
`.venv/bin/python -u scripts/postprocess_omy_session.py --session SESSION --device cuda:1`.
The session lock excludes active collection while the input list is snapshotted,
then releases so collection can resume into new parts during processing.
The printed retry command includes `--run-id` to reuse verified outputs and the
frozen recording list in `session_input.json`.

## Selection, persistent workers, and verification videos

The session postprocessor now defaults to **one Isaac application per batch**.
It starts lazily when a selected attempt needs rendering. Matching recorded
configurations reuse an environment; layout/configuration changes rebuild the
environment within that same application. The profile still renders restored
recorded states at the configured resolution with two passes per frame. Eight
unrecorded renders at the exact initial pose refresh camera history before frame
zero. No physics action is replayed and no dataset rows are added.

```bash
.venv/bin/python -B scripts/postprocess_omy_session.py \
  --handoff SESSION/collection_runs/RUN_ID.json --device cuda:1 \
  --skip-failed --preview-videos
```

`--skip-failed` selects explicit successes before packaging or rendering. Raw
failed/interrupted attempts remain saved. Unknown outcomes are reported among
skipped attempts. Selected successful attempts must still pass all structural,
native-15, asset/provenance and numerical-preservation checks. Without this flag,
valid failures are included as before. Empty selections and fully cached retries
start no Isaac application.

The supervisor stores GPU/selection/preview preferences in the finalized handoff;
`processing_options.json` persists the processing selection. The retry command
prints the corresponding flags. To use another selection, choose another output
run with `--run-id NEW_RUN`; the original handoff and its recording list are reused.
The `--session SESSION --run-id RUN` path remains available for an explicitly
closed layout session and snapshots its recording list before releasing the
collection lock.

Each attempt is independently verified and atomically published. The parent
checks a matching worker acknowledgment before examining its output; pending or
failed jobs never enter `dataset.json`. The worker keeps its native stdout in the
current attempt's `render.log` and detaches that log before acknowledgment, so
later jobs cannot change a published log checksum.

Worker diagnostics live under `workers/isaac_*/`: `worker.log`, `ready.json`,
`job_*.json`, and `result_*.json`. `summary.json` reports application starts,
environment builds/reuses, rendered/skipped/failed counts, and skipped outcomes.
It also records validation, packaging, rendering/verification, and video wall
times; worker metrics separate application startup and environment setup.
A worker crash or timeout permits one automatic worker restart; repeated crashes
stop further worker launches and retain diagnostics for an explicit retry.
The startup watchdog is 300 seconds and each active job has a 900-second timeout.
Ctrl+C stops the worker process group, closes its descendants, and retains
completed outputs. Missing isolation support never triggers an unrestricted
fallback. Kit worker/TBB thread limits are four; review encoding uses two threads.

For diagnostic comparison, `--renderer-mode per-episode` retains the original
one-application-per-attempt path. Normal operation uses `--renderer-mode batch`.

`--preview-videos` encodes review clips on CPU after releasing the rendering GPU.
The launcher equivalents are `--postprocess-skip-failed` and
`--postprocess-preview-videos`; both are opt-in. Preview output includes:

```text
verification_videos/
  all_successes.mp4           # combined native-15 reel
  success_*.mp4              # separate video for each successful episode
  success_*_frames.jpg        # first / middle / last from each camera
  final_frames.jpg
  index.json                 # decoded counts, timing and artifact checksums
```

Labels use the recorded outcome and clips end at the final recorded frame.
New previews retain both the combined MP4 and separate episode videos, verified
for frame count and timing. `index.json` links each individual video and records
its frame range and start time within the combined video.
Diagnostic fixtures are explicitly labeled. Existing matching preview outputs
are checksum-verified and reused. Older combined-only previews require a new
output directory to regenerate individual clips from cached images.
Generate videos separately without Isaac:

```bash
.venv/bin/python -B scripts/preview_omy_session.py --dataset SESSION/postprocess/RUN_ID
```

## Fresh-fixture validation and performance

`scripts/check_omy_preparation.py` exercises preparation, refreshed initial-state
capture, native-15 recording, and repeated resets without opening a leader port.
Its synthetic hold trajectories are marked `diagnostic_only`; assigned outcomes
exercise selection and are not human demonstrations. Use `--host-gpu` to isolate
the physical GPU before Isaac starts. The wrapper confirms child exit and prints
a finalized diagnostic handoff:

```bash
.venv/bin/python -B scripts/check_omy_preparation.py --host-gpu 1 \
  --output-dir output/omy_pipeline_checks/NEW_CHECK --episodes 2 --frames 15
.venv/bin/python -B scripts/postprocess_omy_session.py --handoff PRINTED_HANDOFF \
  --skip-failed --preview-videos --device cuda:1
```

A fresh check used two episodes each for wooden-bowl layouts 00/01, canned
mushrooms, milk carton, and a small plate. All ten prepared in 0.533 simulated
seconds and retained settled initial snapshots. Numerical source rows and
attributes survived postprocessing exactly. The batch used one application,
five environment builds, and five reuses, and produced both policy cameras and
verified 15 FPS videos.

On the RTX 3080, ten **15-frame diagnostic episodes** took **336.3 seconds** with
per-episode launches versus **115.3 seconds** with the persistent worker, including
validation and CPU review encoding: about **2.9× faster** for this startup-heavy
fixture. Both modes showed zero attributed GPU 0 compute/graphics memory; batch
GPU 1 memory peaked at **5,791 MiB**. First/middle/last comparisons over 60 camera
frames had at most 1.98/255 mean absolute RGB error between modes; raw numerical
channels remain exact. This benchmark is not a timing estimate for ten full-length
human demonstrations, and live leader/XR collection latency has not been measured.

Evidence: `output/omy_pipeline_checks/20261023_164744/` contains
`preparation_check.json`, `benchmark.json`, GPU sample logs,
`batch_image_comparison.json`, and the per-mode output datasets and videos.

The final geometry/support check and cache-only retry are under
`output/omy_pipeline_checks/final_20261023_172019/`. `final_validation.json`
records ten source files preserved exactly, one first-pass Isaac application,
five environment reuses, zero applications on retry, and zero attributed GPU 0
memory across 528 samples. `settled_start_preview.jpg` shows the initial camera
views after preparation. All ten final preparation checks passed with explicit
table-support checks. Verification uses fresh diagnostic fixtures; historical
collection datasets are outside this implementation's migration scope.

Regression verification covered 629 distinct OMY tests across the broad and
focused runs. The broad run passed 627 tests; the follow-up also exercised the
updated task-logic fixture and a late worker-exit cleanup case. Syntax, shell,
and whitespace checks passed. The documented `check_omy_preparation.py --host-gpu 1`
command was separately run and produced a finalized handoff after child shutdown.

New OMY recordings include `task_dirs` and portable task-file SHA-256 provenance.
The materializer automatically reads the recorded folder's `manifest.json` and
imports only the selected task; a changed task file is refused before Sim
startup. Older recordings remain compatible with the `benchmark` default.
For an older generated-task recording without folder metadata, pass
`--task-dirs single_pnp` to `scripts/materialize_demo_images.py`; explicit folders
must agree with any recorded folder metadata. Task-source verification does not
claim historical verification of all USD/texture dependencies. Native-15 timing
and pre-action image alignment are unchanged.

## The native-15 contract

For an episode with `N` issued actions, processing writes exactly `N` dataset rows. Every recorded interval must advance `1/15` simulated second and consecutive intervals must be continuous. Declared control rate, physics timestep and decimation are cross-checked against the actual float64 timestamps. Nonzero simulation-clock origins are supported. Intact 30/60-Hz recordings can be archived, but the first mapping/render/export profile refuses them; no resampling or FPS relabeling occurs.

| Final row `t` | Authoritative source |
| --- | --- |
| Timestamp | `t / 15`, after checking the recorded simulation-time grid |
| Issued action | `actions[t]`, no action shift or arm-delta conversion |
| Measured seven arm joints | Checked pre-action `obs/proprio_obs/arm_joint_pos[t]`, resolved by joint name |
| Raw measured gripper | Named `finger_joint` in initial state for `t=0`; `states[t-1]` thereafter |
| Policy camera scene | Initial state for `t=0`; full robot/object `states[t-1]` thereafter |
| Episode boundary | One source episode; never concatenated across attempts |

The last row is at `(N-1)/15` seconds; the recorded action interval ends at `N/15`. There is no invented terminal observation. Wall-clock stalls are retained as quality evidence, not stretched into dataset timestamps. Slow collection can still harm the human demonstration; offline processing cannot undo that.

The authoritative numerical mapping has float64 timestamps. The provisional LeRobot adapter checks actual written Parquet rows in bounded batches, including the expected float32 timestamp representation where used by the writer. It checks row counts, ordering, and numerical round trips before publication. The adapter still needs acceptance with an actual pinned LeRobot version and both PPS loaders.

## Commands

Run from `/tmp2/leocheng/RoboLab`. Use a closed recording, not a file still being written. Choose explicit destinations on the desired filesystem; existing conflicting outputs are refused. Failed jobs retain adjacent `.partial-*` directories and diagnostics.

Inspect the latest HDD attempt without starting Isaac Sim:

```bash
.venv/bin/python -B scripts/postprocess_omy.py index --input output/omy_teleop/20261003_194913_828617/omy_demos.hdf5 --require-native15
```

Add `--output NEW_REPORT.json` to publish a new machine-readable report. `--require-training-eligible` additionally fails on unresolved quality warnings or non-success outcomes.

Package every structurally valid episode from that session, retaining failures and warnings:

```bash
.venv/bin/python -B scripts/postprocess_omy.py package --input output/omy_teleop/20261003_194913_828617/omy_demos.hdf5 --output output/omy_archive
```

This produces `{TaskName}_{UTCtimestamp}_aNNNN/` directories. Re-running verifies and reuses a matching archive; altered source sidecars, conflicting identities or damaged artifacts are errors. `--episode N` selects a zero-based source demo index and is repeatable.

For the latest attempt, render the two policy cameras at the provisional PPS-example resolution. Run this after teleoperation, using the known-working Isaac Lab environment:

```bash
.venv/bin/python -B scripts/postprocess_omy.py render --input output/omy_archive/BananaInBowlTask_20261003T114940000000Z_a0001/trajectory.hdf5 --output output/omy_rendered/BananaInBowlTask_20261003T114940000000Z_a0001 --device cuda:0 --camera-width 320 --camera-height 180 --renders-per-frame 2
```

The policy cameras are `over_shoulder_left_camera` and `wrist_cam`, not the Quest viewpoint. Rendering loads recorded states; it does not replay actions through physics. Omit the explicit dimensions to use the full camera resolution, or choose an explicit scale. The PPS adapter refuses incompatible dimensions rather than resizing silently.

Write a diagnostic numerical mapping without LeRobot:

```bash
.venv/bin/python -B scripts/postprocess_omy.py rows --input output/omy_archive/BananaInBowlTask_20261003T114940000000Z_a0001/trajectory.hdf5 --output output/omy_rows/BananaInBowlTask_20261003T114940000000Z_a0001 --recipe configs/omy_postprocess/pps_recipe.example.json --diagnostic
```

The example recipe deliberately selects raw measured finger angle in radians and the candidate gripper command conversion `(1 - raw_command)/2` (`0=open`, `1=closed`). These are explicit provisional choices, not approved checkpoint units. Raw commands, measured finger angle and normalized measured closure are retained separately. No clipping or inferred closure from commands is allowed. `--diagnostic` admits failed/flagged episodes for inspection, never invalid timing/structure, and never sets training readiness.

## Saved formats

An immutable raw attempt contains:

```text
{TaskName}_{UTCtimestamp}_aNNNN/
  trajectory.hdf5            # one original episode; values/attributes preserved
  metadata.json              # UUID, original source hash/group, outcomes, checksums
  env_cfg.json               # captured source configuration
  calibration.json           # embedded recording snapshot, not today's live file
  timing.npz                 # original timing arrays, including wall timing
  performance.json           # original attempt/session scope explicitly recorded
  quality.json               # validation and selection warnings
  collection_events.json     # when events exist
  xr_status.json             # when the source sidecar exists
```

Legacy recordings without attempt-start events use a clearly marked UTC session-time fallback. Stable UUIDs derive from the original source hash and episode group and survive relocation/repackaging. Interrupted/reset evidence and performance scope survive archive reimports. IDs do not claim unrecorded movement-start precision.

A render cache contains `images.hdf5`, the rendering `env_cfg.json`, `quality.json`, `render.log` and a checksummed `manifest.json`. Numerical channels remain identical to the raw trajectory. The HDF5 adds `omy_materialized_alignment` with source step, pre-action state-row index (`-1` means initial state), exact timestamps and original simulated intervals. Source/profile hashes and resolved local dependencies are checked before cache reuse. Unhashed/remote dependencies and unknown historical collection-time asset identity remain explicit provenance limitations; cache reuse is not fresh remote-asset verification or training approval.

A numerical mapping contains `rows.hdf5`, `recipe.json`, `validation.json` and `manifest.json`. Rows include canonical timestamps, original simulation origin/timestamps, seven measured arm joints, chosen measured gripper representation, mapped eight-channel actions, original commands, raw finger angle and normalized measured closure. It is a diagnostic intermediate, not a LeRobot training dataset.

## Recipe-gated PPS conversion

[`configs/omy_postprocess/pps_recipe.example.json`](../configs/omy_postprocess/pps_recipe.example.json) intentionally leaves the teacher checkpoint, normalization artifact, both proxy configurations/horizons and exact LeRobot version unresolved. The `export` command refuses it before creating a dataset. Do not modify or synchronize the working simulation `.venv` to satisfy export dependencies.

After selecting and testing a pinned processing environment and completing the recipe, invoke:

```bash
/PATH/TO/PINNED/EXPORT/python scripts/postprocess_omy.py export --recipe APPROVED_RECIPE.json --selection SELECTION.json --output NEW_DATASET_DIRECTORY --repo-id local/omy-pilot
```

The selection manifest must explicitly specify roles, a coordinated episode-level split and persistent identity:

```json
{
  "schema_version": 1,
  "episodes": [{
    "source": "rendered/ATTEMPT_ID/images.hdf5",
    "group": "data/demo_0",
    "attempt_id": "ATTEMPT_ID",
    "attempt_uuid": "UUID_FROM_ARCHIVE",
    "split": "train",
    "roles": ["reference", "task"],
    "task_instruction": "Pick up the banana and place it in the bowl"
  }]
}
```

Relative source paths resolve against the selection file's directory. Copied episodes, duplicate UUIDs/IDs or numerical fingerprints are refused within an export selection. Maintain a coordinated master selection across separate export jobs too; the writer does not maintain a global split registry. Final conversion requires packaged attempt identity, successful outcomes, verified pre-action cameras and resolved quality review. A warning review records `reviewer`, `reason`, `accepted_warning_codes`, the exact input HDF5 `source_sha256`, `source_group` and numerical `fingerprint`; it cannot be reused silently on another raw or rendered file. Both role manifests retain the same episode/split provenance. The output is local only, with no upload.

The implemented adapter is provisional `common_v2` API with image storage, not a tested v3/video writer. All conversion outputs remain `training_ready: false`. Recipe completeness and passing writer-contract tests do not establish compatibility with the teacher's units, normalization, image transforms, action horizons or both proxy loaders.

## Verification and remaining acceptance

### Diagnostic settling continuation

[`scripts/preview_omy_settling.py`](../scripts/preview_omy_settling.py) creates a separate 1–2-second physics continuation from a recorded terminal post-state. It holds the last seven arm targets, explicitly opens the gripper, and bypasses success freezing, reset and recorder callbacks. It never appends training rows, opens L100, or changes the source HDF5/camera cache.

Example for the verified attempt:

```bash
.venv/bin/python -u scripts/preview_omy_settling.py --input output/omy_postprocess/20261003_verified/raw/BananaInBowlTask_20261003T114940000000Z_a0001/trajectory.hdf5 --recorded-images output/omy_postprocess/20261003_verified/rendered_final/BananaInBowlTask_20261003T114940000000Z_a0001/images.hdf5 --output output/omy_postprocess/20261003_verified/settling_2s --tail-seconds 2 --tail-fps 60 --device cuda:0
```

Use a new destination for another run. Outputs are `settling_only.mp4`, `recorded_plus_settling.mp4`, `diagnostic_states.hdf5` and `report.json`. The video overlays distinguish recorded frames from diagnostic resimulation. Original 15-Hz frames are repeated only for display at 60 FPS, not resampled into a training dataset. The tail contains the terminal snapshot and 120 subsequent frames across exactly 240 checked 120-Hz physics ticks; its last image is held for one video frame interval.

The diagnostic clock uses the checked actual physics tick index times the recorded float64 timestep, matching collection's clock convention. Isaac's separately accumulated callback clock is retained too; float32 callback timestep rounding must not be mistaken for a changed physics schedule. This does not loosen native-15 training validation.

The tail is a **new diagnostic rollout**, not recovered original footage: saved poses/velocities do not preserve PhysX contact/solver history. It can reveal settling/contact problems but cannot certify that the original stopped episode would evolve identically. Its HDF5 deliberately has no training episode tree and remains `training_ready: false`.

The latest HDD recording was tested through native-15 validation, packaging, state-based two-camera rendering and numerical mapping. It contains 116 rows: last timestamp `7.666666666666667`, duration `7.733333333333333` simulated seconds. Issued arm commands and pre-action measurements are unchanged, raw/rendered numerical rows agree, camera maps retain every source row, and the original raw checksum is unchanged. First/middle/last camera frames were inspected. Invalid leader samples 0 and 4 and the startup wall gap remain flagged; no rows were dropped or automatically approved.

Evidence is under `output/omy_postprocess/20261003_verified/`: `end_to_end_check.json`, `latest_index.json`, `quality_gate_check.json`, the raw archive, `rendered_final/` camera cache, `rows_final/` numerical mapping, and six extracted first/middle/last PNGs in `preview/`. All 36 original raw datasets and nine existing attributes were checked exactly. The cache-reuse check passed, and actual unreviewed training-row selection was refused without creating an output directory. Earlier implementation smoke artifacts are under `output/omy_postprocess/20261003_implementation/`. These are test artifacts, not training-ready datasets.

Final pure postprocessing suite: **290 passed**. The broader OMY/XR regression run also passed (**457 tests**, including the earlier postprocessing revision); the existing deferred-camera suite passed **14 tests** inside a headless Isaac Lab startup. Those totals overlap and are not added together. Syntax/whitespace checks passed for all 16 changed/new Python files. The working simulation dependency environment was not changed.

Run the pure postprocessing tests without starting Isaac Sim:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider --confcutdir=tests/unit tests/unit/test_omy_postprocess_*.py -q
```

Before bulk processing, still required: approved camera/asset profile and gripper conventions; exact PPS/teacher/normalization/LeRobot recipe; actual exported-batch acceptance by both proxy loaders; coordinated dataset selection; and three representative successful plus one failed/interrupted attempt. The current smoke test and synthetic failure cases are not substitutes for that acceptance set or evidence of sufficient training data.
