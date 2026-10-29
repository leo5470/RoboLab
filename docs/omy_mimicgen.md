# MimicGen for OMY-L100 demonstrations

`examples/generate_omy_mimic.py` expands a few L100 teleoperation demonstrations of one pick-and-place task into many, using Isaac Lab Mimic. Isaac Lab bundles it, but importing it needs `ipython` and `ipywidgets`, which `requirements-omy.txt` installs. It does not transfer demonstrations between objects: collect source demonstrations for each task.

```bash
python examples/generate_omy_mimic.py \
    --input output/omy_teleop/<run_1> output/omy_teleop/<run_2> \
    --output-dir output/omy_mimicgen/banana_in_bowl --num-demos 40
```

Each `--input` is a collection run directory containing `omy_demos.hdf5` and `env_cfg.json`. Only successful episodes are used; with a single run, `--episodes 1 2 4` restricts the sources to `demo_1`, `demo_2`, and `demo_4`, and refuses any that are missing or unsuccessful. By default, the first input supplies the target environment. `--target-input` chooses that environment independently of the source pool; `--target-episode` fixes the recorded initial arrangement used as its reset anchor.

## Multiple source demonstrations

The generator supports a pool of compatible demonstrations from different layouts
of the same object/container task. It checks robot configuration, joint ordering, action metadata,
physics/control settings, recording profiles, manipulated assets, and success
semantics before loading the pool. Source scene positions and task/layout names
may differ; the target recording determines the actual scene and its other objects.
Changing the pool does not change the target reset anchor when it is explicit.

For exact source selection, provide `--source-selection POOL.json`:

```json
{"sources": [
  {"file": "SESSION/TASK_A/part_000/omy_demos.hdf5", "episode": "demo_2", "sha256": "SOURCE_FILE_SHA256"},
  {"file": "SESSION/TASK_B/part_000/omy_demos.hdf5", "episode": "demo_1", "sha256": "SOURCE_FILE_SHA256"}
]}
```

Paths resolve relative to the pool manifest. SHA-256 values are optional on input
and always recorded in output provenance. Missing/unsuccessful episodes, duplicate
references, changed source checksums and incompatible contracts are refused.
Without this manifest, all successful episodes in `--input` recordings are pooled;
the usual single-input `--episodes` restriction remains available.

`--selection-strategy` accepts `random` (the compatibility default),
`nearest_neighbor_object`, or `nearest_neighbor_robot_distance`. `--nn-k` defaults
to 3 and is clamped to pool size; use 1 for strictly nearest selection. The chosen
source remains the same across grasp and placement. This improves the choice of
source motion; it does not change the one-step differential IK controller.

`--selection-seed` controls trajectory/source randomness independently of scene
reset seeds. Each attempt records source index, exact file/episode/hash/task,
trial ID, prepared object poses and IK residuals in `mimicgen_summary.json`.
Exported episode attributes also retain `mimicgen_source`. Dataset curation and
the website's source links use the chosen human source, including when it differs
from the recording that defines the target scene.

Batch generation supports `source_pool: true` in `run_config.json`, with
`source_selection_strategy` and `source_selection_nn_k`. It pools qualified source
manifest entries from the current family; qualification jobs remain individual.
Pooling is opt-in while the comparative validation is in progress.

## Paired source-strategy experiments

```bash
OMNI_KIT_ACCEPT_EULA=Y .venv/bin/python -B -u scripts/compare_omy_mimic_sources.py \
  --output output/omy_mimic_tests/NEW_EXPERIMENT --host-gpu 1
```

The default experiment discovers `20261028` sessions, freezes their eligible human
sources, and reproduces each source twice without randomization. Sources with at
least one quality-passing reproduction enter that session's pool. It then runs
ten attempts for each of four arms: scheduled single-source baseline, random
pool, nearest-object pool, and nearest-robot pool. Each session initially uses one
fixed target layout with ten sampled arrangements. The baseline loads the same
pool but forces a balanced, predetermined source index per trial, keeping startup
and target-scene setup comparable across arms.

`--session`, `--limit-sources`, `--qualify-attempts`, and `--pilot-attempts` bound a
smaller smoke test. `--confirm-attempts 50` adds a separate seed set for baseline
and the strongest pooled pilot arm. It can be added to the same completed pilot
output without rerunning verified jobs; other experiment options must stay fixed,
and a started confirmation budget cannot change. Confirmation is optional and must precede
claims of a reliable performance improvement. No pooled default is promoted by
the experiment automatically.

`--trial-manifest` on the underlying generator accepts a `trials` list with `id`
and `seed`; its length must equal `--max-attempts`. Baseline trials additionally
specify `source_index`, activated with `--follow-source-schedule`.
Both success target and attempt budget equal the requested trial count, so setup
rejections remain in the denominator. Actual prepared poses must match across
arms within 0.1 mm translation and 0.05 degrees rotation, using sign-invariant
quaternion angular distance; preparation statuses must also match. Otherwise the
comparison is refused. Reports retain
success/acceptance rates, IK residuals, rejection reasons, source counts and wall
times. Job wall time includes startup and GPU queue waits; use a dedicated,
uncontended run for throughput comparisons. Failed comparison rollouts are retained for inspection and excluded by the
normal acceptance gates. Verified job checksums are checked on retry, and an
experiment lock prevents concurrent supervisors. Failed job directories
must be preserved for explicit recovery before rerunning.

Experiment recordings live in `EXPERIMENT/generated/SESSION/TASK/JOB/PART/`.
Point `build_omy_dataset.py --generated-root EXPERIMENT/generated` at this pool to
exercise the usual up-to-10-human, 50-total selector. Insufficient candidates
still fail without relaxing quality checks. Reduced-count pipeline smoke exports
must use a separate diagnostic output root and are not 50-demo datasets.

Without `--output-dir`, future generation preserves the target recording's session,
task and part folder names:
`output/omy_mimicgen/SESSION/TASK/TIMESTAMP/PART/omy_demos.hdf5`.
An explicit `--output-dir` still sets the exact recording directory. The batch
supervisor writes new recordings to
`RUN/recordings/SESSION/TASK/JOB_ID/PART/omy_demos.hdf5`, with logs and job metadata
under `RUN/jobs/JOB_ID/`. Its selected copies preserve the same session/task
hierarchy under `RUN/selected/`. Unique job folders prevent repeated generations
from colliding. Existing batches using `jobs/JOB_ID/data/` remain readable and
are not moved.

Recorded task folders and task-file hashes are honored during registration. Resets are centered on the selected target episode's actual initial object poses (the first eligible target success when no target episode is specified). Before each attempt, the scene preparation gate checks geometry, support and settling, and captures the prepared initial state. Its unrecorded steps do not add rows to the demonstration.

`--host-gpu N` uses the existing isolated GPU launcher: the requested host GPU is the only GPU exposed to the child, where it is addressed as `cuda:0`. For dedicated batch workers, `--worker-exit` exits after explicitly closing the recording and environment, avoiding a reproduced Isaac Sim 5.0 crash during subsequent plugin teardown. A parent must wait for the worker and independently validate the artifacts. `generation_finalized.json` records completed trial-loop status and recording closure; `mimic/eef_pose` holds the achieved pre-action end-effector poses for diversity comparisons.

The approved 2026 collection batch uses `scripts/run_omy_mimic_collection.py --run RUN_DIRECTORY`, with frozen `source_manifest.json` and `run_config.json`. It qualifies each source, generates 300–500 quality-passing candidates per object family within a 400-attempt per-layout cap, and selects 100 per family. Completed jobs are reused on restart; unfinished job directories are preserved and refused for explicit recovery. The selector combines arrangement, achieved phase-aligned paths, joint configurations, and grasp/release geometry, with source representation constraints. Selected recordings remain in shards with matching environment configurations; `selection.json` is the combined episode index. Coverage reports compare the selected set with random sets having the same source counts. Original trajectories are copied exactly; only feature comparisons are resampled.

### Separating a completed legacy batch

`scripts/split_omy_mimic_batch.py --run RUN` prints a migration plan without
changing recordings. With `--apply`, it separates a finalized batch into original
collection-session folders beside `RUN`. Stop the website and video worker
before applying; the migration also locks the supervisor and video service.
For each session, raw recordings use `SESSION/TASK/JOB/PART/`, selected copies
use `SESSION/selected/TASK/JOB/PART/`, and job metadata stays under `SESSION/jobs/`.
Each session has its own source manifest, selection, configuration and summary.
Future supervisor jobs started with `--run .../SESSION` use this same hierarchy.

The migration verifies payload checksums before publication, preserves persistent
`catalog_id` values in selections, and updates video-ready signatures and render
manifest source references. Video and camera-image files remain in the shared
demo-library cache. The website discovers each session's selection automatically;
its frontend needs no layout changes. Qualification/rejected attempts and existing
100-demo selections remain preserved; the final 50-demo selector still uses the
entire admitted candidate pool.

The original batch is retained under `.legacy/RUN_NAME/`; the old location becomes
a compatibility directory containing `migration.json` and historical recording/
preview links. It has no active selection or source manifest, preventing duplicate
catalog entries. Byte-identical immutable files use hard links on the same
filesystem. Migration records include payload hashes and original cache metadata
for recovery. Existing quality-review holds are retained. Failed publication
rolls back the new folders and cache metadata; repeated application to a completed
migration returns its recorded result.

The `20261028_diverse_010701` migration is complete. Its six session folders
preserve all 491 jobs, 2,660 accepted generated candidates, and the existing 600
selected demos. All 600 HD videos remain ready under the same catalog IDs.
`output/omy_mimicgen/20261028_diverse_010701/migration_verification.json` records
the post-migration catalog checks. The six final-selector dry runs each reached
50 demos (wooden bowl: 6 human + 44 generated; the other five: 10 + 40).
Final training-file export remains a separate invocation of the builder.

## What is generated

Generation runs in the teleoperation environment. Isaac Lab Mimic plans end-effector waypoints; the environment converts each waypoint to an absolute joint command with one damped-least-squares IK step, so the output `omy_demos.hdf5` has the same actions, states, and observations as a teleoperation recording and goes through the same postprocessing (see [postprocess.md](postprocess.md)). An episode ends when the task's success term fires, as in teleoperation. The run-level `robolab_omy_teleop` and `robolab_omy_collection` attributes are copied from the target recording, so `scripts/materialize_demo_images.py` reconstructs the target scene independently of which human source was chosen. The `robolab_omy_mimicgen` dataset attribute records the sources, seed, and randomization; `mimicgen_summary.json` lists every attempt.

The L100 has six axes and never commands `panda_joint3`. IK therefore solves for the other six joints and holds `panda_joint3` at its home value; source episodes that command it are refused.

End-effector poses of the source episodes are not recorded. They are computed by writing the recorded joints (measured, then commanded) into the simulator and reading the gripper base pose, without stepping physics.

## Segments

A demonstration is split in two: up to the grasp, expressed relative to the object, and after it, expressed relative to the container. The grasp is the first step at which the gripper command is closed and the object is 2 cm above its starting height. Object and container default to the `object` and `container` parameters of the task's `success` term; override them with `--object` and `--container`.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--pose-range` | `0.1` | Uniform x/y reset offset in meters for the object and container |
| `--yaw-range` | `0` | Uniform reset yaw offset in radians |
| `--object-pose-range` | `--pose-range` | x/y reset offset in meters for the grasped object alone |
| `--object-yaw-range` | `--yaw-range` | Reset yaw offset in radians for the grasped object alone |
| `--no-randomize` | off | Keep the recorded arrangement, reproducing the source demonstrations |
| `--interpolation-s` | `1.0` | Simulated seconds for approaching each transformed segment |
| `--settle-s` | `2.0` | Simulated seconds to hold the final target while waiting for success |
| `--action-noise` | `0` | Standard deviation in meters of noise on target positions |
| `--max-attempts` | `200` | Attempt budget; the exit status is 1 if it is spent first |
| `--keep-failed` | off | Also export failed attempts (`success` attribute false) |

Randomization replaces the task's reset events for the generation run: the object and container are offset from their scene poses and everything else is reset to its default. The object options give the grasped object a wider range than the container. Sampling does not avoid the other objects, so choose ranges that keep the object clear of them and on the table.

Qualify a new source with `--no-randomize`, choosing that source recording/episode as the target anchor and forcing that source. This reproduces its arrangement and tests segmentation/controller behavior before adding randomization. With a multi-source pool, `--no-randomize` keeps the target arrangement fixed; other sources are still transformed into that target scene.

## Filtering by quality

### Final mixed dataset: 50 demos from a designated session

For automatic discovery and camera-equipped **training inputs**, use:

```bash
# Discover every pending session, select 50 each, render missing cameras, publish.
.venv/bin/python -B scripts/build_omy_dataset.py

# Restrict processing to one original collection session.
.venv/bin/python -B scripts/build_omy_dataset.py \
  --session 20261023_222802_milk_carton --device cuda:1

# Inspect selection without starting Isaac or publishing a dataset.
.venv/bin/python -B scripts/build_omy_dataset.py \
  --session 20261023_222802_milk_carton --dry-run
```

No `--mimic-run` or input file list is needed. The builder joins matching
session/task folders under `output/omy_teleop` and `output/omy_mimicgen`, and
automatically resolves older batch layouts through their source manifests.
Only finalized standalone generations and admitted batch candidates enter the
pool. Active collection sessions are skipped with a nonzero exit status.
`--session` can be repeated; omitting it processes all discovered sessions.
`--collected-root`, `--generated-root`, `--output-root`, and `--cache-root`
override the default directories. `--count` defaults to 50 per session;
By default, up to 10 eligible human demonstrations are retained and generated demos
fill the remaining slots. `--collected-count` overrides this with an exact
collected quota. The six current
sessions each correspond to one family.

The builder saves a selection before rendering, resumes it after interruptions,
reuses verified matching render caches, and renders only missing selected
episodes with the existing isolated persistent Isaac worker. It verifies and
skips completed final datasets on later invocations. Newly arrived recordings
do not silently replace an already completed 50-demo dataset; use new output
and cache roots to build a new version. Insufficient candidates or processing
failures never publish a partial final dataset; other pending sessions continue.
Cache discovery includes collection postprocessing, generated recordings, the
standard sibling `demo_library/renders` directory, and the builder's own caches.

Final outputs contain only:

```text
output/omy_training/SESSION/
  training.hdf5
  dataset.json
```

To complete all available dated sessions except wooden bowl, including generating
missing candidates for the four qualified October 28 source pools, run:

```bash
OMNI_KIT_ACCEPT_EULA=Y .venv/bin/python -B -u scripts/complete_omy_pools.py --render
```

This controller first publishes exactly 50 unique, quality-passing raw trajectories
per session under `output/omy_final_pool50/SESSION/`. It retains up to 10 humans
and fills the remaining slots from all accepted generated candidates. It imports
the completed multisource pilot recordings into matching session folders and,
where needed, generates additional candidates using the pilot's winning source
selection strategy, qualified sources, and other available target layouts.
Quality thresholds remain unchanged. Additional generation is bounded by
`--max-additional-attempts` (200 per session by default); a shortfall is reported
without publishing a partial pool.

The default host GPU is 1 (`--host-gpu` overrides it). Progress and errors are
recorded in `output/omy_pool_completion/summary.json`. The same frozen selection
feeds the camera training builder, so the raw pool and `training.hdf5` contain
identical trajectories. Omit `--render` to stop after the raw pools are complete.
Retries verify and reuse completed pools, generation jobs, camera caches, and
training files. The existing website's 100-demo selections are independent of
these 50-demo training pools.

The website's **Dataset** tab at `/dataset` browses these frozen 50-demo pools
separately from **Demonstrations**. It discovers published session manifests in
`output/omy_final_pool50`, shows human/generated and layout filters, and tracks
camera availability and completed training exports. Selection manifests and
completed training HDF5 files can be downloaded per session. Only an export
whose 50 trajectory fingerprints match the pool is offered for download.

Dataset playback reuses existing matching demonstration videos. For selected
episodes with camera tensors but no video, opening the viewer prepares a native
resolution, 15-FPS shoulder/wrist video on the CPU; frames can be inspected while
it encodes. These previews stay in `output/demo_library/dataset_previews` and do
not enqueue simulation renders. Episodes still waiting for camera processing
remain visible, and their previews become available automatically as processing
finishes. The Dataset page refreshes every 30 seconds.

To replace the five older families with newly generated multisource datasets:

```bash
OMNI_KIT_ACCEPT_EULA=Y .venv/bin/python -B -u scripts/refresh_omy_multisource.py --gpus 0 1
```

This resumable controller qualifies eligible human recordings, loads the qualified
pool for every generation job, and alternates object-distance and robot-distance
nearest-neighbor selection across target layouts. It obtains 60 accepted generated
candidates per family by default, then selects 40 alongside the same 10 retained
humans. Every generated selection must trace to a qualified human, and at least
two different humans must actually be represented. Existing generated recordings
never become generation seeds. Use repeated `--session` options to designate
sessions, or `--generate-only` to omit camera preparation and activation.

Progress, job logs, and immutable generations live under
`output/omy_multisource_refresh_20261029/`. After camera tensors and fingerprints
are verified, each replacement is moved into the standard dataset roots for the
website. Previous pools, training files, and caches are retained under that run's
`previous/SESSION/` folder. Pending camera exports for the other families resume
after the five replacement pools have been generated. The previous export
controller must be stopped before this controller starts, to avoid competing
writers to the standard training/cache folders.

Each `data/demo_N` contains `actions`, `timestamp`, `obs/arm_joint_pos`,
`obs/gripper_pos`, and the two RGB cameras under `obs/images/`. Task identity and
instruction are episode attributes. Actions retain the original seven absolute
joint targets in radians and gripper command (`+1=open`, `-1=closed`). Measured
gripper state is the pre-action `finger_joint` angle in radians, reconstructed
from the initial state and preceding recorded states. Images and measurements
align to the issued action; timestamps are float64 at native 15 Hz. Full
simulation states, raw archives, rendering logs, review videos, and duplicate
camera caches are excluded from the final dataset. Processing state stays under
`output/omy_training_cache/SESSION/`.

This HDF5 format supplies training tensors in documented native robot units;
it is not a LeRobot/PPS adapter or a checkpoint-specific normalization recipe.
Source warnings and contact-review holds remain in the manifest, with
`training_ready: false` until those reviews and downstream loader conventions
are resolved. The builder does not automatically approve flagged demonstrations.

For an explicit batch and a **raw trajectory-only** export, the lower-level
curator remains available:

Use the original collection **session folder name** to select collected demos
and their MimicGen descendants together. The family is inferred from the batch's
`source_manifest.json`; other sessions of the same family are excluded.

```bash
.venv/bin/python -B scripts/curate_omy_session.py \
  --mimic-run output/omy_mimicgen/20261028_diverse_010701 \
  --session 20261023_222802_milk_carton \
  --output output/omy_final/20261023_222802_milk_carton
```

The default is exactly 50 demos. `--count` changes that number. `--dry-run`
assesses and selects without writing; `--output` is optional in that mode.
`--collected-count 10` requests exactly 10 collected plus 40 generated demos.
Without an origin quota, the selector retains up to 10 quality-passing, unique human
demos and fills the remaining slots with generated trajectories, balancing
layouts and measuring diversity relative to the retained humans and previously
chosen generated demos. If more than 10 humans qualify, it selects 10 representative
humans. If fewer qualify, it retains all of them and uses more generated demos;
for requested counts below 10, the human quota is capped at that count. The default
cap is not relaxed when generated candidates are insufficient. Shared comparison features use initial object/container
poses, object motion, measured joints, and gripper commands; no simulator or
end-effector reconstruction is needed. Only comparison features are resampled.
The automatic builder versions this selection policy in its saved options;
older cached selections or completed outputs require separate cache/output
roots to rebuild under the new policy.

The script discovers all collection parts in the designated source session,
including layouts not used as generation seeds, and quality-accepted generated
job candidates (not just the previous 100-demo selection). `--collected-root`
overrides the root containing the original session folders after relocation.
The same `--max-*` motion/displacement thresholds listed below apply to both
origins. Structural errors, non-native timing, unsuccessful or explicitly
diagnostic episodes, and duplicate numerical trajectories are excluded.
Generated episodes must also have recorded IK and contact checks. Collection
recordings without generation checks record those checks as unavailable.
Insufficient candidates or infeasible origin quotas fail without publishing a
partial dataset; no check is relaxed to reach 50.

The new output directory contains `dataset.json` and copied HDF5 shards under
`collected/SESSION/TASK/...` and `generated/SESSION/TASK/...`, with matching
`env_cfg.json` files and generated summaries. Manifest file paths are relative
to the dataset directory. Episodes retain their original names, numerical rows,
and attributes. Each manifest entry includes source identity, checksums,
trajectory fingerprint, origin, validation warnings, and available generation
parent metadata. The script verifies copied trajectories before publishing and
refuses to overwrite an existing output directory.

This is a curated raw HDF5 dataset; it does not render cameras or convert to
LeRobot. Validation warnings and existing contact-review holds remain explicit,
and the manifest retains `training_ready: false` pending review and downstream
camera/export validation. In particular, the current batch's wooden-bowl
contact-review hold is preserved. See [omy_postprocessing.md](omy_postprocessing.md)
for camera processing and recipe-gated conversion.

### Per-recording quality report

A generated episode can succeed and still be poor training data. `scripts/filter_omy_demos.py` reports quality metrics for every episode of an `omy_demos.hdf5` and writes the accepted ones to a new recording. It does not start Isaac Sim, and it also reads teleoperated and rendered recordings.

```bash
python scripts/filter_omy_demos.py output/omy_mimicgen/<run>/omy_demos.hdf5 \
    --report quality.json --output output/omy_mimicgen/<run>/omy_demos_filtered.hdf5
```

An episode is rejected when any metric exceeds its limit, when it is unsuccessful, or when the task's event tracker recorded an unintended contact. Accepted episodes keep their names, so they still match `mimicgen_summary.json` and a rendered copy; the `robolab_omy_quality` attribute of the output records the limits and the rejection counts. Without `--output` or `--report`, nothing is written.

| Option | Default | Largest accepted |
| --- | --- | --- |
| `--max-command-velocity` | `5` rad/s | Change of the commanded joint targets per second |
| `--max-command-acceleration` | `100` rad/s² | Second difference of the commanded joint targets |
| `--max-command-jerk` | `2500` rad/s³ | Third difference of the commanded joint targets |
| `--max-joint-velocity` | `2.61` rad/s | Recorded arm joint velocity |
| `--max-joint-acceleration` | `20` rad/s² | Second difference of the measured arm joints |
| `--max-joint-jerk` | `500` rad/s³ | Third difference of the measured arm joints |
| `--max-limit-excess` | `0` rad | Distance of a commanded joint target beyond its position limit |
| `--max-displacement` | `0.03` m | Movement of any object other than the grasped one |
| `--max-ik-position-error` | `0.03` m | Distance between a target pose and the pose of its joint command |
| `--max-ik-orientation-error` | `0.1` rad | Rotation between a target pose and the pose of its joint command |

Pass `inf` to disable a check. Accelerations and jerks are finite differences at the control rate, so the defaults suit 15 Hz recordings; the table printed after each run shows the median, 90th percentile, and maximum of every metric for choosing others. The command metrics judge the action labels, which the arm's controller smooths: commands that alternate between two targets on every step leave the measured velocity ordinary.

Joint position limits come from the recording's `robolab_omy_mimicgen` attribute, or the Franka limits when it has none. The displacement check needs the grasped object's name: generated recordings carry it, and `--object` supplies it otherwise.

The IK errors and the tracked events are measured during generation and stored for each attempt in `mimicgen_summary.json`, which the filter reads from beside the recording or from `--summary`. After an attempt ends, every joint command is passed through forward kinematics and compared with the end-effector target it was solved for. The rejecting events are gripper contact with the table or another object, a bumped, moved, tipped, or lost object, and a wrong or second object in the gripper; `--allow-event` exempts one. Recordings without these records skip both checks.
