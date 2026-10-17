# OMY demonstration postprocessing for PPS

This document is the implementation plan for turning Quest + OMY-L100 teleoperation recordings into training data compatible with TritiumR's [Proxy Policy Steering (PPS)](https://github.com/TritiumR/pps) DROID joint-position training path.

Revised 2026-10-03 after reviewing the current code and the HDD/SSD benchmarks. **Status: first offline implementation approved and implemented; final training acceptance pending.** The indexer, validator, immutable archive packager, strict OMY camera path, numerical mapper, and provisional recipe-gated PPS writer are available. Live collection and standard evaluation are unchanged. No processing dependencies were installed or synchronized into the working environment.

See [the implemented CLI guide](omy_postprocessing.md) for commands, saved formats, mapping guarantees, tests, and the remaining training gates.

## Recommendation

Keep state-only HDF5 recordings as the permanent source of truth. After collection, validate and select attempts, render policy cameras from recorded states, then export a PPS-compatible LeRobot dataset. This keeps rendering and encoding out of the live teleoperation loop and allows the training export to be rebuilt later.

The confirmed training consumers are both the PPS reference proxy and task proxy. Use a shared raw archive and preprocessing contract, with explicit episode-selection manifests for each training role.

Final data quality is the priority. The first implementation should preserve the actual recorded trajectory and validate its schema, timing, cameras, units, and loader compatibility. It must not change physics, smooth or replace commands, remove useful manipulation pauses, or silently repair a problematic grasp.

### Review summary

- `/tmp2/leocheng/RoboLab` is the sole working checkout. The `/home/leocheng/RoboLab` copy was deleted; neither collection nor processing should depend on it or require SSD staging.
- Build the offline indexer, validator, per-attempt packager, and camera processing first. These do not require the final proxy checkpoint choices.
- Freeze the PPS export recipe only after choosing both proxy configurations, checkpoint/normalization artifacts, measured-gripper convention, and compatible LeRobot version.
- Keep live teleoperation, F9 reset/READY behavior, mapping/calibration, and standard RoboLab evaluation unchanged in the first offline implementation.
- Validate three successful attempts and at least one failed or interrupted attempt before scaling. Neither existing benchmark exports nor a successful conversion alone establish training readiness.

## Current implementation status

| Component | Status as of 2026-10-03 | Implication for this plan |
| --- | --- | --- |
| OMY state-only recording | Implemented; full robot/object states, initial state, issued actions, pre-action proprioception, timing, and provenance | Reuse the raw contract rather than introducing another live recording format |
| Native 15 Hz collection | Supported explicitly; 120 Hz physics with eight substeps per command | Validate actual simulation timestamps; do not assume CLI settings prove cadence |
| OMY performance profile | Opt-in solver cap 16, held-target stepping, compact bounded buffers, query sharing, and corrected XR deadline pacing are implemented | These are not prerequisites still waiting to be coded; standard evaluation defaults remain separate |
| Space/READY collection workflow | Implemented in `collect_omy_xr.py`; current files remain session-level `omy_demos.hdf5` | Preserve its behavior and consume its events/sidecars |
| Offline state-based camera rendering | Implemented for both OMY teleop and collection metadata, with strict preflight, frame mappings, and per-attempt caches | Camera profile remains provisional until reviewed; historical asset identity is not invented |
| Generic LeRobot v3 export | Implemented, but not suitable unchanged for OMY/PPS | Its discovery, feature names, state/action alignment, and version compatibility need an explicit OMY/PPS path |
| Per-attempt archive and durable publication | Implemented offline, with checksums, stable IDs, source/sidecar stability checks, and collision-refusing atomic publication | Direct collector publication remains a separate follow-up |
| PPS-specific exporter and both-proxy loader acceptance | Numerical mapper and provisional `common_v2` adapter implemented; actual pinned LeRobot/both-loader acceptance not completed | Export recipe completeness, source-bound quality review, and explicit role/split selection gate conversion; no output is certified training-ready |

See [`omy_pipeline_implementation.md`](omy_pipeline_implementation.md) for the implemented optimization and isolation checks. Historical SSD paths in older investigation notes do not define the current working location.

## Attempt identity and saved artifacts

Give every recorded attempt a self-contained directory and a persistent ID:

```text
{TaskName}_{UTCStartTimestamp}_a{AttemptNumber}
```

For example, `BananaInBowlTask_20261003T083012123456Z_a0001` uses a UTC timestamp with microseconds and a one-based, session-local recorded-attempt counter. For future direct collector publication, create the ID when movement starts recording after READY confirmation, not when the simulator launches or while waiting for READY. A reset without recorded actions must not create an empty demonstration. A subsequent recorded attempt gets a new ID. Refuse conflicting destinations rather than overwriting them; store a UUID in metadata for cross-session/host identity and duplicate detection.

The first offline implementation must also package existing session files. Prefer a recorded attempt-start event for its UTC timestamp. If only a session timestamp is available, mark that legacy fallback explicitly; do not invent an exact movement-start time. Persist a stable import mapping from source file checksum and `demo_N` to attempt ID/UUID, so retrying a job or copying the source to another path does not generate a second demonstration. A matching already-published attempt is reusable only after its identity and checksums are verified; a conflicting attempt is an error.

The proposed permanent archive layout is:

```text
/tmp2/leocheng/robolab_datasets/raw/
└── BananaInBowlTask_20261003T083012123456Z_a0001/
    ├── trajectory.hdf5
    ├── metadata.json
    ├── env_cfg.json
    ├── calibration.json
    ├── timing.npz
    └── performance.json
```

Each `trajectory.hdf5` holds exactly one attempt, using the existing episode tree (for example, `data/demo_0`): initial scene state, per-step full robot/object states, actual issued actions, pre-action proprioception, task/subtask results, and simulation/wall timing. Keep the HDF5 numerical recording immutable after finalization. Snapshot the effective configuration and calibration alongside it; identify the scene assets and complete dependencies by recoverable version/checksum rather than duplicating all assets per attempt.

`metadata.json` records the schema version, persistent attempt ID and UUID, session ID, session-local attempt number, task/instruction, known UTC start/end times and their source, seed, outcome, termination/reset reason, sample count, simulation duration/rate, calibration/configuration provenance, runtime versions, and artifact checksums. Use `success`, `failure`, or `interrupted` when supported by recorded evidence; explicitly mark an unknown legacy outcome instead of assuming success or inventing a reset reason. Keep outcome and quality-selection labels out of the directory name so reviewing or reclassifying an attempt never changes its identity. Record storage publication and processing status separately from task outcome.

HDF5 timing is the primary per-step source for compact recordings. Generate `timing.npz` from those values and cross-check existing collection sidecars where available; do not manufacture missing source timestamps. `performance.json` must identify whether its measurements are attempt-level or only session-level. Missing legacy sidecars should be reported, not replaced with another attempt's statistics.

Use three artifact layers:

| Layer | Planned representation | Identity and retention |
| --- | --- | --- |
| Raw source | One state-only HDF5 plus sidecars per attempt | Permanent archive; retain successful, failed, and interrupted recordings |
| Camera materialization | Image-augmented HDF5 with pre-action external/wrist RGB frames | Rebuildable cache keyed by attempt ID and rendering-profile/version |
| PPS training export | LeRobot tabular data (Parquet), image/video assets, and metadata, with the version pinned to both intended proxy loaders | Versioned export selected by role-specific manifests; may combine attempts into efficient shards |

Preserve the attempt ID and UUID through rendered observations, previews, quality reports, reference/task-proxy selection manifests, and final episode metadata or an explicit export-to-source mapping. Every training row must be traceable to its original attempt and source step. Sharding or assigning a new LeRobot episode index must not merge episode boundaries or change source identity.

Directory timestamps identify attempts only; they do not determine sample cadence. For native-15 recordings, validate one action/observation per 1/15 simulated second and encode policy-camera video at 15 FPS. Keep wall-clock stalls in diagnostic metadata, not playback timing. Render the recorded pre-action states; do not replay actions with different physics to manufacture a replacement trajectory.

### Publication and compatibility

Use a temporary directory beside the final HDD destination, such as `raw/.partial-<attempt_uuid>/`. Build the one-attempt HDF5 and sidecars there, close and validate them, synchronize the required files/directories, and publish the complete attempt directory by atomic rename within the destination filesystem. Do not expose a final attempt directory containing only some of its artifacts. Restart must reconcile already-published IDs/checksums and partial attempts without duplication or silent data loss. Never delete or modify the original session recording during import. Boundary-only saving does not protect an unsaved in-memory attempt against a process crash.

This is a planned archive layout, not the collector's current behavior. The existing collector writes multiple `demo_N` groups into a session-level `omy_demos.hdf5`, with session events and per-attempt timing/performance sidecars. Changing `--output-dir` alone does not create per-attempt HDF5 files. First implement packaging offline, leaving the live writer unchanged. Direct per-attempt collector publication may follow after separate approval and boundary-save measurements. Keep the indexer/materializer/exporter able to read both archive files and legacy recordings; preserve raw values and the original session/file identity and `demo_N` in source mappings.

Include checks for unique IDs, one episode per attempt file, failed/interrupted retention, no empty READY/reset recordings, collision refusal, interrupted publication recovery, and source-ID preservation through a sharded training export in the first validation milestone. Empty or structurally damaged legacy episodes receive an index/report entry, not a fabricated complete trajectory.

## Storage placement: HDD runtime and dataset archive

Use `/tmp2/leocheng/RoboLab` for code, the working simulation environment, calibration, and scene assets. Keep the dataset archive separate under a configurable root, proposed as `/tmp2/leocheng/robolab_datasets`. Use that HDD filesystem for raw attempts, materializations, exports, reports, and default temporary processing files. No new directories are created by this plan.

The latest live HDD run reached about 10.4 commands/s after excluding its first 30 commands, compared with about 11.0–11.2 commands/s in warmed SSD saved-command XR benchmarks after the same exclusion. Configuration matched, but trajectory, live-input work, and presentation conditions were not an isolated disk A/B. The observed difference does not establish a large sustained SSD benefit or justify making SSD placement mandatory. The SSD checkout has been removed; do not recreate it as part of postprocessing.

| Data | Planned location | Retention |
| --- | --- | --- |
| Active code, simulation `.venv`, calibration, and assets | `/tmp2/leocheng/RoboLab` | Sole working runtime |
| Current session-level collector output | `/tmp2/leocheng/RoboLab/output/omy_teleop/<run_id>/` | Preserve until archival is verified; removal requires a separate retention decision |
| Raw state-only HDF5 recordings and per-attempt sidecars | `/tmp2/leocheng/robolab_datasets/raw/<attempt_id>/` | Permanent source archive |
| Offline camera materializations | `/tmp2/leocheng/robolab_datasets/rendered/<render_profile_id>/<attempt_id>/` | Rebuildable per-attempt cache |
| PPS-compatible LeRobot exports and encoded videos | `/tmp2/leocheng/robolab_datasets/pps/<export_id>/` | Versioned training datasets |
| Validation manifests, previews, and processing reports | `/tmp2/leocheng/robolab_datasets/reports/<export_id>/` | Provenance and quality records |
| Default processing scratch | `/tmp2/leocheng/robolab_datasets/scratch/<job_id>/` | Temporary, byte-limited working data |

### Live collection

Launch from `/tmp2/leocheng/RoboLab`. The existing default output already stays in that checkout; an explicit run directory may be chosen with the collector's current `--output-dir` option. For example, append:

```bash
--output-dir "/tmp2/leocheng/RoboLab/output/omy_teleop/$(date -u +%Y%m%dT%H%M%S%NZ)"
```

The output must be a unique run directory, not the shared dataset root: the collector refuses an existing nonempty output directory. Preserve the state recording together with `env_cfg.json`, collection events, attempt timing, calibration provenance, and runtime/version metadata. Retain the scene assets or a recoverable version of their complete dependencies so offline rendering can reproduce the recorded scene. Setting `--output-dir` changes placement only; per-attempt packaging, publication, and any writer queue are proposed work.

Keep collection state-only and retain buffered recording. The current collector disables periodic HDF5 flushing and exports at episode boundaries. Its tensor buffers remain on the source device, often GPU memory, until export. Export currently performs GPU-to-CPU transfers and HDF5 writes synchronously, so even a boundary-only save can briefly stop XR updates. Camera rendering, video encoding, compression, and archive copying should remain outside active teleoperation.

### Collection I/O isolation: separate, measurement-driven follow-up

Do not make a background writer or collector redesign a prerequisite for the first offline pipeline. These changes require their own approval and runtime regression checks.

1. **Measure before adding concurrency.** Record episode bytes, GPU and CPU buffer peaks, GPU-to-CPU transfer time, HDF5 write/close time, and XR frame gaps around exports. Compare recording enabled and disabled under otherwise matching conditions. If boundary saves already meet the latency budget, retain the synchronous writer. Existing physics/render timings do not establish disk I/O as the sustained control-loop bottleneck.
2. **Retain existing bounds.** Compact fixed-capacity preallocation, attempt step/time limits, and an episode byte budget are already implemented. If snapshots or a writer queue are introduced, extend accounting to active GPU buffers, completed snapshots, transfer buffers, and pending CPU data; a queue length alone is not a byte limit.
3. **If justified, hand immutable CPU snapshots to one writer.** Keep source buffers alive until transfers finish; reset must not mutate queued data. Measure transfer separately, since a background disk writer does not remove synchronization or copy costs. One worker owns HDF5 handles. Route success export, manual reset, interruption, and associated sidecars through the same ownership model.
4. **Apply backpressure between attempts.** Bound pending bytes/episodes and enforce free-space reserves on the actual output filesystem. If capacity is exhausted or the writer fails, remain waiting, keep XR responsive, and show the status. Never silently discard data. Drain pending writes on normal shutdown; report unsuccessful shutdowns and preserve recoverable partial data.
5. **Acknowledge durable publication.** Distinguish `pending`, `published_raw`, and `failed` from task outcome. Use one closed raw HDF5 plus sidecars per attempt, synchronize and publish the complete artifact set, and only then acknowledge a save. An HDF5 flush alone is not the durability protocol; unacknowledged in-memory data can be lost in a crash.
6. **Keep archival outside manipulation.** Offline packaging copies closed source episodes to a temporary archive destination and verifies values, checksums, readability, and provenance before publication. Leave source files untouched. A future cross-filesystem archive needs verified copy-and-publication, not an assumed atomic move, and must not automatically delete the source without an approved retention policy.

If the follow-up is approved, start with one HDF5 writer. h5py serializes HDF5 API calls within a process, so multiple writer threads are not an assumed throughput improvement; a worker can overlap file I/O with non-HDF5 simulation work. This removes blocking writes from the control thread but does not eliminate shared CPU, memory, or disk contention. Source: [h5py threading documentation](https://docs.h5py.org/en/3.15.0/threads.html).

### Offline rendering and export

1. Read selected episodes from the HDD raw archive. Run expensive materialization and encoding after the teleoperation session, or in a separate scheduled processing window, to avoid competing with XR for CPU, GPU, and disk resources.
2. Write materialized recordings and PPS exports through temporary paths on their destination filesystem. The existing materializer accepts explicit `--input` and `--output` paths and creates scratch beside its output; it provides a starting point, not complete per-attempt manifest/durability handling.
3. Stream image frames to disk instead of accumulating a dataset in RAM. Bound active attempts, image/encoding buffers, scratch bytes, and worker count. Start with one rendering process; measure before adding parallel jobs competing for the same GPU/HDD.
4. Validate sample counts, alignment, metadata, checksums, and file readability before publication. Commit the rendered file and its configuration/provenance together. Mark an attempt complete in the resumable manifest only after its required outputs are durable. A changed source checksum, camera profile, or rendering recipe invalidates the cache.
5. Clean only verified, rebuildable scratch data. Preserve raw recordings and failed-job diagnostics until their retention policy permits removal. Check HDD headroom before admitting a job. Optional SSD scratch or training caches may be considered later if measurements justify them, but are not part of the baseline or a reason to recreate the deleted checkout.

Disk placement does not change the PPS schema, simulated-time sampling, or action/observation alignment. Keep the dataset archive separate from the working checkout so future runtime changes do not redefine or remove source data. Processing dependency isolation is described in the implementation stages below.

This section defines the storage plan only. The first offline milestone must verify atomic per-attempt publication, low-space rejection, interruption/retry behavior, bounded processing memory, and source preservation. Full writer-queue and XR backpressure tests belong to the separate collector-writer follow-up, not to an offline implementation that leaves the live writer untouched.

## PPS training representation

PPS's Isaac Lab demonstration converter writes one row per sample with these features:

| Feature | Shape and meaning |
| --- | --- |
| `exterior_image_1_left` | RGB image, 180 × 320 × 3; external camera |
| `wrist_image_left` | RGB image, 180 × 320 × 3; wrist camera |
| `joint_position` | Seven measured Franka joint angles (`float32`) |
| `gripper_position` | One measured gripper value (`float32`) |
| `actions` | Seven arm values and one gripper command (`float32[8]`) |
| `task` | Language instruction attached to each episode |

The converter declares 15 FPS. It also defines microphone features, but the ordinary RGB DROID policy does not require them. The selected PPS policy transforms resize images to 224 × 224. Its joint-position training configuration treats arm actions as absolute joint targets: it subtracts the measured current arm position to form deltas, while leaving the gripper action absolute. Training examples contain a future action sequence; the inspected π0.5 proxy uses a horizon of 15 actions.

Sources: [PPS Isaac Lab converter](https://github.com/TritiumR/pps/blob/a22f9adddc3ac9827dec38381c72b7d4e9390c5f/openpi/examples/Isaaclab/convert_isaaclab_data_to_lerobot.py), [DROID policy transforms](https://github.com/TritiumR/pps/blob/a22f9adddc3ac9827dec38381c72b7d4e9390c5f/openpi/src/openpi/policies/droid_policy.py), [training configuration](https://github.com/TritiumR/pps/blob/a22f9adddc3ac9827dec38381c72b7d4e9390c5f/openpi/src/openpi/training/config.py), and [data loader](https://github.com/TritiumR/pps/blob/a22f9adddc3ac9827dec38381c72b7d4e9390c5f/openpi/src/openpi/training/data_loader.py). These code links refer to revision `a22f9ad` inspected on 2026-10-01.

## Alignment and units

Our recorder stores an action at step `t` and the resulting scene state after that action. Observations are pre-action: observation 0 belongs to the initial scene state, and observation `t > 0` belongs to post-step state `t − 1`. Therefore, the PPS row at time `t` must pair the pre-action image and measured joint state with the action sent at `t`.

| Training-row component | Source for row `t` |
| --- | --- |
| External/wrist images | Render `initial_state` for `t = 0`, otherwise `states[t - 1]` |
| Measured arm position | Recorded pre-action proprioception, cross-checked against the same state used for the images |
| Measured raw finger angle | Named `finger_joint` from `initial_state` or `states[t - 1]`, not post-action `states[t]` |
| Measured normalized gripper closure | Recorded pre-action `obs/proprio_obs/gripper_pos`, kept distinct from the raw finger angle |
| Absolute arm and gripper command | Issued `actions[t]`, with only the recipe's explicit unit/encoding conversion |
| Dataset timestamp | `sim_before_s[t] - sim_before_s[0]`, checked against the verified sampling grid |

Resolve joints by the recorded joint-name list, not by assuming a fixed index. Keep all thirteen recorded linkage joints in the raw archive even though the training action has eight components. The final post-action state remains in raw data; it is not an extra action/image row.

The PPS example converter shifts its `joint_actions` observation forward one sample. That is specific to its recorder's convention; applying the shift to our direct action recording would misalign the data. Export the mapped and limited Franka command actually sent to the environment, not raw L100 readings and not the robot's achieved joint positions as a substitute for the command.

Our OMY gripper command uses `+1 = open` and `−1 = closed`. The inspected PPS joint-position environment accepts `0 = open` and `1 = closed`, giving the following candidate command conversion for that profile:

```text
pps_gripper_command = (1 - omy_gripper_command) / 2
```

Confirm that conversion against the selected training and evaluation recipe before enabling its export. Preserve the original command internally; do not apply gripper inversion or arm delta conversion twice.

Measured gripper state needs a distinct mapping. The PPS pot converter reads a finger-joint angle, while PPS evaluation can take the eighth raw articulation position. Those paths do not establish a single normalized convention. Keep the raw finger position and normalized closure available internally, and explicitly choose the field and units to match the selected PPS training/evaluation configuration. Do not assume the command encoding is also the correct measured-state encoding. Refuse training export if this recipe choice is missing or incompatible with the checkpoint; indexing, packaging, and camera rendering can still proceed.

Sources: [PPS binary gripper action](https://github.com/TritiumR/pps/blob/a22f9adddc3ac9827dec38381c72b7d4e9390c5f/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/binary_zero_one_joint_actions.py), [PPS pot observations](https://github.com/TritiumR/pps/blob/a22f9adddc3ac9827dec38381c72b7d4e9390c5f/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/pot/mdp/observations.py), and [PPS evaluation input handling](https://github.com/TritiumR/pps/blob/a22f9adddc3ac9827dec38381c72b7d4e9390c5f/eval_steering.py).

## Sampling rate

OMY collection defaults to 60 control steps per simulated second and also supports `--control-hz 30` and `--control-hz 15`, with physics fixed at 120 Hz. The inspected PPS pot environment runs at 15 control steps per simulated second. Setting export metadata to 15 FPS for a 60 Hz recording without resampling would make that trajectory appear four times longer. A native 15 Hz collection already has one command and one pre-action observation for each 1/15 simulated second; its FPS should still be checked against recorded simulation timestamps.

Retain every raw sample in HDF5. The first PPS export profile should accept verified native-15 episodes only and preserve one row for every recorded action. Check both `sim_after_s[t] - sim_before_s[t]` and consecutive `sim_before_s` differences against 1/15 s, using an explicit float64 timestamp tolerance. Configuration metadata alone is not sufficient. Do not insert, drop, interpolate, or duplicate samples to disguise gaps.

Read and archive older 30/60 Hz episodes, but do not silently export them under a 15 FPS label. A later resampling profile must select samples by simulation time, document discarded intermediate commands, and pass separate contact-rich quality checks. Keep wall-clock timestamps for latency analysis; they must not redefine simulated motion.

For an `N`-sample native-15 episode, exported row timestamps are `0, 1/15, ..., (N - 1)/15`, and the `N` action intervals span `N/15` simulated seconds. Video frames use the same row indices and 15 FPS cadence. Slow rendering or encoding during offline processing changes job duration, not dataset duration.

### Collection performance and limits of postprocessing

The implemented OMY candidate keeps CUDA physics at 120 Hz, caps scene position iterations at 16, uses held-target stepping and compact recording, and schedules nominal 30 Hz XR rendering. The corrected deadline pacer avoids unnecessary waits for overdue renders. These changes are opt-in and do not establish 15 Hz wall-clock operation for every grasp. Twelve/eight solver-iteration experiments did not justify lowering the grasp profile's cap further.

The latest live HDD attempt recorded 116 commands over exactly 7.7333 simulated seconds, with a successful task result. Its overall wall rate was 9.7 commands/s including a roughly one-second first-step spike; after excluding 30 startup commands it was about 10.4 commands/s. HDF5 checks found the intended 1/15 s intervals, cap 16, held-target stepping, and no nonfinite numerical leaves. Two startup commands lacked valid fresh leader input, which belongs in the quality report rather than being silently erased. Evidence: [`latest raw run`](../output/omy_teleop/20261003_194913_828617/omy_performance.json) and [`XR investigation`](../output/omy_pipeline_benchmark/20261003_next_target/README.md).

Correct simulated-time timestamps produce a 15 Hz dataset without encoding wall-clock compute waits as slow-motion playback. They do not make the operator's experience equivalent to realtime: lag-induced hesitation, corrective movements, slips, or bad contacts remain part of the trajectory. Preserve useful pauses and review those defects; do not automatically time-warp demonstrations. Higher-quality offline rendering can improve images, not repair the recorded physics. Increasing solver quality and replaying actions to create a different trajectory is not part of this plan.

The historical investigation is in [`omy_pipeline_optimization.md`](omy_pipeline_optimization.md); the implemented status is in [`omy_pipeline_implementation.md`](omy_pipeline_implementation.md). A 15 Hz wall-clock performance guarantee is not a gate for building the offline pipeline. Timestamp/alignment correctness and representative grasp quality are gates for accepting training episodes.

## Implementation stages for approval

The intended order is **index/validate → package → render → export → both-proxy acceptance**. Each stage should produce an inspectable result and be independently retryable. A proposed OMY-specific entry point is `scripts/postprocess_omy.py` with stage-specific commands; this script does not exist yet and the interface will be finalized during implementation. Do not turn the generic exporter into a mandatory dependency of live teleoperation.

### Stage 1 — read-only indexing and validation

- Discover explicit inputs and session-level `omy_demos.hdf5` as well as packaged `trajectory.hdf5`; do not rely on the generic exporter's `data.hdf5` discovery rules. Read both `robolab_omy_teleop` and `robolab_omy_collection` metadata.
- Process only closed/finalized inputs, not the file a live collector is still appending to. Verify source identity/checksums remain stable through import. Preserve original attempt UUID/source mappings when importing an already-packaged file; use numerical episode fingerprints to flag possible duplicated content across repackaged sources for review and split isolation.
- Validate initial-state availability, action/state/proprioception/timing row counts, finite values, known joint names/order, expected action dimensions, simulation-time continuity, and consistency with effective configuration. Distinguish per-step leaves from event or episode-summary arrays; subtask summaries need not have one entry per action.
- Cross-check pre-action joint observations against initial state and shifted post-step states with documented numerical tolerances. Report missing provenance, stale/invalid leader samples, configuration fallback, reset boundaries, and unusually large wall gaps. Structural failures and quality warnings must be separate.
- Write an index and machine-readable quality report per attempt, including source checksum/group, outcome evidence, sample count, simulation rate/duration, timing diagnostics, and acceptance/rejection reasons. Do not launch Isaac Sim, open the serial port, or modify raw inputs during this stage.
- Default training selection is structurally valid, successful, native-15 attempts without unresolved quality flags. Flagged attempts require an explicit recorded review decision; failures, interrupted attempts, and unknown outcomes remain available in the archive. Never silently trim stale samples or reset artifacts to make validation pass.

Deliverable: repeatable indexing/validation of real recordings and synthetic malformed fixtures, without needing a selected checkpoint or LeRobot installation.

### Stage 2 — immutable per-attempt packaging

- Copy each complete legacy episode into `raw/<attempt_id>/trajectory.hdf5`, preserving numerical values, dataset attributes, and source metadata. Keep the original episode/group identity in the manifest even if the one-attempt destination uses `demo_0`.
- Create configuration/calibration/timing/performance sidecars with their true provenance and completeness status. Preserve task/subtask/event records that exist; do not invent unrecorded contact forces or silently substitute newly evaluated task scores.
- Use stable IDs/UUID mappings, checksums, collision refusal, a temporary artifact directory, validated durable publication, and restart reconciliation as described above. Preserve sources and all existing artifacts on failure.
- Missing/invalid legacy inputs receive a report and quarantine/selection status; do not fabricate missing states or timestamps. Per-attempt packaging is not permission to delete session recordings.

Deliverable: independently verifiable one-attempt archives and idempotent reruns, with no collector/writer changes.

### Stage 3 — state-based policy-camera materialization

- Reuse the existing pre-action state-restoration renderer. Extend task/environment selection to accept both collection and recorded teleop metadata and choose the OMY absolute-joint variant explicitly; no silent relative-IK fallback.
- Restore recorded configuration and validate recoverable assets, referenced USD layers, meshes, textures, and material dependencies. Legacy absolute paths may be remapped from an old checkout only through an explicit, recorded root mapping and verified asset identity; do not silently render today's different scene.
- Define a versioned camera/render profile: DROID external-left and robot-mounted wrist cameras, calibrated poses/intrinsics, resolution, RGB convention, renderer settings, and passes per frame. The Quest head-tracked viewpoint, XR resolution multiplier, and ALVR video stream are not policy-camera observations. The inspected PPS converter's 320 × 180 RGB layout is a compatibility candidate, not permission to crop/reconfigure cameras without recording the recipe.
- Render `initial_state` for frame 0 and `states[t - 1]` thereafter, with no physics stepping between these restores. Record rendering configuration separately from collection physics; higher-quality render settings must not replace or reinterpret the raw trajectory.
- Warm the renderer, invalidate/update scene and camera caches, and verify fresh frames at frame 0, wrist motion, grasp, lift, and release. Inspect black/stale frames, ghosting, missing/white textures, colors, and gripper/object geometry before bulk processing. Additional render passes must be measured and recorded, not assumed to fix every image artifact.
- Stream frames into per-attempt materialized HDF5, preserve non-image channels, and publish images plus camera configuration/provenance together. Cache by source checksums, asset/configuration identity, camera profile, and renderer/software recipe. Resume only validated complete attempts; safely retry incomplete ones.

Deliverable: aligned external/wrist RGB recordings, previews, and reproducible per-attempt render reports. No live simulator or leader-control code changes are required.

### Stage 4 — recipe-gated PPS export

- Implement an OMY/PPS-specific export path rather than using the generic v3 exporter unchanged. Specify exact feature names, dtypes, image/video layout, joint order, gripper state/command encodings, task labels, episode boundaries, and version metadata in a saved recipe.
- Retain seven absolute issued arm targets in radians. Let the selected PPS transform produce arm deltas where configured. Validate gripper state and command separately against both recipes and teacher normalization; retain original values and conversion provenance for round-trip comparisons.
- For the first native-15 profile, export every accepted sample with verified simulated-time timestamps and matching 15 FPS images/video. Preserve frame count and source-step mappings, including episode ends and short action-horizon cases. Do not resample or concatenate across attempt boundaries.
- Produce separate reference/task selection manifests over the shared archive. Include attempt UUIDs, source steps, export recipe/version, normalization identity, task coverage, and deterministic episode-level splits. Copied or repackaged instances of the same source episode must not enter opposing splits; coordinate held-out evaluation across both roles.
- Use bounded-memory tabular/video writing, configurable shard sizes, and verified temporary publication. Save machine-readable provenance and quality reports with the export. Do not upload to a remote dataset service as part of local conversion.
- Refuse a training-ready designation until the exact chosen PPS loaders accept the exported dataset. If the selected LeRobot version lacks our desired fields, keep provenance in an explicit sidecar mapping rather than assuming arbitrary feature support.

Deliverable: a reproducible local dataset compatible with the selected, pinned recipes for both proxies. The unresolved decisions below gate this stage, not stages 1–3.

### Stage 5 — end-to-end acceptance and scaling

Process three representative successful BananaInBowl attempts and at least one failed or interrupted attempt. Compare numerical output against the immutable sources, inspect key camera frames and previews, exercise recovery/failure cases, and load batches through both selected proxy configurations. Record the test results and approved recipe before admitting bulk jobs.

Scale only after acceptance: process bounded batches, reuse verified per-attempt caches, and benchmark HDD read/write, encoding, memory, and loader throughput before adding worker concurrency or optional scratch placement. Training adequacy/generalization needs a separate dataset-size and policy-evaluation study.

### Processing dependency isolation

`h5py` and NumPy are available in the current simulation environment; `pyarrow` and LeRobot are not installed there. Their absence does not block planning or the lightweight first stages. Do not upgrade, replace, or run an unconstrained dependency sync against the working `.venv` merely to make export imports succeed.

Run camera materialization in the known-working Isaac Lab environment. Use a separate, pinned processing environment for Parquet/LeRobot export and loader compatibility, with a repeatable dependency specification/lockfile chosen against the PPS revision. Pass durable file paths/manifests between the rendering and export processes. Test import, file-schema, video-decoding, and both-loader compatibility in that environment; avoid accidental imports of Kit or opening the leader device during export. Environment installation is future implementation work requiring approval, not part of this documentation update.

## Existing RoboLab pieces

The offline camera materializer in [`scripts/materialize_demo_images.py`](../scripts/materialize_demo_images.py) and [`robolab/core/replay/materialize.py`](../robolab/core/replay/materialize.py) rebuilds an environment and renders observations from recorded states with pre-action alignment. Before this update its OMY branch required `robolab_omy_collection`; the offline implementation now recognizes recorded `teleop_omy_xr.py --record` sessions through `robolab_omy_teleop` too. No manually supplied task or unrelated relative-IK registration is needed.

The materializer writes via a temporary HDF5 path. The OMY wrapper now adds per-attempt resumability, source/profile and resolved-dependency checksum checks, configuration-sidecar publication, and durable artifact-set acknowledgment. Pure tests and a saved-state render of the latest HDD recording verify the mapping; these are not certification of the final PPS dataset.

RoboLab also has a generic LeRobot v3 exporter in [`robolab/core/export/lerobot_exporter.py`](../robolab/core/export/lerobot_exporter.py), invoked by [`scripts/convert_to_lerobot.py`](../scripts/convert_to_lerobot.py). It expects `data.hdf5` or `run_*.hdf5`, uses different feature names, and pairs actions with post-step joint positions. It also generates timestamps from the configured FPS without resampling. These differences mean a separate OMY/PPS export path and alignment handling are required before it can serve this collection. Preserve the generic exporter's current behavior for its existing users.

## Acceptance checklist

The first end-to-end milestone needs evidence for all of the following:

- **Raw preservation:** source checksums remain unchanged; packaged values, attributes, full linkage states, initial state, and issued commands match the source.
- **Identity:** one episode per packaged file, stable retry IDs, no duplicate source episodes, explicit mappings for every exported row, and no empty READY/reset attempts manufactured by processing.
- **Structural validation:** count/shape mismatch, nonfinite values, missing states/provenance, unknown joint order, and invalid timing are reported or rejected; event summaries are not mistakenly treated as per-step channels.
- **15 Hz contract:** every accepted command advances 1/15 simulated second, consecutive rows are continuous, exported timestamps/15 FPS video agree, and wall stalls cannot lengthen dataset playback. Older rates are refused by the first export profile rather than relabeled.
- **Alignment:** all numerical pre-action observations are checked against the source; frame 0 and later render-state choices are verified with controlled fixtures and inspected real frames. Commands are not shifted using the PPS example recorder's convention.
- **Gripper units:** open, closed, and intermediate measured states and issued commands map correctly through both selected policy transforms and normalization artifacts; inverse conversion recovers raw values within stated tolerances.
- **Image quality:** both cameras have the agreed dimensions, dtype/color convention, poses, and fresh frames; rendered grasp/lift/release geometry matches recorded states. A task-success flag alone does not establish a visually or physically clean demonstration.
- **Selection and splits:** failed/interrupted/unknown or unresolved flagged attempts are retained but excluded by default; review overrides are recorded; duplicate sources and held-out episodes cannot leak across role-specific train/validation/evaluation splits.
- **Recovery and resources:** collisions, low disk space, partial output, interrupted jobs, changed cache inputs, and publication errors cannot overwrite sources or advertise incomplete outputs as complete. Memory/scratch use stays within configured budgets.
- **Source stability:** actively written or changed inputs are refused; both legacy session imports and packaged-file reimports preserve identity without generating extra training episodes.
- **PPS compatibility:** both intended training configurations load and transform an exported batch, including action horizons and episode ends, without feature/unit/version mismatches. Save the compatibility report with the recipe.
- **Isolation:** standard RoboLab evaluation and generic export behavior remain unchanged; processing does not require Quest/SteamVR, open the L100 port, or run heavy jobs concurrently with teleoperation by default.

This milestone establishes format, preservation, camera, and loader correctness, not that three episodes are sufficient to train either proxy. If suitable retained recordings do not cover the milestone, collect new validation attempts after implementation; do not substitute benchmark replays as evidence of representative human demonstration quality.

## Confirmed training targets: reference and task proxies

The collected data will be used to train both the reference proxy and task proxy in the PPS workflow. Retain one raw archive and reuse validated camera materializations. Express the episode/task selection for each role in versioned manifests rather than assuming that the two proxies use identical data or duplicating raw recordings. Record the source episode IDs, task labels, train/validation assignments, and export version so each training dataset is reproducible. Coordinate evaluation splits across roles to prevent unintended training/evaluation overlap.

Both roles must share the agreed camera schema, joint order, action and gripper units, sampling convention, and pre-action alignment. Normalization must be checked against the selected frozen base policy and proxy training recipes: PPS supports reusing teacher normalization during distillation, so independently normalizing their action spaces can make their comparison inconsistent. Store the normalization identifier and provenance in each training manifest, and validate the transformed actions through both training configurations before bulk export.

### Decisions to settle before freezing the training export

| Decision | Required record or check | Blocks |
| --- | --- | --- |
| PPS revision and both proxy configurations | Exact source revision, configuration names, feature/repack transforms, action horizon, and loader behavior | PPS compatibility certification |
| Base/teacher checkpoint and normalization | Checkpoint identity plus recoverable normalization artifact/checksum; verify each proxy's intended use | Training-unit validation and training-ready export |
| Measured gripper convention | Named raw finger angle versus normalized closure, units/range, and any required explicit conversion | Final `gripper_position` export |
| Gripper action convention | Confirm whether the inspected 0-open/1-closed mapping applies to the selected recipes | Final action export |
| LeRobot format and dependency versions | Exact supported version/schema, video decoding behavior, and a passing batch through both loaders | Export environment and final dataset writer |
| Camera/render profile | Review external/wrist placement, resolution, image transforms, renderer/pass count, and source/asset provenance | Approved camera cache and bulk materialization |
| Reference/task coverage and splits | Versioned role manifests, reviewed episode selection, and coordinated held-out tasks/episodes | Training dataset selection; a BananaInBowl compatibility pilot may precede full coverage |

Do not guess missing choices from field names or normalize each proxy independently. Stages 1–2 can proceed after implementation approval; stage 3 can produce review previews under an explicitly provisional camera profile. Stages 4–5 must resolve the applicable recipe choices before labeling an export training-ready.

### Scope excluded from the first implementation

Direct per-attempt publication inside the live collector, background HDF5 writers, new SSD staging, controller-based XR control, leader mapping changes, solver reductions, standard evaluation modifications, 30/60-to-15 resampling, automatic trajectory cleanup, bulk training, and remote dataset upload are not part of the first offline implementation. They require separate evidence and approval where applicable.

Next action: review the generated policy-camera frames and settle the pinned training recipe, then test representative attempts and load batches through both intended PPS proxy configurations. Do not bulk-process or label data training-ready based only on successful numerical mapping or the provisional writer tests.

Source: [PPS distillation script](https://github.com/TritiumR/pps/blob/a22f9adddc3ac9827dec38381c72b7d4e9390c5f/openpi/scripts/distill_pytorch.py).
