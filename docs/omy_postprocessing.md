# Offline OMY postprocessing: implemented usage

The entry point is [`scripts/postprocess_omy.py`](../scripts/postprocess_omy.py). Indexing, packaging and numerical mapping use only the existing HDF5/NumPy environment. Rendering starts a separate headless Isaac Lab process. Nothing here opens the L100 port, starts XR, changes live collection, or modifies standard RoboLab evaluation.

The design and outstanding training decisions remain in [postprocess.md](postprocess.md). This implementation preserves raw trajectories; it does not repair bad grasps or remove manipulation pauses.

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
