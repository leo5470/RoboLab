**Mimic implementation progress and review gates**

Started 2026-09-16. This is the execution log for the [implementation plan](mimicgen_implementation_plan.md). The first milestone implements source auditing and a DROID Mimic source-review environment. Annotation and synthetic generation remain gated on source review.

**Implemented interfaces**

- `robolab/datagen/audit.py`: read-only HDF5/config validation, source/config fingerprints, installed versions, and a static benchmark task census. The census identifies candidates; it does not verify success/scoring consistency or declare tasks generation-ready.
- `robolab/datagen/envs/droid_mimic_env.py`: a `ManagerBasedRLMimicEnv` subclass using the recorded relative-IK scale, actual controlled `base_link` pose, root-frame object poses, and DROID gripper conventions. Unsupported action layouts, clipping, and nonidentity body offsets are rejected explicitly.
- `robolab/datagen/envs/registration.py`: reconstructs the recorded task values and registers a separate review environment. Task success is queried explicitly; the runner owns episode boundaries. Evaluation freezing and automatic export are not used.
- `scripts/datagen/review_source.py`: checks reconstructed targets against the live IK controller, measures replay drift, runs sequential trials, and optionally renders recorded states alongside physics replay. State reconstruction is labeled separately and is not evidence of successful physics execution.

The review runner uses the existing predicate extractor for task intent. It requires one explicit success term. General task conversion, branch selection, manual boundary overrides, Mimic metadata export, generation retries, multi-environment validation, and SkillGen remain unimplemented.

**Reproduce the first milestone**

Use the collection environment (currently Isaac Lab 2.2.0 / Isaac Sim 5.0.0.0). Run from the repository root. Output reports/directories must be new paths.

```bash
.venv/bin/python -m pytest --confcutdir=tests/unit tests/unit/test_datagen_audit.py tests/unit/test_datagen_review_video.py -q

.venv/bin/python scripts/datagen/audit_sources.py \
  --input output/keyboard_demos/banana_wifi/keyboard_demos.hdf5 \
  --output output/mimic_review/source_audit_new.json

OMNI_KIT_ACCEPT_EULA=Y .venv/bin/python -u scripts/datagen/review_source.py \
  --input output/keyboard_demos/banana_wifi/keyboard_demos.hdf5 \
  --output-dir output/mimic_review/replay_new \
  --headless --device cuda:1 --physics-device cpu --reference-video
```

`--device` selects the renderer GPU through AppLauncher. `--physics-device` controls the environment physics device; keep it consistent with collection for the baseline comparison. The review video uses the saved viewer pose and 960×540 resolution. This command uses two sequential trials by default. Always inspect `review.json` status and errors as well as the process exit code; simulator shutdown may hide an exception in terminal output.

**User review gates**

1. Source replay: review the reference/replay comparison, final task success, motion quality, and measured drift. Decide whether the source can proceed to annotation or requires repair/recollection.
2. Annotation: review proposed grasp/lift and placement boundaries on synchronized video before accepting the segment specification.
3. First generation: review a small batch of successes and failures before accepting synthetic trajectories.
4. Randomization: review proposed pose ranges and example scene resets before expanding them or scaling the dataset.

Passing automated checks does not accept a user review gate. Decisions and the exact reviewed artifact paths should be recorded here as work progresses.

**Current evidence**

- Source-audit and video regression tests: 19 passed.
- Source: one structurally valid, successful 579-action BananaInBowl episode, 15 Hz control; no existing Mimic metadata.
- Static census: 95 standard-helper candidates and 25 custom-review tasks.
- Live controller-target checks passed; maximum action/target round-trip error was `3.58e-7`. Gripper extraction passed for two- and three-dimensional action arrays; object/root-frame consistency passed in the single source environment.
- Two full 579-step physics replays completed after separate resets. Both failed the authoritative success predicate. First end-effector position error above 1 mm occurred at 16.53 s and 17.00 s. Peak position errors were 66.92 mm and 64.25 mm for the end effector, and 55.85 cm and 52.68 cm for the banana. This does not establish the cause of contact divergence.
- Original HDF5 and saved environment configuration hashes are unchanged.
- No source motion or annotation has been accepted for generation. The source-review gate is pending the user's decision.

**Source review artifacts — pending decision**

Use `output/mimic_review/replay_004/`:

- [Side-by-side video](../output/mimic_review/replay_004/source_vs_replay_browser.mp4): 38.6 seconds; left is recorded-state visual reconstruction, right is actual action replay. The left side is not a new successful physics rollout.
- [Final comparison frame](../output/mimic_review/replay_004/comparison_0578.jpg).
- [Drift plot](../output/mimic_review/replay_004/replay_drift.png).
- [Machine-readable result](../output/mimic_review/replay_004/review.json), with source/config/code fingerprints, versions, controller checks, task intent, replay results, and video frame counts.
- Per-step metrics: `replay_metrics_0.json` and `replay_metrics_1.json` in that directory.
- [Source audit and task census](../output/mimic_review/source_audit_v2.json).

The videos contain all 579 frames. The renderer returned 1280×720 despite the requested viewer resolution, so capture resizes frames explicitly to 960×540 per panel. Capturing a camera-free headless environment requires an explicit render after each step. Earlier attempts `replay_001`–`replay_003` are diagnostic only; `replay_003` is explicitly marked invalid for video review because its viewport buffer was stale.

Recommended next experiment, after source review: reconstruct commanded end-effector targets from recorded **pre-action** states and actions, then test target-tracking replay. This distinguishes accumulated relative-action drift from failures that remain when tracking the demonstrated trajectory. It is not implemented or accepted yet. If the source motion itself is unsuitable, collect a cleaner source before annotation. Do not proceed directly to synthetic generation on the strength of the source's recorded success flag.

Remaining phase-0/1 work includes full runtime task-semantics auditing, original asset/code provenance where available, nonidentity-root/multiple-environment frame tests, and generation recorder/reset validation. The current result is a source-review prototype, not completion of the full implementation plan.
