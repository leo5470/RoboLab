# Deferred-Image Demo Collection Implementation Brief

Implement deferred-image keyboard demonstration collection in the RoboLab repository.

Repository:

```text
/tmp2/leocheng/forks/RoboLab
```

## Goal

Make keyboard teleoperation responsive by collecting actions and simulator states without rendering policy-camera observations during the live attempt. Add an offline tool that reconstructs aligned camera observations from the recorded states and produces a standard RoboLab HDF5 demonstration.

## 1. Live Collection Mode

Extend:

```text
examples/collect_keyboard_demos.py
```

Add:

```text
--defer-images
```

When enabled:

- Do not create or compute policy-camera sensors or observation groups during teleoperation.
- Keep the Kit/WebRTC viewport operational; it does not depend on policy-camera observations.
- Record:
  - Initial simulator state
  - Actions
  - Per-step scene states
  - Success
  - Seed
  - Environment configuration
  - Action/control timestep metadata
- Do not record `obs/image_obs` or `obs/viewport_cam`.
- Preserve proprioceptive observations if inexpensive and needed.
- Failed attempts must be removable without leaving large HDF5 holes.
- Preserve existing behavior when the flag is absent.
- Print the effective recording mode and measured control rate.

Consider extending:

```text
robolab/registrations/droid/auto_env_registrations_rel_ik.py
```

It already supports `include_viewport_camera`; add a clean way to omit policy-camera sensors entirely while preserving the WebRTC viewer.

## 2. Offline Image Materialization

Create:

```text
scripts/materialize_demo_images.py
```

Example interface:

```bash
python scripts/materialize_demo_images.py \
  --input output/keyboard_demos/state_only/keyboard_demos.hdf5 \
  --output output/keyboard_demos/final/keyboard_demos.hdf5 \
  --device cuda:0 \
  --hdf5-compression gzip
```

The tool must:

- Load the recorded `env_cfg.json`, initial state, actions, per-step states, seed, and metadata.
- Reconstruct the environment with the original policy-camera configuration.
- Restore recorded simulator states for rendering instead of relying only on open-loop action replay. Contact-rich action replay may drift.
- Generate each pre-action observation with correct temporal alignment:
  - Observation 0 corresponds to `initial_state`.
  - Observation `t` corresponds to recorded post-step state `t - 1` for `t > 0`.
- Render and store policy-camera observations under the same HDF5 paths and shapes expected by ordinary RoboLab image demonstrations.
- Copy all non-image trajectory fields without changing their values.
- Preserve episode attributes, provenance, success status, and environment metadata.
- Write atomically through a temporary output file, then rename after successful completion.
- Never overwrite an existing output unless an explicit `--overwrite` option is supplied.
- Validate that image count, action count, and state count are aligned.
- Close Isaac Sim and HDF5 handles cleanly on errors and interrupts.
- Support selecting individual episodes and processing all episodes.

## Relevant Existing Code

- `examples/run_recorded.py`
- `robolab/core/replay/scene_state.py`
- `robolab/core/replay/env_config.py`
- `robolab/core/logging/streaming_hdf5_handler.py`
- `robolab/core/logging/recorder_manager.py`
- `robolab/core/events/basic_recorders.py`
- `robolab/core/observations/observation_utils.py`
- `robolab/registrations/droid/auto_env_registrations_rel_ik.py`
- `docs/replay.md`
- `docs/keyboard_teleoperation.md`

## Important Constraints

- Do not regenerate images through open-loop actions alone.
- Avoid calling `env.reset_to()` for every frame if it resets counters, recorders, or episode state. Prefer the lowest-level safe scene-state restoration API, followed by simulator synchronization and rendering.
- Ensure articulation joints, rigid-object poses, velocities, and camera configuration are restored correctly.
- Do not record the third-person `viewport_cam` unless explicitly requested.
- Keep normal recorder defaults unchanged outside the keyboard deferred-image path.
- Preserve unrelated working-tree changes.
- Use `apply_patch` for edits.

## Testing

Add focused tests for:

- Deferred collection config contains no policy-camera sensors.
- State-only episodes contain actions and states but no image observations.
- Temporal mapping uses initial state for frame 0 and post-state `t - 1` afterward.
- Materialized image count equals action count.
- Non-image arrays are copied exactly.
- Existing output protection and `--overwrite`.
- Interrupted conversion leaves no partial final file.
- Existing recorder and replay tests remain passing.

Perform at least one `BananaInBowlTask` smoke test if the simulator is available. Report:

- State-only live control rate
- Materialized camera names and resolutions
- Action, state, and image counts
- Final HDF5 structure
- Tests executed and their results

Update:

- `docs/keyboard_teleoperation.md`
- `docs/replay.md`

Document the live collection command, offline materialization command, temporal alignment, storage requirements, and replay-fidelity limitations.
