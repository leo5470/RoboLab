# Replaying Recorded Episodes

RoboLab records every episode to HDF5 ([Data Storage and Output](data.md)). `examples/run_recorded.py` replays a recorded episode: it rebuilds the environment, restores the recorded initial scene state, and steps the recorded actions open-loop while the usual termination checking, subtask tracking, and video/HDF5 recording run as normal. Use it to verify recordings, sanity-check task conditionals against a known trajectory, or debug environment changes against a reference episode. The replay helpers (config overlay, state restore, validation, provenance) live in `robolab/core/replay/` for use in your own drivers.

For rendering camera observations from recorded states without re-stepping physics — how a state-only keyboard recording becomes an image demonstration — see [Materializing Deferred Images](#materializing-deferred-images-scriptsmaterialize_demo_imagespy).

## Quick Start

Replay the bundled demonstration:

```bash
python examples/run_recorded.py --headless
```

Replay your own recording:

```bash
python examples/run_recorded.py --task <TaskName> --recorded-data-folder <folder> --headless
```

The script expects this layout:

```
<recorded-data-folder>/
└── <TaskName>/
    ├── data.hdf5        # the recording (override the filename with --file)
    └── env_cfg.json     # env config saved by the recording run (used by default, see below)
```

Evaluation runs already produce this layout (they write `run_{i}.hdf5` instead of `data.hdf5`), so an eval output replays directly with `--file`, and `--episode` selects the demo within the file:

```bash
# Replay env 2's episode from run_0.hdf5 of an eval output
python examples/run_recorded.py --task <TaskName> --recorded-data-folder output/<run_folder> --file run_0.hdf5 --episode 2 --headless
```

By default `demo_0` is replayed; in a multi-env recording, `demo_i` is env `i`'s episode. Results are written to `output/playback_<folder>_<task>/`, including the replay's own exported HDF5 (`run_<episode>.hdf5`), videos, and subtask logs.

## What Playback Restores

A faithful replay needs three things from the recording, and all three are restored by default:

| Restored | Source | Notes |
|---|---|---|
| Initial scene state | `data.hdf5` → `demo_0/initial_state` | Robot joints and object poses are reset via `env.reset_to()`, exactly as recorded. Without this, a fresh `reset()` re-settles objects from USD poses and the replay diverges mid-episode. |
| Actions | `data.hdf5` → `demo_0/actions` | Stepped open-loop; no policy is involved. |
| Environment config | `env_cfg.json` sidecar | Object init poses, physics/solver params, termination params, seed, instruction — the exact values the episode was recorded with (see next section). |

Runtime choices always come from the CLI, never the recording: `--num_envs`, `--device`, `--headless`, rendering settings, and recorder configuration.

## Replaying with the Recorded Env Config (`--env-config`)

By default (`--env-config recorded`), playback overlays the `env_cfg.json` saved next to the recording onto a freshly built config, so replay is unaffected by later changes to the repo's task and scene definitions. Fields that no longer exist in the current config schema (or changed type) are skipped and listed in a warning.

Pass `--env-config current` to rebuild the config from the current repo instead — useful to ask "how does this recorded trajectory behave under my *new* task definition?". A yellow warning is printed whenever the recorded config is not fully in effect (sidecar missing, `current` requested, or fields skipped), since the env config then differs from recording time and behavior may diverge.

The overlay restores config **values**, not code or assets. It cannot protect against:

- changed predicate/conditional *implementations* (the config stores only the callable's import path),
- changed USD file *contents* on disk (the config stores only asset paths),
- a different simulator stack (see below).

## Faithful Reproduction Checklist

Contact-rich physics amplifies tiny numerical differences, so reproducing a recorded outcome requires matching the recording context:

1. **Same simulator stack.** IsaacSim 5.0 and 5.1 ship different PhysX builds; recorded outcomes are not invariant across them. Recordings carry `isaaclab_version` / `isaacsim_version` / `recorded_at` HDF5 attrs and playback prints a notice on mismatch.
2. **Single env, recorded and replayed.** All parallel envs share one batched physics scene, so a trajectory recorded in a multi-env batch evolves slightly differently when replayed alone (and vice versa). Record with `--num_envs 1` and replay with `--num_envs 1` (the default); playback prints a notice when replaying with more.
3. **Recorded env config in effect.** Keep the `env_cfg.json` sidecar next to the recording and leave `--env-config recorded` (the default).
4. **Bundle the replay's own export.** Replay→replay is deterministic. To create a demonstration file that reproduces reliably, replay a recorded success once and keep the HDF5 that the *replay* exports (`output/playback_.../run_0.hdf5`, small — no image observations) together with the `env_cfg.json` from that same playback folder. The bundled `examples/recorded_data/RubiksCubeAndBananaTask/` demo was produced this way.

Do not add settling steps before replay: settling is part of the recorded action stream, and the initial-state restore already puts the scene in the exact recorded pre-settle state.

## Validating Replay Fidelity (`--validate-states`)

The recording also stores the full scene state at every step (`demo_0/states`). `--validate-states` compares the simulated state against it each step and reports drift — turning "the replay diverged" from a guess into a measurement:

```bash
python examples/run_recorded.py --headless --validate-states
```

A faithful replay reports:

```
STATE VALIDATION: replay tracked the recording over 648 steps; max drift 0.0000 on None (tolerance 0.01).
```

A diverging replay reports when and where it left the recording:

```
STATE VALIDATION: drift first exceeded tolerance 0.01 at step 431 (rigid_object/banana/root_pose: 0.0413).
STATE VALIDATION: replay diverged from the recording. Max drift 42.2192 at step 406 on
rigid_object/banana/root_velocity; first exceeded tolerance 0.01 at step 0. Fields over tolerance (8/12): ...
```

Comparison uses env 0 (matching the single-env recipe). Note that quaternion sign flips (`q` vs `-q` encode the same rotation) are not normalized, so a pose drift of ~2.0 on an otherwise tracking replay usually indicates a sign flip rather than a real divergence.

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--task`, `-t` | `RubiksCubeAndBananaTask` | Task name; must be a folder inside the recorded data folder. |
| `--recorded-data-folder`, `--dir` | `examples/recorded_data` | Folder containing `<TaskName>/<file>`. |
| `--file` | `data.hdf5` | HDF5 filename inside `<recorded-data-folder>/<TaskName>/` (e.g. `run_0.hdf5`). |
| `--episode` | `0` | Demo index to replay (`demo_<episode>`; in multi-env recordings `demo_i` is env `i`'s episode). |
| `--env-config {recorded,current}` | `recorded` | Replay with the recorded `env_cfg.json` (faithful) or the current repo's config. |
| `--validate-states` | off | Compare sim state against the recorded per-step states and report drift. |
| `--num_envs` | `1` | Parallel envs; keep at 1 for faithful reproduction. |
| `--disable-subtask` | on | Disable subtask progress checking during replay. |
| `--headless` | off | Run without a viewer (standard AppLauncher flag). |

The materialization tool has its own flags; see [Materializing Deferred Images](#materializing-deferred-images-scriptsmaterialize_demo_imagespy).

## Materializing Deferred Images (`scripts/materialize_demo_images.py`)

Keyboard demos collected with `--defer-images` ([Keyboard Teleoperation](keyboard_teleoperation.md#deferred-image-collection---defer-images)) hold actions and per-step states but no camera observations, so nothing renders during teleoperation. `scripts/materialize_demo_images.py` renders those images afterwards and writes an ordinary RoboLab image demonstration:

```bash
python scripts/materialize_demo_images.py \
  --input output/keyboard_demos/state_only/keyboard_demos.hdf5 \
  --output output/keyboard_demos/final/keyboard_demos.hdf5 \
  --device cuda:0 \
  --hdf5-compression gzip
```

This is *not* a replay. `run_recorded.py` re-steps the recorded actions and lets physics evolve; materialization never steps physics at all. Each frame writes a recorded scene state straight into the simulator (`InteractiveScene.reset_to`, then `sim.forward()` → `sim.render()` → `scene.update()`), so the rendered image is of the state the recording actually reached. Open-loop action replay would be the wrong tool here: contact-rich trajectories drift, and the images would slowly stop matching the actions and states stored beside them.

### Temporal alignment

RoboLab records observations pre-step (`PreStepFlatPolicyObservationsRecorder` stores `env.obs_buf`, computed at the end of the previous step), so observation `t` is the scene *before* action `t` is applied:

| Frame | Rendered from | Because |
|---|---|---|
| `0` | `demo_N/initial_state` | the observation before the first action is the reset scene |
| `t > 0` | `demo_N/states[t - 1]` | the observation before action `t` is the state step `t - 1` left behind |

Each episode ends up with exactly one image per action, and materialization refuses an episode whose action and state counts disagree (a truncated recording) rather than silently shifting every frame.

### What is copied and what is added

Every recorded channel — `actions`, `states`, `initial_state`, `ee_pose`, `robot_root_pose`, `bbox`, `obs/proprio_obs/*`, subtask status — is copied through with its values, dtypes, and episode attrs (`num_samples`, `success`, `seed`) unchanged. The tool only adds `obs/image_obs/<camera>` (and `obs/viewport_cam/<camera>` with `--record-viewport-camera`), plus `initial_state/cameras/<camera>` extrinsics, which a state-only recording could not capture because no camera existed.

The output is written through a temporary file and renamed only after every episode has been rendered, so an interrupted run leaves no half-rendered demonstration behind — and never overwrites an existing output unless `--overwrite` is given. The env config the images were rendered with is written next to the output as `env_cfg.json`, so the materialized file replays like any other recording.

### Environment reconstruction

The recording's `env_cfg.json` sidecar must sit next to the input. It is overlaid onto a freshly registered environment exactly as in [Replaying with the Recorded Env Config](#replaying-with-the-recorded-env-config---env-config), with one exception: a state-only config lists its policy cameras as `null` (that is how they were removed), and those entries are dropped before the overlay so the cameras registered for rendering survive. Everything else — object poses, physics parameters, terminations, instruction — comes from the recording.

Each episode is reset with its recorded `seed` before its states are restored, so seeded reset-time randomization (lighting, camera pose) reproduces the scene the attempt was recorded in.

### Fidelity limitations

- **Re-rendered, not recovered.** The images are a fresh render of the recorded states, not the pixels the live run would have produced. Rendering is not bit-exact across runs, drivers, GPUs, or simulator versions, and the tool prints the usual stack-mismatch notice when the recording came from a different IsaacSim/IsaacLab build.
- **Temporal accumulation is reproduced, not eliminated.** The RTX real-time renderer accumulates across frames, so a single render of a state is not fully converged — measured on an RTX 4090 (IsaacSim 5.0), re-rendering the *same* state after a large scene change still leaves ~27% of the image difference as ghosting from the previous frame. Live recording has exactly the same property (one render per control step), which is why the default is one render per frame: materialized frames carry the same accumulation history as live-recorded ones, since the states are walked in order. `--renders-per-frame N` converges each frame further, at N× the cost, but then the images are *cleaner* than a live recording's rather than equivalent to one.
- **Config values, not code or assets.** As with replay, changed predicate implementations or changed USD contents on disk are out of scope; only recorded config *values* are restored.
- **Only what the state captures.** Anything not in the recorded scene state and not reproduced by the seeded reset is not reconstructed — for example a scene mutated by an interval event mid-episode.
- **Rendering cost moves, it does not vanish.** Materialization renders every frame of every kept episode; budget it as an offline pass. Since it is not latency-bound, `--renderer pathtracing` and full camera resolution are affordable here even when teleoperation could not afford them.

### CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--input`, `-i` | required | State-only recording to materialize; its `env_cfg.json` must be alongside it. |
| `--output`, `-o` | required | Image demonstration to write. Must differ from the input. |
| `--episode` | all | Demo index to materialize; repeat for several. Demo names are preserved. |
| `--overwrite` | off | Replace an existing output (refused by default). |
| `--hdf5-compression {lzf,gzip,none}` | `gzip` | Compression for the rendered images; copied channels keep theirs. |
| `--camera-scale` | `1.0` | Scale camera width/height; `1.0` is the configured resolution. |
| `--task` | from the recording | Task class name, when the recording does not name one. |
| `--record-viewport-camera` | as recorded | Also render the third-person `viewport_cam` group. |
| `--renderer {realtime,pathtracing}` | `realtime` | RTX renderer for the offline pass. |
| `--renders-per-frame` | `1` | `sim.render()` calls per frame; `1` matches the live recording cadence. |
| `--gui` | off | Show the Isaac Sim window (rendering is offscreen by default). |
| `--device` | `cuda:0` | Simulation device (standard AppLauncher flag). |
