# Demonstration video library

The existing RoboLab dashboard includes `/demonstrations`: a task-by-task viewer
for collected recordings and **selected** MimicGen demonstrations. Generated
candidates enter the catalog only through a completed `selection.json`.

## Start the website and automatic video worker

From the repository root, use two terminals:

```bash
.venv/bin/python scripts/serve_demo_library.py --port 8081
.venv/bin/python scripts/serve_demo_library.py --worker --gpus 0 1
```

Open http://127.0.0.1:8081/demonstrations. For remote access, forward the port:

```bash
ssh -N -L 8081:127.0.0.1:8081 USER@ROBOLAB_HOST
```

Then open the same URL on your computer. The server binds to loopback by
default. Pass `--host 0.0.0.0` for an intentionally shared lab-network service.

The regular `robolab-dashboard` command also exposes the library with default
roots. Its existing evaluation, task and scene pages remain available.

## Discovery and playback

Defaults are `output/omy_teleop`, `output/omy_mimicgen`, and runtime state under
`output/demo_library`. Override roots with repeated `--collected-root` and
`--generated-root` arguments. Both processes must use the same roots and
`--state-dir`. `--refresh-seconds` defaults to 30. New tasks, sessions and
selections are discovered without a restart and without date-prefix rules.

The catalog caches manifest and HDF5 metadata by modification time and size.
It keeps the last complete cached version of an artifact while an update is
incomplete. Raw collected episodes are visible even without videos; available
postprocessing manifests provide recorded outcomes and video links. Historical
collections are grouped by task rather than their archive-directory name.

The browser provides task navigation, collected/generated filters, layout,
session, outcome, availability and review filters, search, pagination, shareable
episode URLs, speed and loop controls, and source comparison. Seeking is served
using HTTP byte ranges. Camera videos retain the recorded 15 FPS timing.

Recorded success, diversity selection and contact-review status are distinct.
The wooden-bowl contact-review hold stays visible. Rendering is for inspection;
it does not certify contact physics or training readiness.

## Selected-video automation

`scripts/run_omy_mimic_collection.py` automatically invokes the video service
after finalizing its selection. An already-running worker discovers that
selection on its next reconciliation. The persistent worker also backfills
historical selections. Selection publication is atomic.

The worker admits exactly the selected episodes, reuses matching individual
previews, restores recorded states for missing policy cameras, encodes separate
MP4s and verifies decoding, frame count and timing before publishing them.
Rendering uses the configured camera resolutions without a size override or
downscaling: currently 1280×720 per camera, producing a 2560×792 side-by-side
MP4 including the banner. Rendering settings are part of queue/cache identity;
old 320×180 previews stay playable while native-resolution replacements render,
but do not satisfy completion. Low-resolution camera caches are not upscaled.
The video worker writes uncompressed intermediate HDF5 images to avoid gzip
compression/decompression in the rendering and encoding loops. Pixel values
and MP4 encoding settings are unchanged; intermediate files use more disk space.
Other postprocessing callers retain their default gzip storage.
Both GPUs are supported, using the same exclusive GPU locks as generation.
Workers release ownership between bounded batches so waiting generation jobs
can run. The website itself needs no GPU and imports no simulator.

Render a specific run once, rather than watching for future selections:

```bash
.venv/bin/python scripts/serve_demo_library.py --worker --once \
  --run output/omy_mimicgen/RUN --gpus 0 1
```

Use a completed selection for the run. If a persistent worker already owns the
service lock, this command leaves rendering to that worker. Use that run's
`video_status.json` to observe completion.

For a custom runtime state directory, set `video_state_dir` in the MimicGen
`run_config.json` to the same absolute path passed to the website and worker.

## Runtime state and recovery

```text
output/demo_library/
  manifest_cache.json       # incremental metadata cache
  queue.sqlite3            # durable admission, attempts and job states
  worker.lock              # one queue service across all GPUs
  renders/                 # reusable camera caches and failure evidence
  workers/                 # isolated simulator worker logs/acknowledgments
  videos/EPISODE_ID/
    video.mp4              # individual episode video
    poster.jpg
    ready.json             # verified stream and recording identity
```

Each generated run receives `video_status.json`, with selected, ready, pending,
running and failed counts. Video completion is reported separately from
generation/selection completion. The website displays pending/rendering/failure
states while usable videos remain playable.

The queue retries a failed job up to three attempts and continues other jobs.
Interrupted running jobs return to pending after the next worker acquires the
exclusive service lock. SIGINT/SIGTERM finish the current rendering operation,
close the renderer, and stop claiming work. Partial rendering output is retained
with diagnostics and never advertised as ready. A changed source/configuration
invalidates its queued result. Deselected episodes leave the active catalog;
their cached artifacts are retained.

Stop services using Ctrl+C in their terminals, or SIGTERM for background
processes. After an abrupt worker kill, also stop any surviving isolated Isaac
workers before restarting, so the shared GPU locks can be released.

## Validation

The integration tests cover discovery after startup, selected-only admission,
last-complete manifest handling, deselection, interruption recovery, bounded
retries, video byte ranges and root restrictions. Existing rendering and
diversity tests cover the reused pipeline components:

```bash
.venv/bin/python -m pytest --import-mode=importlib -q tests/test_demo_library.py \
  tests/test_omy_postprocess_videos.py tests/unit/test_omy_diversity.py \
  tests/unit/test_omy_mimicgen.py tests/unit/test_omy_postprocess_render.py \
  tests/unit/test_omy_postprocess_batch.py
```
