# Keyboard Teleoperation and Demo Collection

Use [`examples/collect_keyboard_demos.py`](../examples/collect_keyboard_demos.py) to control a single DROID environment with relative end-effector commands and record demonstrations to HDF5. Its default `wifi` profile defers policy-camera images during remote collection; a local Isaac Sim window is also supported.

## Start the Collector

Run the collector in `tmux` so it remains alive if the SSH connection drops:

```bash
tmux new -s robolab-stream
cd /tmp2/leocheng/forks/RoboLab
source .venv/bin/activate

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMNI_KIT_ACCEPT_EULA=Y \
python examples/collect_keyboard_demos.py \
  --task BananaInBowlTask \
  --num-demos 10 \
  --output-dir output/keyboard_demos/banana_wifi \
  --device cpu \
  --livestream 2 \
  --teleop-profile wifi \
  --defer-images \
  --rendering_mode performance
```

`--num-demos` counts successful demonstrations, not attempts. The `wifi` profile defers images automatically when livestreaming. `--defer-images` above makes that explicit. After collection, run the [offline image materializer](#deferred-image-collection---defer-images). Keep `env_cfg.json` beside the recording.

Use **`--device cpu` for this single-environment collector on this server**. The viewport still renders and encodes on the GPU. CPU PhysX avoids GPU submission/readback overhead for this small scene. The timestep, eight physics substeps per action, solver settings, and success predicates are unchanged. CPU/GPU trajectories need not be bit-identical; deferred rendering restores recorded states rather than replaying actions. Explicit `--device cuda:0` remains supported; the launcher device default is still CUDA.

The Wi-Fi profile fixes the stream at 960×540, disables blocking frame submission, removes Kit's extra render throttle, and samples current contact forces lazily instead of maintaining unused histories. Actions are paced at the task's real-time rate (15 Hz for DROID), with input polling during spare time. The viewport renders once per action. `--contact-history` restores history buffers for custom consumers that need them.

For `--device cpu`, the Wi-Fi profile also limits Kit/TBB to 16 workers and CPU PhysX to 2 workers (capped by CPU affinity). Tensor operations still use one PyTorch thread. This reduces worker coordination overhead for one small environment; it does **not** lower physics accuracy, solver iterations, or the 120 Hz physics / 15 Hz control rates. Startup reports the actual worker settings. GPU and `standard` profiles keep their existing worker settings unless explicitly overridden.

Use `--kit-threads N --physics-threads N` to tune other hardware; `0` leaves that setting untouched. For comparison with this server's original worker counts, use `--kit-threads 32 --physics-threads 8`. Advanced `--kit_args` overrides remain last. When worker tuning is active, Kit preference saving is disabled for this process so its `/persistent/physics/numThreads` setting cannot silently affect the next application. Existing preferences are still loaded. Restart the collector to apply changes.

To record images live, add `--no-defer-images`. Policy cameras then default to 640×360 with uncompressed HDF5 flushed every 25 actions. `--camera-scale 1` restores full sensor resolution; `--record-viewport-camera` adds a third-person sensor. These sensor settings are independent of stream resolution. `--teleop-profile standard` retains the original image/stream defaults for comparison. `--hdf5-compression lzf` trades some responsiveness for smaller image files.

For less Wi-Fi traffic, add:

```bash
--stream-width 640 --stream-height 360
```

Use `--stream-width 1280 --stream-height 720` for a sharper stream. These flags configure capture and viewport size together. After environment creation, the collector also pins the actual viewport texture size: Kit can restore a saved 1280×720 texture independently of the window size and viewer configuration. The startup log reports `Operator viewport render texture: (960, 540)` for the default Wi-Fi settings.

Do not connect the WebRTC client as soon as `Streaming server started` appears. Wait until the terminal prints:

```text
Attempt 1; collected 0/10. READY — physics is paused. Click the streamed viewport and press N to start.
```

Connecting while the USD stage and RTX renderer are still being rebuilt can produce a temporary image followed by a blank screen. If that happens, close or refresh the client and reconnect after the `Attempt` message appears.

Connect the client directly to `192.168.11.43` (this server's current internal address). The installed stack uses TCP `49100` for signaling and a media port selected from `47998`–`48020`. Check the negotiated UDP path if video remains laggy despite a fast control loop.

## Two Operator Views in One Stream

Add `--view-layout inset` to display the normal overview plus a wrist-camera view that follows the gripper:

```bash
python examples/collect_keyboard_demos.py \
  --task BananaInBowlTask --device cpu --livestream 2 \
  --teleop-profile wifi --defer-images --view-layout inset --inset-camera wrist \
  --num-demos 10 --output-dir output/keyboard_demos/banana_wifi
```

The stream stays **960×540 total**, with a **320×180 wrist inset** in the upper-right corner. `wrist` is the default, so an existing command with `--view-layout inset` uses it after restarting. The private operator camera uses the DROID wrist camera's configured mounting offset and intrinsics and follows the gripper. It does not reuse the differently calibrated `wrist_camera` baked into the robot asset, or enable the `wrist_cam` policy sensor.

The Wi-Fi stream hides editor panels so the views fill the window. The main overview retains its normal mouse navigation; the inset follows its mount and cannot be repositioned with mouse gestures. For editor menus/panels, override with `--kit_args '--/app/window/hideUi=false'`.

- Press **`B`** in the streamed viewport to hide/show the inset, including while waiting for `N`. Hiding it disables its rendering, reclaiming that work without restarting or rejecting an attempt.
- Use `--inset-camera top` to restore the fixed top-down view, centered on the configured viewer target from 1.5 m above it.
- Use `--view-layout single` (the default) to start without an inset.
- This is a GPU-rendered operator view, not a policy-camera sensor. It performs no image readback or HDF5 image writes and does not alter the viewpoints used for deferred image materialization.
- Both views update on the existing render cadence. There is one WebRTC connection and no increase in stream dimensions. Encoding bandwidth can still vary with image content, and a second camera adds rendering work; it is not free.
- With custom stream sizes, the inset uses approximately one third of each dimension, rounded down to even pixels. Inset mode requires at least 240×180.

Restart the collector to load the new option; an already running process is unchanged. Do not launch a second collector on the same streaming port.

### Two-view validation (2026-09-11)

Matched runs on the same RTX 4090, CPU physics, `BananaInBowlTask`, deferred images, performance renderer, and a verified 960×540 viewport texture. Each unpaced run measured 300 actions after 40 warmup actions; editor panels were hidden in both runs.

| Layout | Unpaced steps/s | Step time p95 |
|---|---:|---:|
| Single overview | 31.5 | 44.3 ms |
| Overview + 320×180 top-down inset | 27.4 | 51.1 ms |

The top-down two-view paced test maintained **15.0 steps/s over 150 actions**, with a 56.9 ms p95 step time against a 66.7 ms control period. A real-render smoke test also checked the 960×540 composite, independent cameras, `B` press/release, reset survival, arm movement, cleanup, and that no synthetic demos were saved. Fifteen focused unit tests passed at that revision. These measurements predate the wrist inset; use `--inset-camera top` to reproduce that layout.

The wrist inset was subsequently checked with 150 measured actions after 30 warmup actions: **33.7 unpaced steps/s** (31.3 ms p95), and **15.0 paced steps/s** over another 150 actions (56.4 ms p95). The rendered camera position matched the gripper pose plus the calibrated mounting offset within 1 mm after motion and reset. The image was visually checked, no policy-camera sensor was created, and no synthetic demonstrations were saved. Nineteen focused unit tests passed. This was a separate run, not a controlled speed comparison against the top-down view.

These are **server-side measurements without a connected WebRTC client**, not end-to-end Wi-Fi latency measurements. For a local comparison, run each layout with `--benchmark-steps 300 --benchmark-warmup 40` and separate output directories. Benchmark mode discards its attempts.

## Deferred Image Collection (`--defer-images`)

`--defer-images` removes policy-camera rendering and image writes from the live loop and reconstructs images later from recorded scene states. The viewport still renders for the operator. Physics and Kit pacing also matter; see the measurements below.

Collect state-only:

```bash
python examples/collect_keyboard_demos.py \
  --task BananaInBowlTask \
  --defer-images \
  --num-demos 10 \
  --output-dir output/keyboard_demos/state_only \
  --device cpu \
  --livestream 2 \
  --rendering_mode performance
```

Then materialize the images offline:

```bash
python scripts/materialize_demo_images.py \
  --input output/keyboard_demos/state_only/keyboard_demos.hdf5 \
  --output output/keyboard_demos/final/keyboard_demos.hdf5 \
  --device cuda:0 \
  --hdf5-compression gzip
```

The result is an ordinary RoboLab image demonstration: `obs/image_obs/<camera>` at the same paths, shapes, and dtypes a live image recording produces.

The WebRTC/local viewport keeps working unchanged. It draws the USD stage through the Kit viewport and never reads a policy camera, so teleoperation looks the same — only the sensors that feed `obs/image_obs` are gone.

### What the state-only file holds

| Recorded | Not recorded |
|---|---|
| `actions` — the relative-IK commands | `obs/image_obs/*` — policy camera images |
| `states` — full post-step scene state, every step | `obs/viewport_cam/*` — third-person sensor images |
| `initial_state` — the scene the attempt started from | camera extrinsics under `initial_state/cameras` (no camera exists yet) |
| `obs/proprio_obs/*` — joint, gripper, and EE observations | |
| `ee_pose`, `robot_root_pose`, `bbox`, subtask status | |
| `seed` and `success` on each demo | |
| `env_cfg.json` sidecar, plus control-timing metadata on the HDF5 `data` group | |

Proprioception is kept: it is computed from articulation state and is inexpensive compared with image capture.

### Buying back control rate

After policy cameras are removed, physics execution, sensor reads, viewport rendering, and frame pacing still take time. The optimized profile reduces this overhead while retaining the task's simulation timestep and solver settings. The choice of CPU/GPU backend also matters for one environment.

`--render-interval` counts physics substeps between viewport updates. Keep the task default of `8` (one render per action) as a starting point. Larger values make visual feedback less frequent and may delay input delivery by Kit extensions:

```bash
--render-interval 16   # optional tradeoff: only 7.5 Hz feedback at real time
```

Deferred materialization renders each recorded state explicitly afterwards. With live images, skipped renders can produce repeated camera observations; the collector warns about this combination. Buffered keyboard input is drained before each action, but a remote input provider may still depend on Kit updates.

The flag only matters when something is actually being displayed (`--livestream`, or a local window). A headless run with no policy cameras renders nothing at all.

Two collector defaults change in this mode:

- **Flush interval** defaults to `0` (write once per attempt). Growing tensor storage avoids repeatedly copying the history. Override with `--flush-interval N` for periodic writes; data since the last flush can be lost if the process crashes.
- **`--camera-scale` is inert**, because there is no policy camera to scale. Choose the resolution at materialization time instead (`--camera-scale` on `materialize_demo_images.py`, which defaults to the full configured resolution).

`--record-viewport-camera` still works with `--defer-images`, but it creates a sensor that renders every control step — it gives back part of the latency you deferred. The collector prints a warning when both are used.

### Storage

State-only recordings are roughly three orders of magnitude smaller than image recordings — a few hundred KB per episode instead of a few GB — so a long collection session fits comfortably on disk, and only the demonstrations you keep pay for image storage. Budget for the materialized file separately: one 1280×720 RGB camera is ~2.7 MB per frame uncompressed, so a 500-step episode with two cameras is ~2.7 GB raw, or a few hundred MB with `--hdf5-compression gzip`.

### Frame alignment

RoboLab records observations *pre-step*, so observation `t` is the scene before action `t` is applied. Materialization reproduces exactly that: frame 0 is rendered from `initial_state`, and frame `t` from post-step state `t - 1`. An image demonstration produced this way indexes identically to a live-recorded one.

### Limitations

- The images are rendered from the recorded states, not re-simulated, so they track the trajectory exactly — but they are a *re-render*, not the same pixels the live run would have produced. Rendering is not bit-exact across runs, drivers, or simulator versions.
- Materialization needs the `env_cfg.json` sidecar next to the recording, and the same task, scene, and asset definitions the recording was made with. See [Replaying Recorded Episodes](replay.md) for what a recorded config can and cannot restore.
- Reset-time randomization (lighting, camera pose) is reproduced by resetting with the recorded per-episode seed before the states are restored. Effects that are not part of the seeded reset or the recorded state — a task whose scene changes on an interval event, for instance — are not reconstructed.

## Keyboard Controls

Click inside the streamed viewport before using the keyboard.

| Key | Command |
|---|---|
| `N` | Start the currently armed attempt |
| `W` / `S` | Move along positive / negative X |
| `A` / `D` | Move along positive / negative Y |
| `Q` / `E` | Move along positive / negative Z |
| `Z` / `X` | Rotate around positive / negative X |
| `T` / `G` | Rotate around positive / negative Y |
| `C` / `V` | Rotate around positive / negative Z |
| `K` | Toggle the gripper between open and closed |
| `L` | Clear held motion commands if an input becomes stuck |
| `R` | Reject the current attempt and reset the scene |
| `B` | Hide/show the selected camera inset when started with `--view-layout inset` |

Hold a movement key for continuous motion and release it to stop. Multiple movement keys may be held together. Do not use `Space` to start an attempt: Omniverse also binds it to timeline play/pause.

The seven recorded action values are:

```text
[dx, dy, dz, rx, ry, rz, gripper]
```

The gripper value stored by RoboLab is `0` for open and `1` for closed.

## Attempt Lifecycle

1. Wait for the terminal to print the `Attempt` message.
2. Click the streamed viewport and press `N` once.
3. Control the arm and complete the task.
4. On success, RoboLab finalizes the demonstration automatically.
5. On timeout, the attempt is marked failed and the environment resets.
6. After every reset, click the viewport and press `N` again before using movement keys.

By default, failed, timed-out, and manually rejected attempts are discarded. Add `--save-failures` to write them to a separate HDF5 file.

Press `Ctrl+C` in the server's tmux pane for a clean shutdown. This closes the active HDF5 handle before Isaac Sim exits.

To leave the collector running and detach from tmux, press `Ctrl+B`, then `D`. Reattach with:

```bash
tmux attach -t robolab-stream
```

## Output

With the example command, successful demonstrations are written to:

```text
output/keyboard_demos/banana/keyboard_demos.hdf5
```

The same directory also contains `env_cfg.json`. If the requested HDF5 filename already exists, the collector chooses a numbered filename such as `keyboard_demos_1.hdf5` rather than overwriting it.

Both files matter for a state-only recording: `materialize_demo_images.py` reads `env_cfg.json` from the same directory to rebuild the scene, so keep them together.

## Troubleshooting

### The scene stops after an attempt times out

After success, timeout, or `R`, the collector resets the scene and waits for a fresh `N` in the streamed viewport. Movement keys alone do not start the next recording. The terminal prints `READY` while waiting and `RECORDING` once `N` is received. Repeated READY messages mean the collector is responsive but has not received a start command.

Click the viewport, release held keys, and press `N` once. Pressing `R` while READY is ignored; it cannot reject the next recording. If the video is frozen too, reconnect the client and press `N` after refocusing the viewport.

### The scene appeared and then went blank

The client probably connected before the scene finished loading. Confirm that the Python process is still running and TCP port `49100` is listening, then reconnect after the terminal shows the `Attempt` message.

```bash
ps -ef | rg collect_keyboard_demos
ss -lntp | rg 49100
```

### `Failed to startup plugin carb.windowing-glfw.plugin`

GLFW is the local X11 window backend. Its startup warning is expected when Isaac Sim runs headlessly through WebRTC without `DISPLAY`. It is not fatal by itself.

If this warning appears for each key press **and the arm does not move**, inspect the tmux output. The previous attempt may have timed out, leaving the collector at the next `Attempt` prompt. Click the viewport and press `N` again. If `N` is not received, disconnect and reconnect the WebRTC client, refocus the viewport, and retry.

### Movement is delayed

The terminal reports the control rate, step-time distribution, physics/render/other work, and WebRTC QoS when a client supplies it. An illustrative line is:

```text
Control rate: 15.0/15.0 steps/s; step p50/p95 34/58 ms; physics 7 ms; render 10 ms; IK/record/other 17 ms; max 61 ms; over budget 0/75; WebRTC RTT unavailable, bitrate target 12.5 Mbps
```

The first rate includes real-time pacing. Step timings exclude that intentional wait; region values are mean wall-clock milliseconds **per control action**. `physics` sums the simulator calls across the eight substeps, excluding nested renders. It is not a GPU-kernel measurement. `IK/record/other` includes control, scene updates, sensors, predicates, and recording. `over budget` counts actions exceeding the task period (66.7 ms at 15 Hz); `max` exposes rare stalls that a median hides.

WebRTC RTT is separate from simulation speed. `awaiting client QoS` means no recent feedback; `RTT unavailable` means feedback did not supply a positive RTT. Neither means zero latency. Bitrate is an encoder target, not measured network throughput. A reported `requested size` is a QoS recommendation for a future frame, not the current encoded resolution; missing `0×0` dimensions are omitted.

If the rate stays near 15 but viewing/control still feels delayed, check Wi-Fi and the client's decode/display path. Ping during streaming, not only while idle:

```bash
ss -tnp | rg 49100       # the peer address is your streaming client
ping -c 20 <client-ip>   # or, from the client, ping the server
```

From the client, `ping 192.168.11.43` is the equivalent check. If RTT spikes under stream traffic, try 640×360, move closer to the access point, use its 5/6 GHz band if available, or compare with Ethernet. Clean ICMP alone does not rule out media buffering or slow client decoding.

If the control rate itself is low, check that the launch uses `--device cpu --teleop-profile wifi --defer-images`, and that an old `--flush-interval 10 --hdf5-compression lzf` override has not survived in your command. Avoid sharing the rendering GPU with a heavy training or evaluation job. The collector does not change system-wide CPU governors.

### Low GPU utilization and occasional physics lag

Low GPU utilization is normal here: CPU PhysX advances one small environment, while the GPU renders a small viewport at a paced 15 Hz. Moving physics to CUDA solely to raise utilization can make this workload slower (see the CPU/CUDA comparison below).

- Rising **physics** time points toward simulation execution or CPU scheduling. Compare the worker counts above without changing timestep or solver settings.
- Rising **render** time points toward viewport/renderer/stream work. Press `B` to disable the wrist inset temporarily, or compare a single view.
- Rising **IK/record/other** time points toward control, sensors, task logic, recorder work, or scheduling. State-only recording already avoids periodic image compression and writes.
- Stable 15 Hz and few over-budget actions, but delayed video, points toward the stream/client path rather than simulation throughput. The server logs cannot measure end-to-end input-to-display latency.

This host was using the `powersave` CPU governor during profiling. That is a reason to investigate CPU frequency behavior, not proof of throttling. The collector does not modify governors, GPU clocks, or system services. See NVIDIA's [Isaac Sim performance guidance](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/reference_material/sim_performance_optimization_handbook.html) for worker-count and CPU governor considerations.

#### Worker tuning validation (2026-09-11)

Sequential runs on the same RTX 4090 with CPU physics, the 960×540 overview plus 320×180 wrist inset, deferred images, and the same seed/actions. Each mode measured 290 actions across free motion, descent/contact-hold, and raising phases after 30 warmup actions; no cProfile or connected streaming client. Kit/TBB/PhysX counts were explicitly verified, and persistent preference saving was disabled.

| Workers (Kit/TBB/PhysX) | Unpaced capacity | Mean step at 15 Hz pacing | Mean physics at 15 Hz pacing | Largest paced step | Over-budget paced steps |
|---|---:|---:|---:|---:|---:|
| Original 32/32/8 | 29.5 steps/s | 56.1 ms | 7.94 ms | 63.2 ms | 0/290 |
| Tuned 16/16/2 | 30.9 steps/s | 53.9 ms | 7.02 ms | 60.4 ms | 0/290 |

This is roughly 5% more unpaced capacity, not a large throughput change. Both maintained about 15 Hz; neither reproduced the occasional >100 ms non-render stalls observed during a connected collection session. The new live timing breakdown is needed to investigate those stalls under actual keyboard/network traffic. These results do not establish an end-to-end Wi-Fi latency improvement.

The final collector's `--benchmark-realtime` smoke test separately completed 150 measured neutral actions after 30 warmup actions: 14.96 steps/s, 53.4 ms p95, 57.6 ms maximum, and zero over-budget actions. Both wrist and overview rendering were enabled. The benchmark files contained no saved synthetic demos, the original saved PhysX preference remained unchanged, and 29 focused unit tests passed.

### Measured server performance

On 2026-09-10, `BananaInBowlTask` on this server with an idle RTX 4090, WebRTC enabled but **no connected client**, and the same 120 Hz physics / 15 Hz action timing:

| Configuration | Unpaced steps/s | Median step | p95 step |
|---|---:|---:|---:|
| Previous pipeline, CUDA, live images | 5.73 | 167 ms | 174 ms |
| Previous pipeline, CUDA, deferred images | 6.09 | 164 ms | 169 ms |
| Previous pipeline, CPU, deferred images | 10.36 | 96 ms | 99 ms |
| Wi-Fi profile, CUDA, deferred images | 11.37 | 88 ms | 91 ms |
| Wi-Fi profile, CPU, deferred images | 33.06 | 30 ms | 33 ms |

Each measurement excluded 20 warmup steps, then recorded 100 steps (150 for optimized CPU), with cProfile enabled. These measure server capacity, not end-to-end Wi-Fi latency. Interactive collection is capped at 15 steps/s to keep action speed and episode duration consistent; it does not execute actions at the unpaced benchmark rate.

The profiler found Kit sleeping inside each render: disabling its frame limiter after environment creation reduced the render call from about 67 ms to 10 ms. Lazy current-force reads removed most sensor-update overhead. CPU physics further removed the small-scene CUDA overhead. No physics timestep, action decimation, collision geometry, or solver iteration counts were reduced.

Repeat a finite benchmark in a separate output directory:

```bash
python examples/collect_keyboard_demos.py \
  --task BananaInBowlTask --device cpu --livestream 2 \
  --teleop-profile wifi --defer-images \
  --benchmark-steps 150 --benchmark-warmup 20 --profile \
  --output-dir output/teleop_benchmark/cpu_wifi
```

This bypasses `N`, issues neutral actions, discards the attempt, and writes `benchmark.json` and `benchmark.prof`. It does not save a synthetic successful demo. For the comparison path, use `--teleop-profile standard --device cuda:0 --no-defer-images`. Connect a client during a longer benchmark if you also want to include stream encoding/network load. Inspect the profile with `python -m pstats <output-dir>/benchmark.prof`.

To test pacing and deadline misses with the wrist view, use a separate run/output directory:

```bash
python examples/collect_keyboard_demos.py \
  --task BananaInBowlTask --device cpu --livestream 2 \
  --teleop-profile wifi --defer-images --view-layout inset \
  --benchmark-steps 300 --benchmark-warmup 40 --benchmark-realtime \
  --output-dir output/teleop_benchmark/wrist_paced
```

`wall_steps_per_second` includes pacing; `steps_per_second` measures capacity from step execution time alone. JSON also includes physics/render/other means, p95/max step time, and `over_budget_steps`. Default benchmark mode is unpaced regardless of the interactive `--realtime` setting. Do not run a benchmark alongside collection on the same streaming port. Neutral-action benchmarks do not cover every contact configuration or reproduce human keyboard/network traffic.

### CUDA out of memory while recording

The default `--flush-interval 25` keeps each image batch bounded. If memory still grows, lower it to `10`. Use `--flush-memory-cleanup` only as an OOM workaround because forced garbage collection and CUDA-cache clearing introduce periodic control stalls.

For a quick teleoperation test without recording camera images:

```bash
python examples/collect_keyboard_demos.py \
  --task BananaInBowlTask \
  --num-demos 1 \
  --no-record-images \
  --device cuda:0 \
  --livestream 2 \
  --rendering_mode performance
```

### A movement key remains active

Press `L` to clear held motion while preserving the gripper's open/closed state. The controller ignores duplicate presses and unmatched releases (including releases after reset), and `K`/`N`/`R` fire once per press. If a key-release event is lost on the network, refocus the viewport and press `L`.
