# Live XR collection latency experiments — 2026-10-27

Subsequent update: the user requested applying the combination to the collection
scripts on 2026-10-28. See the
[production integration and verification](omy_collection_latency_implementation_20261028.md).
The measurements and promotion recommendation below describe the preceding
diagnostic study.

Status: repeated full-combination bowl replay and a first manual L100 A/B are
complete. Two operator-confirmed full-combination processes show a small host
command-time improvement, and the operator reports less lag in the manual
candidate. Production defaults are unchanged; full physical L100-to-display and
grasp/release qualification remain pending. Blank-view and headset-sleep runs
are excluded from timing claims.

## Confirmed bowl results

All rows below use the normal XR collection loop and identical synthetic bowl
commands. A is the existing two-worker/eight-connection/GPU-check baseline. WCS
combines zero PhysX CPU workers, one CUDA connection and shared CPU snapshots.

| Run | Configuration | Mean elapsed command | p95 | Updates/s | Render-start interval p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| A4 | Baseline | 110.81 ms | 121.20 ms | 8.99 | 65.61 ms |
| WCS1 | Full combination | 108.42 ms | 119.03 ms | 9.19 | 63.30 ms |
| A5 retry | Return baseline | 111.35 ms | 121.27 ms | 8.95 | 66.20 ms |
| WCS2 retry | Full combination repeat | 109.06 ms | 119.74 ms | 9.14 | 63.70 ms |
| C3 | One CUDA connection only | 111.80 ms | 122.01 ms | 8.91 | 66.28 ms |
| W2 | Zero PhysX CPU workers only | 110.24 ms | 120.83 ms | 9.04 | 65.20 ms |

At matching episode/input-row positions, WCS improves mean elapsed command time
by 1.56–2.57% across these baseline/candidate comparisons. The matched A/A
difference is 0.50%; the WCS/WCS difference is 0.58%. This is a repeatable small
improvement in this screen, substantially less than the headless saving. It does
not reach the 15 Hz command target. There are only two fresh processes per
configuration; command rows within a process are not independent trial replicates.

Recorded states, actions, observations, dataset schemas and complete-episode
task events match the A4 reference. Both WCS blocks passed continuous headset
power checks, with 41 and 32 Awake checks and unchanged sleep history within
each block. All four measurement windows sampled a stable 1905 MHz GPU clock.
ALVR version, OpenVR configuration and session settings still match the initial
snapshot exactly; the configuration audit is retained with the artifacts.

ALVR still reports desynchronization recovery warnings and unstable pipeline
statistics. These runs do not establish headset presentation latency or a
physical motion-to-photon improvement. The existing bowl release failure also
prevents treating matching replay outputs as a physical-behavior qualification.

The operator-confirmed one-connection-only run retains two workers and the
existing GPU task/event path. At matching input positions it is 0.35–0.88%
slower than the two baselines. The earlier faster C2 screen therefore does not
establish a reliable live benefit from the driver setting alone. Its recorded
outputs still match. The confirmed zero-worker-only run retains eight connections
and the GPU task/event path. It improves matched means by 0.49–0.98%, compared
with the observed A/A difference of 0.50%. The full combination is another 1.05%
faster than this W2 run. The simpler individual settings have not established the
same benefit as the repeated full combination.

## Actual serial L100 comparison

The user then controlled the physical L100 in two fresh collection processes,
baseline followed by WCS. Each used 80 seconds of admitted commands; the summary
uses active seconds 20–80. Both headset images/tracking and the calibrated READY
pose were confirmed. The user was asked to use gentle free-space motion first,
then normal bowl manipulation, with F9 resets available. These are actual serial
recordings, not synthetic input. Human commands and contact phases differ, so
their timing difference is observational rather than a matched causal estimate.

| Metric | M1 baseline | M2 full combination |
| --- | ---: | ---: |
| Mean elapsed command | 116.75 ms | 115.19 ms |
| Command p95 | 132.40 ms | 125.98 ms |
| Command p99 | 137.41 ms | 128.99 ms |
| Updates/s | 8.52 | 8.64 |
| Application render-start interval p95 | 71.89 ms | 68.31 ms |
| Buffered source age at consumption p95 | 12.95 ms | 9.61 ms |
| Buffered source to environment-step return estimate p95 | 138.35 ms | 131.73 ms |

The last metric adds buffered sample age to consumption-to-step-return duration.
It does not include a verified encoder-acquisition timestamp or headset
presentation, and is not physical motion-to-photon latency. All 1,391 recorded
leader rows are valid. GPU clocks were 1905 MHz; headset power checks were Awake
throughout both processes (36 and 32 checks). ALVR desynchronization warnings
remain present and do not support declaring display delivery qualified.

Operator feedback is preserved verbatim in `operator_feedback.json`:

- Baseline control: "A little slower than before but acceptable"; grasp/release:
  "Grasp and release worked".
- Candidate compared with baseline: "Less lag or stutter"; grasp/release:
  "I did not complete a release".

M1 contains a successful 483-command attempt and a 206-command attempt interrupted
by the diagnostic time limit. M2 contains a 604-command manually reset attempt and
a 98-command attempt interrupted by the time limit, with no successful placement.
The user's report leaves candidate release unverified. Different human commands,
manual reset and the interrupted attempts do not establish a candidate-caused
success regression or comparable success rates. The pre-existing saved sticking
failure remains an independent baseline issue described below.

## Decision and remaining work

WCS remains the leading diagnostic candidate: a small repeated bowl replay gain
and encouraging first manual feedback. Retain existing production defaults for
now. Before promotion, complete:

1. Opposite-order manual A/B with completed candidate grasp, release, placement
   and F9/READY handling; the first manual pair is not a success qualification.
2. Both pail layouts and a small-object confirmation, using retained failures and
   comparable action/contact phases.
3. Comparable 5–10 minute baseline/candidate blocks to inspect queue buildup,
   tails and drift, plus usable display-delivery measurements.
4. Any further factor isolation needed to simplify the candidate: the WS screen
   was invalidated by headset sleep, and most individual/combination profiles
   have not received two operator-confirmed reversed-order processes.

The CPU-physics reference remains unqualified because support-contact preparation
failed; no CPU-versus-GPU live ranking is available. The right-side release
regression and left-only contact declaration also need resolution before claiming
reliable physical behavior. All experiment-only collectors and telemetry are
stopped at this block boundary; SteamVR/ALVR are left available. The operator was
told no headset is needed during analysis and will be notified before another
VR round.

## Actual live baseline

The operator-controlled bowl recording is
`output/omy_diagnostics/20261027_xr_l100_latency/A0_live_retry1/omy_demos.hdf5`.
It contains two successful attempts (155 and 123 commands) and a failed attempt
(622 commands). This run reached its 900-command diagnostic limit and closed;
the failed attempt was saved automatically. It did not use the new latency
prototypes.

Effective baseline: CUDA physics on the RTX 3090, rendering/SteamVR/ALVR on the
same GPU, TGS position cap 32, two GPU solver partitions, two PhysX CPU workers,
eight CUDA connections, original robot/colliders/mimics, 120 Hz physics, 15 Hz
commands, render interval four, shared placement checks and compact recording.
Native XR resolution multiplier is 0.8. CUDA/renderer enumeration was checked
against physical PCI bus IDs: CUDA 0 and renderer 0 both select the RTX 3090 at
`0000:65:00.0`; the RTX 3080 is CUDA 1 at `0000:B3:00.0`.

The three attempts average approximately 119 ms of command work, 8.3–8.4 command
updates/s and 16.6–16.8 application render calls/s. This application render rate
is distinct from the ALVR stream/headset refresh. The third attempt spends about
74.5 ms in physics, 25.0 ms in rendering, 8.75 ms in task checks and 5.27 ms in
recording per command. These are host regions including synchronization.

## Saved release failure

The user reported the bowl remaining attached after opening. The recording
already contains the failure; no additional operator demonstration is needed.
In `demo_2`, the command switches to open at row 87 and remains open for the
remaining 35.67 simulated seconds (63.68 wall-clock seconds). The driven finger
joint is within 0.001 rad of fully open by row 102. The bowl remains suspended;
its final root height is 0.26344 m.

A separate three-second continuation from saved post-state row 200, with the arm
held and the gripper commanded open, reproduces the failure at the same physical
settings. Per-finger PhysX contact data reports:

| Body | Substeps with contact | Largest reported overlap |
| --- | ---: | ---: |
| Left inner finger | 0/360 | None |
| Right inner finger | 360/360 | 2.658 mm |

The bowl finishes the continuation at approximately 0.26434 m root height.
Contact separation is the PhysX-reported quantity, not an exhaustive mesh
intersection measurement.

The current DROID `contact_gripper` declaration points only to the left inner
finger. Consequently, all 57 periodic detachment checks after the driven joint
opened report detached while right-side contact persists. This is a detection
blind spot; the contact declaration does not itself create the physical sticking.
Both-sided contact coverage and the right-side release regression need to be
addressed before treating the physical behavior as qualified. Solver cap 32 by
itself does not prevent this failure.

The latency comparisons retain the existing physical configuration. This failure
is a baseline regression fixture and must remain visible when assessing any
candidate; timing improvements do not certify release correctness.

## Measurement setup

The diagnostic wrapper runs the actual `examples/collect_omy_xr.py` collection
loop, including native XR, normal pacing, recording, preparation and episode
lifecycle. It retains full command and render-start timestamps in addition to
the existing region timings. Analysis excludes the first 20 seconds of active
control and uses the subsequent 60 seconds. Reset/READY intervals are retained
but excluded from active-command averages.

For repeatability, the screen replays the saved successful `demo_0` commands,
restoring its recorded initial state before each attempt. This is a synthetic
source through the collection loop, with automatic READY/reset control. It opens
no serial source and does not establish physical L100 response latency. Synthetic
recordings have invalid physical leader fields and explicit
`synthetic_replay=true`, `training_ready=false` markers. Raw controls remain
stored so phase matching can be checked.

Cold action replay does not reproduce all original hidden solver/contact state.
The first repeated-input baseline has four full failed replays and one partial
replay of the originally successful input. Those outcomes are retained; original
recorded success is not used to discard a failed replay or declare correctness.

An external observer records GPU telemetry and ALVR events. ALVR timestamps are
observer receipt times; reported pipeline statistics are not a camera measurement
of physical motion-to-photon latency. Renderer timings and stream refresh must
not be conflated.

The user reported headset power-off during `CS1_replay`. That run is explicitly
invalid for headset validation and excluded from analysis, even though its host
timing appeared much faster. SteamVR/ALVR still reported an active stream. The
batch was stopped. A gated retry admitted no commands because the headset image
was blank. Frozen CPU-initialized and default-eight-connection GPU-initialized
viewers were subsequently confirmed visible by the user, followed by the normal
collection handoff. A display-active/streaming flag alone
does not establish that the user has a valid headset image.

Headset power history places the sleep/wake interval at approximately
22:05:21.69–22:05:55.73 local time. It overlaps `WS1_replay` startup, so that run
is also excluded from headset validation: no image was confirmed after waking
before its automatic control began. The OS later reporting Awake does not prove
that the XR handoff recovered. The unusually fast WS/CS host measurements are
therefore not accepted as live improvements.

After recovery, the user confirmed the normal A4 collection view and the CS retry
view. Both were held at READY until confirmation. CS retry 2 (two workers, one
connection, CPU snapshot) measured 110.20 ms mean and 120.12 ms p95, only 0.61 ms
faster in mean than A4. Subsequent WCS and return-baseline measurements are in the
confirmed table above.

Later blocks used an external headset power guard alongside operator image
confirmation. The guard rejects sleep/history changes even if a stream still
appears active; it cannot replace visual confirmation. Experiment telemetry is
stopped between operator sessions and resumes by appending. The first reverse-
baseline launch was held by the power guard before any test commands; its retry
completed with 79 Awake checks and unchanged sleep history. The completed WCS1
block had 41 such checks.

The second operator-gated WCS process (`WCS2_verified`) was stopped by the power
guard after the headset slept while waiting for image confirmation. It admitted
no timed commands and is excluded. After the user returned, its fresh retry
(`WCS2_verified_retry1`) completed and produced the confirmed repeat above.

The CPU-physics/GPU-rendering reference reached XR but failed scene preparation:
table, bowl and bin support contacts were reported missing. No control commands
were admitted. It is excluded from latency ranking; this study has not established
a usable CPU-physics collection reference under the current preparation checks.

## Artifacts

- [Saved failure analysis](../output/omy_diagnostics/20261027_xr_l100_latency/saved_failure_analysis.json)
- [Per-finger continuation summary](../output/omy_diagnostics/20261027_xr_l100_latency/saved_failure_contact_detailed200/summary.json)
- [Current live timing analysis](../output/omy_diagnostics/20261027_xr_l100_latency/live_analysis.json)
- [Matched input-position comparisons](../output/omy_diagnostics/20261027_xr_l100_latency/confirmed_comparisons.json)
- [Recorded replay equivalence checks](../output/omy_diagnostics/20261027_xr_l100_latency/live_verification.json)
- [Actual serial L100 analysis](../output/omy_diagnostics/20261027_xr_l100_latency/manual_analysis.json)
- [Operator feedback](../output/omy_diagnostics/20261027_xr_l100_latency/operator_feedback.json)
- [Diagnostic collection wrapper](../output/omy_diagnostics/20261027_xr_l100_latency/live_entry.py)
- [Synthetic action-source adapter](../output/omy_diagnostics/20261027_xr_l100_latency/replay_source.py)
- [Full hardware validation plan](omy_xr_l100_latency_validation_plan.md)

Production defaults and source assets are unchanged by these diagnostic runs.
