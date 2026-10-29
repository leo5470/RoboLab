# Implemented OMY collection optimizations

Checkpoint before implementation: `e7752289`. This implements the CUDA-only
[measurement follow-up](omy_cuda_optimization_measurements_20261024.md).

## Collection behavior

New dedicated collection sessions now default to **CUDA on GPU 0, 32 position
iterations, two GPU solver partitions and shared placement checks**. The Bash
launcher retains 120 Hz physics, 15 Hz commands, render interval four, compact
recording and held-target stepping. The direct Python collector also defaults
to CUDA; its action/render cadence remains explicitly configurable.

```bash
./scripts/collect_omy_xr.sh SimplePnp291Bowl00Task --log
```

Shared placement checks reuse the same observed result for termination and
subtask consumers within an audited single-environment boundary. Full parameter
keys, boundary/reset invalidation and owned return values retain the existing
settling window and state-machine behavior. `--placement-checks independent`
provides the original evaluation behavior for comparison. General policy
evaluation and teleoperation remain on their existing configuration paths.

## Optional finger geometry

The original colliders remain the default. The experimental hull profile is
available through both collection entry points:

```bash
./scripts/collect_omy_xr.sh SimplePnp291Bowl00Task --finger-colliders hulls --log
```

It changes the four finger/pad collider approximations before cooking, then
restores the reference finger masses, COM transforms and full inertia tensors
before collection preparation or control. This avoids the unintended mass
increase observed when changing collider approximation alone. Default mass and
inertia buffers also receive the preserved properties, so retries retain them.
The bowl and other object collision geometry are retained.

The versioned [mass reference](../robolab/core/omy_teleop/robotiq_finger_mass.json)
records the existing model's properties and source/stack identity. The profile
checks the robot USD hash, Isaac Sim/Lab versions, expected collider count and
mass-property readback. It requires one CUDA environment and rejects scale or
mass overrides. It is a behavioral reference for this asset, not a physical
calibration. Different object shapes still need contact qualification.

## Recording and resume

Recording performance metadata includes requested/effective placement-check
behavior and the collider profile/reference identity. Session contracts include
both new settings, and hull sessions include the reference fingerprint.
Conflicting explicit resume flags or changed reference data are rejected.

Sessions created before these options preserve their original contract and
effective independent placement evaluation/original colliders. Older sessions
without a partition setting retain eight partitions. Saved solver caps,
including an unset optional cap, are restored rather than replaced by new
collection defaults. Neither the saved recordings nor their metadata are
rewritten during compatibility checks.

## Production-path verification

The updated replay harness exercises the actual collection configuration and
environment hooks; it does not install the diagnostic monkeypatch for these
profiles. Sixteen CUDA replays used the source bowl attempt, 32 iterations, two
partitions, two physics workers and unchanged action/physics rates.

| Production path | Repeats | Mean work | p95 |
| --- | ---: | ---: | ---: |
| Independent placement checks, original geometry | 4 | 75.56 ms | 83.29 ms |
| Shared placement checks, original geometry | 4 | **74.19 ms** | **81.45 ms** |
| Shared checks and mass-preserving finger hulls | 4 | **72.68 ms** | **78.33 ms** |

Timing uses exactly source rows 20–99 and excludes rendering/XR. Four additional
contact-instrumented hull runs are excluded from the speed table. All sixteen
runs place successfully. The hull contact runs show worst reported overlap
**0.878 mm**, with no persistent attachment after opening; last contact is at
most **0.033 simulated seconds** after the open command.

The cache preserves all recorded action/observation/state/subtask values across
four matched episodes (16 dataset leaves per episode), plus task events and
termination timing. Runtime finger mass/COM/inertia reads match the versioned
reference. These are repeated replays of one recorded grasp, not independent
operator trials or a collection-wide success estimate. Live XR timing still
needs measurement.

Tests cover the production cache's timer/reset behavior, scalar/batch consumers,
session compatibility and conflicts, CUDA default, reference identity, asset
and version mismatch rejection, mass overrides, non-finger property retention
and default buffer updates. Existing collection launcher and evaluation
isolation checks are included in regression validation. The
[verification record](../output/omy_diagnostics/20261024_implementation/verification.json)
contains test commands and results; the
[machine-readable replay analysis](../output/omy_diagnostics/20261024_implementation/analysis.json)
contains timing, provenance, equivalence and release measurements. No leader or
headset was opened by these checks.
