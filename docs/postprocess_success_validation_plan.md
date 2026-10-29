# Postprocessing success validation — implementation plan

Status: proposed implementation; no success-validation behavior has been added.

## Outcome and scope

Add a default verification stage for recorded-success OMY demonstrations. Record
whether the reconstructed terminal state still satisfies the task and remains
successful during a short physics continuation. Keep the original collection
outcome and a separate postprocessing verdict: `passed`, `failed`, `inconclusive`,
or `not_run`, with explicit reasons.

Start with Simple-PnP tasks using `object_resting_in_container`. Use the recorded
success function and parameters. Additional goal types require supported adapters
that declare their contact, history, and stability requirements. Unsupported
goals and unavailable historical dependencies produce `inconclusive`, not a pass.
A reconstructed continuation verifies the loaded task/profile; it cannot recover
the original simulator's unrecorded contact/solver history or prove that the
historical success label was correct.

### Physics-profile compatibility and grasp quality

Keep terminal continuation distinct from full recorded-action replay. A terminal
check cannot detect a grasp lost earlier in transport. The
[headless articulation experiments](omy_articulation_latency_experiments_20261027.md)
include a replay that dropped a box about half a second before the commanded
opening, then landed in the bin and satisfied the terminal goal. Preserve terminal
task success and trajectory-quality findings as separate report fields.

New collection sessions already use TGS, a 32 position-iteration cap, two GPU
partitions, 120 Hz physics, 15 Hz commands and original finger colliders. The
existing camera materializer restores recorded states without advancing physics;
it does not replay actions under a separate stronger solver profile. Preserve the
recorded profile for ordinary terminal continuation. If compatibility with a
different profile is requested, declare that target explicitly and replay the
complete recorded action sequence from its initial state. Do not infer that a
larger iteration count guarantees success. Older sessions can retain 16/8, so an
explicit 32/2 compatibility replay has a different purpose from rendering them.

For compatibility replay, identify both recorded and target physics: solver type,
iteration bounds, partitions, timestep, command cadence, collider identity and
mass properties, actuator gains/limits, and mimic gearing/compliance. Runtime
overrides need reproducible application before simulation creation; a skipped
callable or an authored but unsupported USD property cannot establish the target
configuration. Save scheduling settings separately in execution provenance.

Observe both inner fingers, contact separation, object motion relative to the
gripper, bin support and requested opening during the replay. Flag unexpected
grasp loss while closing is still commanded, including loss followed by a passing
terminal goal. Use task-specific grasp-quality rules so an intentionally released
object is distinguished from an unrequested drop. Retain endpoint versus later
settling success separately, and report pose drift alongside reported velocities;
these experiments also found nearly stationary poses with velocity spikes that
prevented the original settling predicate from passing. Do not silently relax the
task thresholds to rescue that case.

A replay under the recorded profile is useful on a calibration subset and when
investigating failed target-profile replays; it is not required for every episode
merely to duplicate the recorded success label. Initial-state restoration still
omits contact/solver caches, so record reconstruction sensitivity explicitly.

## 1. Define the report and evidence contract

Add `robolab/core/omy_postprocess/success.py` for the simulator-independent
configuration, verdict rules, provenance, report validation, and cache handling.
Write `success_validation.json` and a compact state/contact/predicate trace per
episode. Include:

- Original outcome, attempt identity, source hash, and terminal state row.
- Recorded versus reconstructed task/configuration identity, resolved assets,
  simulator version, checker version, and diagnostic settings.
- Terminal geometry/velocity checks, contact refresh observations, per-requirement
  failures, success timeline, displacement/tilt, and final verdict.
- Temporal predicate initialization and any missing historical state.
- Verification mode, recorded/target physics identities, and trajectory-quality
  flags independent of the terminal task verdict when action replay is requested.

Preserve both `recorded_outcome` and `postprocess_success` in dataset/session
metadata. Keep processing completion separate from the verification verdict.
Missing reports and old caches are explicitly unverified.

## 2. Implement terminal-state continuation

Add a simulator-side checker alongside the existing materialization runtime.
Reuse its recorded-environment reconstruction and `restore_scene_state`, with
stricter checks for skipped task/physics fields, missing dynamic state, and
unavailable required contact sensors. Do not silently substitute current defaults.

Restore `states[N - 1]`, including velocities and robot joints. The last policy
camera frame uses `states[N - 2]` for pre-action alignment and is unsuitable for
this check. Clear predicate timers, stale contact buffers, frozen flags, and
episode bookkeeping; advance the predicate clock consistently with physics.

For the initial Simple-PnP profile:

- Check restored geometry and kinematics before continuation. Log initial
  violations so an airborne object falling into the bin later cannot erase them.
- Refresh contacts through actual physics. Distinguish unavailable initial
  contacts from a measured failure; record restoration transients separately.
- Run two simulated seconds, extending the declared horizon when the task's
  configured settling interval requires it. Preserve the recorded physics profile.
- Hold the final recorded robot command, including its gripper command. Do not
  force a release or reposition objects to obtain a passing result.
- Evaluate the original success predicate each control step and track motion
  and contact changes each physics step. Require a fresh temporal success window
  and success throughout the final 0.5 seconds. Apply declared displacement/tilt
  limits from the restored endpoint so a substantially different later resting
  state cannot pass as the recorded placement.
- Bypass automatic termination/freezing, reset callbacks, and recorder writes so
  the continuation can observe instability after the first successful sample.

Report an invalid terminal arrangement or sustained task failure as `failed`.
Report incompatible reconstruction, unsupported history-dependent predicates,
missing evidence, simulator errors, and ambiguous restoration artifacts as
`inconclusive`. Do not treat the first false sample of a freshly reset temporal
predicate as a failure by itself.

## 3. Integrate with postprocessing and retries

Run the checker after an episode's recorded camera materialization, before the
worker releases its environment. Integrate through
`scripts/materialize_demo_images.py`, `batch_worker.py`, `render.py`, and
`session.py`. Reuse the existing isolated GPU and serialized worker.

Reset all simulation/manager/contact state before the next episode. Test this
explicitly for persistent workers. Preserve exact training row counts, values,
timestamps, and camera alignment; diagnostic continuation belongs in sidecars.

Cache success evidence independently from image caches, keyed by source,
task/predicate code, configuration, dependencies, checker version, and settings.
An existing valid image cache can receive a missing check without re-encoding its
frames. Reuse only complete, checksummed verification evidence. Bound job and
teardown wall time and retain partial diagnostics after interruption.

Enable checking by default for supported recorded-success tasks. Provide an
explicit skip option that records `not_run`. Backfill existing processed sessions
through the same entry point. Infrastructure failures remain visible and retryable;
a measured task failure does not abort unrelated episodes.

## 4. Use the verdict in videos and export

Update `videos.py` to label each episode with its recorded outcome and verification
verdict. Keep one combined review MP4, including labeled failures/inconclusive
cases for inspection. Keep diagnostic settling frames outside the native-15
recorded timeline; they need not produce another video. Delete intermediate clips
only after combined-video verification, following the existing retention rule.

Include verification evidence identity in preview-cache compatibility so newly
verified episodes do not reuse obsolete labels. Publish a new preview version
when necessary rather than modifying acknowledged artifact sets in place.

Require valid `passed` evidence for normal success-only training export through
`export.py`; preserve diagnostic access to other outcomes. Passing this check
does not bypass existing structural, visual-quality, or loader-acceptance gates.

## 5. Validation and rollout

CPU tests cover verdict aggregation, temporal warmup, terminal-row selection,
unsupported tasks, report/cache invalidation, resume behavior, video labels,
combined-only retention, and export exclusion.

GPU fixtures cover a valid resting placement, airborne target, tabletop support,
held target, tipped bin, drifting object, delayed fall after apparent success,
and a placement that becomes successful only after substantial continuation
motion. For action replay, include an unrequested grasp loss followed by a passing
placement, plus nearly stationary poses with inconsistent reported velocities.
Verify that held targets are not rescued by an injected release. Check
batch-worker isolation against fresh processes and verify raw data preservation.

Run the checker on a small mixed set of real recorded-success demos. Review
false failures and reconstruction sensitivity, measure added time per episode,
and compare verdicts with terminal/settling visuals. Then enable the default
stage for new sessions and backfill selected existing sessions. Acceptance is
correct discrimination and reproducible evidence, not making every recorded
success receive a passing verdict.
