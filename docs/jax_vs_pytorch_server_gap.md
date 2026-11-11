# JAX vs. PyTorch pi0.5 success-rate investigation

Date: 2026-09-04

## Executive conclusion

The recorded gap is real: the JAX run completed **30/110 episodes (27.3%)**, while the
unsteered PyTorch/PPS run completed **22/110 (20.0%)**, a difference of **-7.3 percentage
points**.

The current runs do **not** isolate a PyTorch implementation regression. They use identical
tasks and initial simulator states, but different diffusion-noise streams:

- the JAX policy draws noise from its sequential JAX PRNG;
- the PPS client supplies `(task, run, env_id, chunk)`, and the PyTorch server derives noise with
  NumPy's PCG64 generator.

That difference changes the first predicted chunk substantially and immediately sends the
closed-loop rollouts down different trajectories. With only five noise draws per task, most
task-rate changes are a single episode, i.e. a 20-point task-level jump. The pooled gap is not
statistically distinguishable in either an ordinary two-proportion test (`p = 0.204`) or an exact
task-stratified conditional test (`p = 0.158`).

There is also a smaller, measured backend difference even when observation and noise are held
fixed. The converted PyTorch model and JAX model disagree by up to **0.0211 rad** on arm-joint
targets in the saved parity test. The likely source is their different mixed-precision execution:
JAX restores the parameter tree as bf16, whereas the PyTorch checkpoint is fp32 and only the
PaliGemma/expert submodule is selectively cast to bf16. Framework kernels also differ. That small
difference can be amplified by repeated closed-loop replanning and may contribute to the extra
placement failures, but these rollouts cannot separate its effect from the different noise.

**Bottom line:** there is no evidence here of a scaling, action-order, horizon, normalization, or
accidental-PPS-steering bug. The defensible explanation is **an uncontrolled random-noise A/B
comparison, plus a real but much smaller JAX/PyTorch numerical drift**. A paired same-noise
rollout is required to determine whether the backend drift changes success rate.

## Inputs inspected

Results:

- [JAX episode results](../output/pi05_single_stage_5eps/episode_results.jsonl)
- [PyTorch container results](../output/pps_jointpos_container_placement_5eps/episode_results.jsonl)
- [PyTorch surface results](../output/pps_jointpos_surface_placement_5eps/episode_results.jsonl)
- all 22 per-task `env_cfg.json` files, 22 JAX HDF5 files, 22 PyTorch HDF5 files, and representative
  videos under those directories

Serving/evaluation paths:

- `scripts/pps_single_stage/start_pps_server.sh`
- `policies/pps/client.py` and `policies/pps/run.py`
- `policies/pi0_family/client.py` and `policies/pi0_family/run.py`
- `/tmp2/leocheng/pps/scripts/serve_pps.py`
- `/tmp2/leocheng/pps/src/pps/droid/{config,serve,wire}.py`
- `/tmp2/leocheng/pps/src/pps/patch.py`
- the two OpenPI implementations used for parity checking

The `scripts/serve_pps.py` named in the request is not inside this RoboLab checkout. The local
launcher explicitly changes into `PPS_DIR` (default `/tmp2/leocheng/pps`) and invokes the file
there, so that adjacent checkout is the implementation reviewed here.

## Recorded result

| Scope | JAX | PyTorch/PPS base | Difference |
|---|---:|---:|---:|
| All tasks | 30/110 (27.3%) | 22/110 (20.0%) | -7.3 points |
| Container placement | 26/80 (32.5%) | 19/80 (23.8%) | -8.8 points |
| Surface placement | 4/30 (13.3%) | 3/30 (10.0%) | -3.3 points |

The nine tasks whose success counts changed were:

| Task | JAX | PyTorch | Difference |
|---|---:|---:|---:|
| SauceBottlesCrateTask | 4/5 | 3/5 | -1 |
| MustardInLeftBinTask | 2/5 | 0/5 | -2 |
| MustardInRightBinTask | 4/5 | 3/5 | -1 |
| ToyInBinTask | 1/5 | 0/5 | -1 |
| YogurtInBowlTask | 2/5 | 1/5 | -1 |
| SpoonInMugTask | 2/5 | 1/5 | -1 |
| FruitsOnionTask | 3/5 | 2/5 | -1 |
| SmallerObjectButterInBinTask | 3/5 | 4/5 | +1 |
| FruitsOnionToPlateTask | 4/5 | 3/5 | -1 |

Thirteen tasks had unchanged aggregate counts. Eight of the nine moving tasks went down, but the
primary count-based tests still do not reject equal backend success probabilities:

- JAX Wilson 95% interval: **19.8%–36.3%**
- PyTorch Wilson 95% interval: **13.6%–28.4%**
- pooled two-proportion z-test: `z = -1.27`, two-sided `p = 0.204`
- exact test conditioned on each task's combined number of successes: two-sided `p = 0.158`

The task-stratified test is preferable as a check because task difficulty is highly heterogeneous.
Neither test proves equivalence; they say that 110 episodes per backend are insufficient to
distinguish this observed gap from finite-sample variation.

## What was controlled

### Simulator and evaluation protocol

After replacing nondeterministic function-pointer addresses in the serialized configs, every
JAX/PyTorch `env_cfg.json` pair differs in only two expected fields: `policy` and
`recorders/dataset_export_dir_path`. Camera setup, physics, robot, observations, action terms,
instructions, five parallel environments, episode limits, and task definitions match.

Across all 22 tasks and five environments, **3,550 initial-state datasets containing 45,050 scalar
values are bit-identical** between the HDF5 recordings (`np.array_equal`; zero mismatched
datasets). Thus the result is not explained by a changed scene, robot pose, camera pose, or task
configuration.

Both clients use the same observation extraction, image resize, request schema, action unpacking,
15-action open-loop horizon, and `> 0.5` gripper threshold. `PpsClient` subclasses the JAX-run
client and only appends the steering/noise-pairing keys.

PyTorch inference was slower (23.88 versus 17.09 aggregate policy-inference minutes), but total
simulator stepping time was essentially unchanged (96.31 versus 97.13 minutes). RoboLab advances
fixed simulation steps synchronously, so wall-clock inference latency does not change the control
rate or explain success.

### No PPS steering was active

Every PyTorch result row is labelled `pps_base`; the launch/evaluation defaults are `mode=none`,
`gamma=0`, and no arm. In this path:

1. `wire.pop` resolves the request to `mode=none`;
2. `SteeredPolicy._resolve_arm` returns `None`;
3. with no chunk log, no guidance closure is created;
4. the policy calls OpenPI's native PyTorch `Policy.infer(obs, noise=noise)`;
5. the patched `denoise_step` returns OpenPI's original velocity tensor unchanged when no guidance
   function is bound.

Therefore the 20.0% run measures the frozen PyTorch base, not a PPS residual.

### Configuration, transforms, and normalization

- Both model configs are pi0.5 with padded action width 32 and action horizon 15.
- Both use ten Euler denoising steps.
- The joint-position config applies `AbsoluteActions(mask=[first 7 dimensions])`, then returns the
  first eight real channels: seven joints plus gripper.
- The client executes all 15 returned actions before replanning on both paths.
- The JAX and converted checkpoints' `assets/droid/norm_stats.json` files have the same SHA-256:
  `57ce9956f9e07d65f8a8205aabec72d436a2c8927f53edb40c7a77b14a5a90c7`.
- First-chunk commands from both recorded runs are finite, have plausible joint ranges, and have a
  strictly binary gripper channel after client postprocessing.
- The relevant lightweight PPS checks pass: **28 tests passed** across
  `test_droid_wire.py`, `test_droid_serve.py`, and `test_droid_config.py`.

This rules out the common gross errors: wrong normalization assets, delta actions sent as absolute
positions, a 7/8-channel mismatch, swapped action channels, wrong chunk horizon, wrong denoising
step count, or steering silently applied to the control.

## Why the trajectories differ

### 1. The two evaluations did not use the same diffusion noise

OpenPI's JAX `Policy` starts from a JAX key and splits it once per request; when no explicit noise
is supplied, `Pi0.sample_actions` calls `jax.random.normal`. The result depends on the sequential
request history.

The PPS client instead sends task/run/environment/chunk indices. `pps.droid.wire.seed_for`
hashes those into an integer, and `noise_for` calls
`np.random.default_rng(seed).standard_normal(..., dtype=float32)`. Even if the same integer were
given to both APIs, JAX and NumPy use different generators and would not produce the same tensor.

The PPS scheme itself shows no defect in this run. Reconstructing all inferred chunks from the
episode lengths gives:

| PPS-noise check | Value |
|---|---:|
| Inference chunks | 5,090 |
| Unique seeds | 5,090 |
| Seed collisions | 0 |
| Generated scalar samples | 2,443,200 |
| Mean | 0.000422 |
| Standard deviation | 0.999640 |
| Fraction below zero | 0.499984 |

The issue is not biased or repeated PPS noise; it is that the JAX and PyTorch arms were evaluated
on **different valid draws**.

The scale of that confound is visible before closed-loop state can diverge. Comparing the aligned
first 15 executed actions for all 110 task/environment pairs gives:

| Arm-target difference | Actual runs, different noise | Saved parity, same noise |
|---|---:|---:|
| Mean absolute difference | 0.06112 rad | 0.00396 rad |
| Median absolute difference | 0.03741 rad | 0.00197 rad |
| 95th percentile | 0.19893 rad | 0.01805 rad |
| Maximum | 0.88153 rad | 0.02113 rad |
| Gripper decisions that differ | 77/1,650 | 0/45 |

The columns are not a formal variance decomposition—the same-noise check has only three synthetic
observations—but they establish that the actual runs differ much more at their first chunk than
the controlled backend-only parity sample. After those first different actions, robot/object poses
and camera frames differ, so every subsequent policy request compounds the divergence.

### 2. JAX and PyTorch are close, not numerically identical

The saved `/tmp2/leocheng/pps/runs/parity_droid/{jax,torch}.npz` artifacts feed both backends the
same observation and exact same epsilon. Across three deterministic observation regimes and 45
actions, the controlled comparison reports:

- arm mean absolute difference: **0.00396 rad**;
- worst arm difference: **0.02113 rad**;
- maximum raw gripper difference: **0.05015**;
- gripper decisions crossing the client threshold: **0/45**.

The JAX checkpoint is restored with the whole parameter tree requested as bf16. The PyTorch
converter emits fp32 safetensors (`config.json` records `"precision": "float32"`), after which
OpenPI casts the PaliGemma/expert module selectively to bf16 while keeping selected layers—and
outer action/time projections—fp32. JAX/XLA and PyTorch/CUDA attention and matrix kernels also do
not share an operation order. Ten Euler steps integrate the resulting velocity differences.

This is small for one teacher-forced chunk, but a failed episode can contain 60 replans. Small
differences can change contacts and releases, after which closed-loop observations cease to be
nearby. The parity result validates conversion at a useful tolerance; it does not guarantee equal
rollout outcomes.

### 3. The lost successes are concentrated after grasp

The existing comparison helper undercounts surface-placement failures because its `stage()`
classifier recognizes `in_container` but not `object_on_top`. Classifying both as step-2 placement
conditions gives:

| Final state-machine result | JAX | PyTorch |
|---|---:|---:|
| Success | 30 | 22 |
| Explicit step-1 grasp failure | 68 | 67 |
| Explicit step-2 placement failure (`in_container` or `on_top`) | 7 | 16 |
| Ambiguous multi-object `any` message | 5 | 5 |

The ambiguous rows are all `OneBottleOnShelfTask`; its final message reports the first incomplete
object and cannot reliably distinguish grasp from placement. Excluding those equal-size ambiguous
groups, the PyTorch run reaches the post-grasp state about as often (success + explicit placement
failure: 38 versus 37) but converts fewer of those attempts into success.

This pattern is inconsistent with a gross gripper-channel bug and is compatible with trajectory
differences around transport/release. It does **not** identify whether those differences came from
the new noise draw or the smaller backend numerical drift.

A representative pair is `SauceBottlesCrateTask`, environment 0:

- [JAX video](../output/pi05_single_stage_5eps/SauceBottlesCrateTask/Put_the_red_bbq_sauce_bottle_in_the_crate_0_env0_viewport.mp4): success at step 232 (15.47 s), two target-drop events.
- [PyTorch video](../output/pps_jointpos_container_placement_5eps/SauceBottlesCrateTask/Put_the_red_bbq_sauce_bottle_in_the_crate_0_env0_viewport.mp4): reached the grasp condition but failed the container condition at step 600 (40.0 s), ten target-drop events.

The videos show visibly different approach/retry trajectories from the same authored scene, as
expected once the first action chunks use different epsilon.

## Source-version and evidence limitations

- The likely JAX checkout is `xuningy/openpi` at `aa642056...`; PPS pins upstream OpenPI at
  `15a9616a...`. The JAX model, policy, policy-config, transforms, and converter files relevant to
  this path are byte-identical between those checkouts. `DroidOutputs` differs only by changing
  `[:, :8]` to `[..., :8]`, which is equivalent for these unbatched server requests. The PyTorch
  sampler differs mainly by making `torch.compile` configurable; PPS deliberately disables it so
  the steering hook can execute.
- The result folders do not contain a server manifest, source commit, checkpoint digest, noise
  tensor/seed, or dependency versions. Therefore the exact server processes that produced these
  files cannot be cryptographically tied to the currently inspected dirty worktrees.
- The saved same-noise parity test uses three synthetic observations, not recorded RoboLab policy
  camera frames. The HDF5 files do not contain those two policy images.
- `scripts/parity_droid.py` hard-codes the detailed `torch=`/`jax=` labels rather than deriving them
  from argument order. With `--compare jax.npz torch.npz`, those two printed value labels are
  reversed; the absolute-difference and threshold-flip calculations used above are unaffected.

These limitations are why this report does not assign a percentage of the eight lost successes to
randomness versus mixed precision.

## Recommended decisive experiment

1. Wrap the JAX policy so it consumes the same task/run/environment/chunk tuple as PPS and passes
   the exact same NumPy-generated epsilon into `Policy.infer(..., noise=epsilon)`.
2. Run JAX and PyTorch on the same five (preferably at least 20) paired trials per task. Keep the
   current simulator configs and 15-action horizon.
3. Compare paired episode outcomes with McNemar's exact test, and report task-stratified results.
4. Record the two policy camera frames and raw returned chunks at every replan. On the same logged
   observation/epsilon pairs, measure per-dimension parity and gripper threshold crossings.
5. If a paired gap remains, ablate PyTorch precision: current mixed precision versus a
   JAX-matching bf16 treatment of the outer action/time projections. This directly tests the only
   demonstrated backend-level discrepancy.
6. Save a run manifest containing both Git revisions (including a dirty-tree patch/hash), OpenPI
   revision, checkpoint and normalization hashes, full server metadata, seed algorithm/salt, and
   launch arguments.

For PPS steering experiments, the correct control remains `mode=none` on the **same PyTorch
server and same deterministic seed schedule**. The 27.3% JAX number should not be used as the
gamma-zero PPS baseline.

## Reproduction commands used

```bash
python scripts/pps_single_stage/compare_to_baseline.py \
  --baseline output/pi05_single_stage_5eps \
  --arm output/pps_jointpos_container_placement_5eps \
        output/pps_jointpos_surface_placement_5eps

cd /tmp2/leocheng/pps
.venv/bin/python scripts/parity_droid.py \
  --compare runs/parity_droid/jax.npz runs/parity_droid/torch.npz

PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider \
  tests/test_droid_wire.py tests/test_droid_serve.py tests/test_droid_config.py
```

