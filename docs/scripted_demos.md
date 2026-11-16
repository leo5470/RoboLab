# Scripted grasp-library demonstrations

Scripted pick-and-place demonstrations for Simple-PnP objects that need side or handle grasps, written as PPS
training files (one `pps.hdf5` per object family). Grasps come from
[GraspDataGen](https://github.com/NVlabs/GraspDataGen) (Robotiq 2F-85), are mapped to DROID's gripper frame, and are
chosen at each reset with MolmoSpaces-style filtering and ranking. The arm follows a five-waypoint plan through
absolute-pose DLS IK that produces joint targets for the stock DROID joint-position environment, so the recordings
share the registration that policies are evaluated in.

A WorldState scripted expert (`robolab/policies/scripted/`) exists on later branches. It derives top and side grasps
from bounding boxes; this pipeline uses simulated GraspDataGen grasps, which handle objects such as the pitcher that
can only be lifted by the handle.

| Code | What |
| --- | --- |
| `robolab/core/scripted_demos/grasps/` | `frames` (rigid transforms, GraspDataGen to DROID map), `library` (GraspDataGen YAML to grasp arrays), `objects` (the batch objects) |
| `robolab/core/scripted_demos/` | `kinematics` (numpy FK/IK of the DROID arm), `select` (grasp selection and plans), `motion` (segment timing, executor, command emulation), `sim` (app, envs, IK controller, scene state), `collect` (attempt loop), `writer` (PPS HDF5), `paths`, `records` |
| `scripts/scripted_demos/` | `run.sh`, simulator scripts, `validate_pps.sh`, `demo_viewer.py` |
| `scripts/scripted_demos/grasps/` | mesh and robot-geometry export, GraspDataGen runs, library build, renders |
| `tests/unit/test_scripted_demos_*.py` | numpy-only tests |

`kinematics`, `select`, `motion`, `writer`, `records` and `grasps` import nothing from Isaac Sim. `sim` and
`collect` need the Isaac venv, and `sim.launch_app` must run before anything imports isaaclab.

## Running

`scripts/scripted_demos/run.sh` runs a script with a RoboLab venv and this checkout first on `PYTHONPATH`, writing no
bytecode, so a venv can be borrowed read-only (`ROBOLAB_VENV=/path/to/RoboLab/.venv`; default: this checkout's
`.venv`):

```bash
SIM_GPU=0 scripts/scripted_demos/run.sh SCRIPT.py ...   # simulator, isolated on host GPU 0 with bubblewrap
scripts/scripted_demos/run.sh --pxr SCRIPT.py ...       # USD only, with the venv's bundled pxr; no GPU
scripts/scripted_demos/run.sh --cpu SCRIPT.py ...       # plain Python (numpy tools, the viewer)
```

The renderer enumerates GPUs through Vulkan and ignores `CUDA_VISIBLE_DEVICES`, so simulator mode gives the process a
private `/dev` with only the chosen GPU (as `robolab/core/omy_postprocess/gpu_isolation.py` does). `SIM_GPU` has no
default. Never run two simulators on a 10 GB card: PhysX then fails to allocate and the state silently becomes NaN.
A simulator process uses about 7 GB and runs at ~190 ms per rendered step; Kit's first update takes ~3.5 minutes.

### Data locations

| Variable | Default | Holds |
| --- | --- | --- |
| `GRASP_DATA_DIR` | `output/grasp_data` | meshes, masses, robot chain, GraspDataGen runs, grasp library, trial and screening logs (`qualified/<task>.json`) |
| `SCRIPTED_DEMOS_DIR` | `output/scripted_demos` | one directory per family: `pps.hdf5`, `attempts.jsonl`, `manifest.json`, `summary.json`, `videos/` |
| `KIT_PORTABLE_ROOT` | `output/kit_portable` | Kit's data, cache and logs (otherwise Kit writes into the venv, and a shared shader cache recompiles on every start) |
| `DEMO_VIEWER_CACHE` | `output/demo_viewer_cache` | encoded viewer videos |
| `GRASPDATAGEN_DIR` | none | GraspDataGen checkout at `8e09005`; needed by `run_graspdatagen.sh`, `export_robot_geometry.py`, `build_library.py` |
| `PPS_REPO`, `PPS_PYTHON` | none, `$PPS_REPO/.venv/bin/python` | pps checkout for `validate_pps.sh`; `collect_demos.py` skips validation without it |

Every long run resumes from its JSONL and HDF5 outputs.

## Pipeline

| Phase | Work | Gate |
| --- | --- | --- |
| 0 | Environment bring-up; each object's simulated mass | the stock registration runs; masses recorded |
| 1 | Mesh export and frame check | exported meshes agree with the stage's bounds |
| 2 | Arm chain, gripper geometry and GraspDataGen to DROID frame map; GraspDataGen runs; grasp library | grasps that held under the object's own mass |
| 3 | Executor and floating-gripper to arm transfer (lift) test | lifts succeed with library grasps |
| 4 | Selector and full pick-and-place trials | per-object success rate |
| 4b | Fallback screening (D9), only for objects that fail Phase 4 | screened grasps pass the Phase 4 trials |
| 5 | Collector and PPS writer | PPS `--validate-only` passes; episodes replay exactly |
| 6 | Batch collection | as Phase 5, per object |

```bash
R=scripts/scripted_demos/run.sh; D=scripts/scripted_demos
# Phase 0-1: meshes in the RoboLab object frame, masses, frame check
$R --pxr $D/grasps/export_meshes.py
SIM_GPU=0 $R $D/probe_objects.py
# Phase 2: arm chain, gripper geometry and frame map; GraspDataGen; library; renders (matplotlib)
$R --pxr $D/grasps/export_robot_geometry.py
CUDA_VISIBLE_DEVICES=0 $D/grasps/run_graspdatagen.sh
$R --cpu $D/grasps/build_library.py
$R --pxr $D/grasps/render_grasps.py
# Phase 3-4b: lift test, pick-and-place trials, per-layout screening, trials on screened grasps
SIM_GPU=0 $R $D/trials.py transfer
SIM_GPU=0 $R $D/trials.py pnp
SIM_GPU=0 $R $D/trials.py screen --objects 303 --grasp-class handle
SIM_GPU=0 $R $D/trials.py pnp --objects 303 --qualified
# Phase 5-6: collection (several objects per process; PPS validation after each if PPS_REPO is set), replay check
SIM_GPU=0 $R $D/collect_demos.py --objects 301 294 230 283 --episodes 50
SIM_GPU=0 $R $D/collect_demos.py --objects 303 --qualified --num-layouts 14
SIM_GPU=0 $R $D/replay_check.py --file <family>/pps.hdf5 --episodes 5
```

Each script documents its options in its header and under `--help`.

### Demo viewer

```bash
scripts/scripted_demos/run.sh --cpu scripts/scripted_demos/demo_viewer.py   # http://localhost:8765
```

Every family under `SCRIPTED_DEMOS_DIR` (and one level below) that holds a `pps.hdf5` is listed. An episode plays as
the two policy cameras (exterior and wrist, 224 px, 15 fps, encoded with ffmpeg on first request; `--prerender`
encodes everything up front), beside the collector's hi-res renders, phase timeline and joint traces. Good / bad /
unsure marks with notes go to `<family>/review.json`. The page polls every 15 s, so episodes appended by a running
collection appear without a reload; files are opened read-only and without HDF5 locking, so the viewer never blocks
a writer.

## As MimicGen sources

Each recorded episode also stores `pre_action_state/` (not read by PPS): the robot's full joint vector and the object
and bin root poses before every action, in Isaac Lab's recorder layout, plus the joint order. With it, a scripted
episode is a MimicGen source of kind `"scripted"` (`robolab/core/scripted_demos/mimic_source.py`, registered in
`robolab/core/omy_teleop/mimic_sources.SOURCE_READERS`; see docs/omy_mimicgen.md, "Source kinds"). It reads back as
the same `SourceEpisode` a teleoperation recording of the same motion gives: actions with the gripper in the
generator's +1 open / -1 closed convention, and the grasp step defined the same way (gripper closed, object 2 cm up).
A source reference is `{"file": "<family>/pps.hdf5", "episode": "episode_000012", "kind": "scripted"}`, for example
in a path-screen `--sources` file. Episodes collected before the state group existed are refused; collect them again.

Two differences from teleoperation sources are not resolved by reading them:

- **`panda_joint3`.** The generator holds it at home (the OMY leader has six axes) and refuses sources that move it;
  scripted episodes move all seven joints (D1, D2). Replaying the stored plans of the 53 scripted and 50 MimicGen
  pitcher episodes through the command emulation with joint 3 held at home, 51/53 and 40/50 still converge (failures
  at pregrasp and preplace), and every grasp id keeps at least one converging episode.
- **Environment.** Scripted episodes are recorded in the stock DROID joint-position env (R2); the generator runs in
  the OMY teleoperation env and compares each source's env contract with the target's.

## Batch objects

Pitcher (303), mug (301), utilityjug_a01 (283), utilityjug_a02 (284), milkjug_a01 (230), milkjug_a02 (231) and
cheez_it (294), each on its Simple-PnP v1 layouts other than the evaluation layouts 00, 05, ..., 45. The mug, cheez_it,
milkjug_a01 and utilityjug_a01 passed Phase 4 with the default selection; utilityjug_a02 passed with an 8 cm carry
clearance; the pitcher (handle grasps only) and milkjug_a02 needed Phase 4b screening. All seven were collected and
passed Phase 5.

## Requirements

- **R1.** Every recorded row follows the PPS wire convention: observations go through `Pi0DroidJointposClient`'s own
  conversion, and `actions` are the joint targets actually sent to `env.step` plus a 0 (open) / 1 (closed) gripper.
- **R2.** Episodes are recorded in the stock DROID joint-position envs (`auto_register_droid_envs(task_dirs=
  ["single_pnp"], ...)`) with the default WRIST_LEFT cameras: 15 Hz, `dt = 1/120`, decimation 8.
- **R3.** Only successful episodes are kept (the env's `success` termination). File attributes `fps=15`,
  `task=<family>`, `prompt=<default instruction>`; episode attributes `success`, `num_samples`, `layout_task`,
  `source_family`, plus the plan, grasp id, seed and executor summary.
- **R4.** Training layouts never include the evaluation layouts 00, 05, ..., 45.
- **R5.** Everything is reproducible from manifests: GraspDataGen commit, gripper and object hashes, parameters, seeds.
- **R6.** Nothing is written into the venv or another user's tree; outputs go to the data locations above.

## Design decisions

- **D1 Execution.** Isaac Lab's `DifferentialIKController` (pose, absolute, DLS) on the Robotiq `base_link` over all
  seven joints, run in this code every control step. The joint delta is clamped to ±0.04 rad, then to the soft limits,
  and the joint targets go to the stock joint-position env. The commands depend only on the plan, so recorded actions
  replay exactly, and the registration is the one policy rollouts use.
- **D2 Seven joints, own writer.** `panda_joint3` is free. The PPS file is written here because
  `omy_postprocess/pps_hdf5.verify_file` requires joint 3 to be constant.
- **D3 Grasp source.** GraspDataGen with `--gripper_config robotiq_2f_85`, run offline per object, validated twice on
  the same guesses: with the object's simulated mass and with GraspDataGen's default 1 g. Only MolmoSpaces' run-time
  selection logic is ported; none of its data or code is used.
- **D4 Mesh export.** One OBJ per object in the object frame `WorldState.get_pose` reports, with root rotation and
  scale applied; GraspDataGen's own USD-to-OBJ conversion is not used.
- **D5 Gripper frame map.** One fixed transform from GraspDataGen's `base_link` to DROID's: DROID x = GDG z,
  DROID y = -GDG y, DROID z = GDG x, plus the measured offset along the approach axis (1.5-2 mm). Each grasp is also
  stored with its 180° twin about the approach axis, since the gripper is symmetric.
- **D6 The collector owns the loop.** It uses `create_env` and its own step loop instead of `robolab.eval.runner`, so
  it has in-process `WorldState` (poses, bounding boxes, contacts) and controls episode boundaries.
- **D7 Code home.** The layout in the table at the top.
- **D8 Python environments.** `run.sh`, with any RoboLab venv (Isaac Lab 2.2 / Isaac Sim 5.0 tested).
- **D9 Fallback screening.** Enabled per object only when it fails the Phase 4 gate. Screening stores the grasps that
  succeeded per layout, with the selection and motion configuration, in `qualified/<task>.json`; `--qualified` draws
  only from those and refuses files without a configuration.
- **D10 GPU.** One simulator per 10 GB card; the GPU is always chosen explicitly (`SIM_GPU`).

## Tests

```bash
.venv/bin/python -m pytest --confcutdir=tests/unit tests/unit/test_scripted_demos_*.py
```

They need numpy, h5py and PyYAML only. `tests/unit/scripted_demos_helpers.py` holds the toy scenes, grasps and
libraries they share.
