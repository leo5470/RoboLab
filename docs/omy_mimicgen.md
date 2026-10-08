# MimicGen for OMY-L100 demonstrations

`examples/generate_omy_mimic.py` expands a few L100 teleoperation demonstrations of one pick-and-place task into many, using Isaac Lab Mimic. Isaac Lab bundles it, but importing it needs `ipython` and `ipywidgets`, which `requirements-omy.txt` installs. It does not transfer demonstrations between objects: collect source demonstrations for each task.

```bash
python examples/generate_omy_mimic.py \
    --input output/omy_teleop/<run_1> output/omy_teleop/<run_2> \
    --output-dir output/omy_mimicgen/banana_in_bowl --num-demos 40
```

Each `--input` is a collection run directory containing `omy_demos.hdf5` and `env_cfg.json`. Only successful episodes are used; with a single run, `--episodes 1 2 4` restricts the sources to `demo_1`, `demo_2`, and `demo_4`, and refuses any that are missing or unsuccessful. The task, control rate, stepping profile, recording profile, and solver settings are taken from the first run's recording, so mix only runs collected with the same settings.

## What is generated

Generation runs in the teleoperation environment. Isaac Lab Mimic plans end-effector waypoints; the environment converts each waypoint to an absolute joint command with one damped-least-squares IK step, so the output `omy_demos.hdf5` has the same actions, states, and observations as a teleoperation recording and goes through the same postprocessing (see [postprocess.md](postprocess.md)). An episode ends when the task's success term fires, as in teleoperation. The run-level `robolab_omy_teleop` and `robolab_omy_collection` attributes are copied from the first source run, so `scripts/materialize_demo_images.py` treats the output as it treats that source. The `robolab_omy_mimicgen` dataset attribute records the sources, seed, and randomization; `mimicgen_summary.json` lists every attempt.

The L100 has six axes and never commands `panda_joint3`. IK therefore solves for the other six joints and holds `panda_joint3` at its home value; source episodes that command it are refused.

End-effector poses of the source episodes are not recorded. They are computed by writing the recorded joints (measured, then commanded) into the simulator and reading the gripper base pose, without stepping physics.

## Segments

A demonstration is split in two: up to the grasp, expressed relative to the object, and after it, expressed relative to the container. The grasp is the first step at which the gripper command is closed and the object is 2 cm above its starting height. Object and container default to the `object` and `container` parameters of the task's `success` term; override them with `--object` and `--container`.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--pose-range` | `0.1` | Uniform x/y reset offset in meters for the object and container |
| `--yaw-range` | `0` | Uniform reset yaw offset in radians |
| `--object-pose-range` | `--pose-range` | x/y reset offset in meters for the grasped object alone |
| `--object-yaw-range` | `--yaw-range` | Reset yaw offset in radians for the grasped object alone |
| `--no-randomize` | off | Keep the recorded arrangement, reproducing the source demonstrations |
| `--interpolation-s` | `1.0` | Simulated seconds for approaching each transformed segment |
| `--settle-s` | `2.0` | Simulated seconds to hold the final target while waiting for success |
| `--action-noise` | `0` | Standard deviation in meters of noise on target positions |
| `--max-attempts` | `200` | Attempt budget; the exit status is 1 if it is spent first |
| `--keep-failed` | off | Also export failed attempts (`success` attribute false) |

Randomization replaces the task's reset events for the generation run: the object and container are offset from their scene poses and everything else is reset to its default. The object options give the grasped object a wider range than the container. Sampling does not avoid the other objects, so choose ranges that keep the object clear of them and on the table.

Run with `--no-randomize` first for a new task or new source data. The arrangement then matches the sources, so most attempts should succeed; a source that keeps failing points at the source data or the grasp segmentation rather than at the randomization range.
