# Simple-PnP v2 diagnostics

Planning and investigation tools for [the v2 plan](../../docs/simple_pnp_v2_plan.md).
They produce **diagnostic** measurements, not qualification evidence: the
batch generator, qualification runner, and visibility renderer are not
implemented yet. Run everything from the repository root. Write outputs under
`output/simple_pnp_v2_validation/`, which is gitignored.

| Tool | Needs | Purpose |
| --- | --- | --- |
| `usd_python.sh` | project `.venv` with Isaac Sim | Run a script with `pxr` outside Isaac, using Isaac's bundled USD libraries (plain `.venv/bin/python` cannot import `pxr`). |
| `handle_directions.py` | `pxr` | Preliminary handle directions from mesh outlines, in each asset's composed frame. Confirm with renders before freezing. |
| `collision_bottoms.py` | `pxr` | Collision-shape vs visual bottoms, collision approximations/offsets, and root rotations scenes must preserve. Exit 1 if any collision shape lies below its visual bottom. |
| `reach_map.py` | `pxr` | Kinematic top-down reach map for the DROID Franka + Robotiq 2F-85. FK is checked against the authored reset pose. Ignores collisions. About 3 minutes for three heights. |
| `capacity_estimate.py` | numpy | Rough Monte Carlo estimate of per-entry layout capacity under the table, clearance, strict-novelty, and optional reach constraints. |
| `run_isolated.py` | `bwrap`, a GPU | Run an Isaac script on one host GPU via `isolated_gpu_launch`. |
| `start_probe.py` | Isaac, a GPU | Reset, prepare, and hold scenes in the OMY collection profile across seeds and clearances, optionally with a ground-plane candidate copy. Records every rigid body, including the table. |
| `analyze_start_probe.py` | numpy | Summarise probes against the table's current top. Compare cases within a run and, optionally, across two runs. |

Examples:

```bash
tools/simple_pnp_v2/usd_python.sh tools/simple_pnp_v2/collision_bottoms.py --retained \
    --report output/simple_pnp_v2_validation/collision_bottoms.json
tools/simple_pnp_v2/usd_python.sh tools/simple_pnp_v2/reach_map.py \
    --output output/simple_pnp_v2_validation/reach_map.json
.venv/bin/python tools/simple_pnp_v2/capacity_estimate.py \
    --reach-map output/simple_pnp_v2_validation/reach_map.json

# Reproduction check on a new machine (reset determinism, then the ground-plane fix):
.venv/bin/python tools/simple_pnp_v2/run_isolated.py cuda:0 tools/simple_pnp_v2/start_probe.py \
    --task SimplePnp230MilkjugA0100Task SimplePnp064ComputerMouse02Task --seeds 0 1 2 0 \
    --output output/simple_pnp_v2_validation/probes/determinism_gpu0
.venv/bin/python tools/simple_pnp_v2/run_isolated.py cuda:0 tools/simple_pnp_v2/start_probe.py \
    --task SimplePnp230MilkjugA0100Task --clearances 0.002 0.00001 --ground-z -0.700 \
    --output output/simple_pnp_v2_validation/probes/ground_gpu0
.venv/bin/python tools/simple_pnp_v2/analyze_start_probe.py output/simple_pnp_v2_validation/probes/determinism_gpu0
```

Isaac processes often exit 139 at shutdown after writing results. `start_probe.py`
writes each task's JSON before teardown, so judge outcomes from those files.

## Results to reproduce (2026-11-02, RTX 3090, Isaac Sim 5.0.0 / Isaac Lab 2.2.0)

- **Reset determinism:**
  - post-reset states identical across seeds;
  - positions identical across separate processes;
  - successive in-process resets drift 0.2–0.5 mm, even with the same seed;
  - the mouse layout amplifies that to 3–4 mm / 7–8°.
- **Upstream ground plane:**
  - the table root rises from z −0.350 to −0.347 in the first step;
  - with `--ground-z -0.700` the table does not move;
  - contact starts (1e-5) move objects 0.01–0.22 mm in total;
  - 2 mm starts drop about 2 mm.
- **Collision bottoms:** none below visual bottoms, maximum +0.83 mm. Only `computer_mouse.usd` has an up-tilting root rotation.
- **Reach and capacity:** reach and capacity numbers as recorded in the plan.
