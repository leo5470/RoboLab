# pi05_droid_jointpos single-stage evaluation, through the PPS server

The jointpos counterpart of [`../pi05_single_stage`](../pi05_single_stage). **Same model, same 22
tasks, same protocol** — the difference is which server answers the requests:

|  | `pi05_single_stage` | `pps_single_stage` |
|---|---|---|
| server | openpi `serve_policy.py`, **JAX** | PPS `scripts/serve_pps.py`, **PyTorch** |
| checkpoint | `gs://openpi-assets-simeval/pi05_droid_jointpos` | the same, converted to PyTorch |
| client | `policies/pi0_family/run.py` | `policies/pps/run.py` (adds the steering keys) |
| steering | none possible | `MODE`/`GAMMA`/`ARM`, default **none** |

PPS adds `gamma*(v_task - v_ref)` *inside* the denoising loop, and openpi's JAX sampler is a jitted
`lax.while_loop` with no hook — hence the PyTorch copy. With `MODE=none` the wrapper computes
`v + 0.0`, which is bitwise `v` (checked end to end by the PPS repo's
`scripts/wire_identity_droid.py`), so the base arm and every steered arm are measured through one
code path rather than two.

## Why this exists

The published numbers in `output/pi05_single_stage_5eps` were produced by the JAX server. The two
backends load the same weights in slightly different dtypes — JAX restores bf16 params, the
PyTorch path loads fp32 safetensors and casts selected params to bf16 — and ten Euler steps
accumulate that difference. Per chunk it is small (worst arm-joint disagreement 0.021 rad on
targets spanning ~2.2 rad), but a rollout is 60 chunks of closed loop, so **the base arm has to be
re-measured on this server** before any steered arm can be compared against it. That is what these
scripts do.

## Run

```bash
# terminal 1 — the server (GPU 1 by default; IsaacSim takes GPU 0)
./scripts/pps_single_stage/start_pps_server.sh

# terminal 2 — the task sets
./scripts/pps_single_stage/run_container_placement.sh    # 16 tasks
./scripts/pps_single_stage/run_surface_placement.sh      #  6 tasks
```

`NUM_ENVS` defaults to **5** here, not 1: that is what the 5eps baseline ran, and the comparison is
only meaningful at the same episode count. `VIDEO_MODE` defaults to `viewport` for the same reason.

Server variables: `PPS_DIR`, `CKPT`, `PORT`, `SERVER_GPU`, `MODE`, `GAMMA`, `ARM`, `LOG_DIR`.
Evaluation variables: `REMOTE_HOST`, `REMOTE_PORT`, `NUM_ENVS`, `NUM_RUNS`, `MODE`, `GAMMA`, `ARM`,
`VIDEO_MODE`, `OUTPUT_FOLDER_NAME`, `ISAAC_PYTHON`. Extra arguments pass through to the runner.

## A steered arm over the same tasks

```bash
LOG_DIR=/tmp2/leocheng/pps/runs/droid/obs \
  ./scripts/pps_single_stage/start_pps_server.sh          # record the observation corpus

ARM=pumpkin=/path/v_ref/final.pt,/path/v_task/final.pt \
  ./scripts/pps_single_stage/start_pps_server.sh
MODE=pps GAMMA=0.8 ARM=pumpkin OUTPUT_FOLDER_NAME=pps_g0.8_container \
  ./scripts/pps_single_stage/run_container_placement.sh
```

One server process holds several arms, so a whole gamma curve needs one model load. Every request
carries `(episode, env_id, chunk)`, from which the sampler's noise is derived — so two arms run
days apart face identical epsilon at the same point of the same trial, and exact McNemar applies
across them.

## Comparing against the baseline

```bash
python scripts/pps_single_stage/compare_to_baseline.py \
    --baseline output/pi05_single_stage_5eps \
    --arm output/pps_jointpos_container_placement_5eps \
          output/pps_jointpos_surface_placement_5eps
```

Note what is and is not paired. Scenes are identical — benchmark tasks reset to their authored
state, so every episode of every env starts from the same place in both runs. The **noise is not**:
the JAX server drew epsilon from its own rng, this one derives it from the trial indices. So the
per-episode outcomes are not expected to match one for one; the task-level and pooled rates are
what the comparison is about.
