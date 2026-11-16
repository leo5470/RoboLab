# How to evaluate trained PPS proxy pairs on an object's v2 layouts

This is the procedure behind the pail L/T/old evaluation of 2026-11-13
(`output/omy_layout_arms/evaluation_20261113/`), turned into two scripts:

- `scripts/pps_single_stage/eval_proxy_pairs.py` starts one PPS server, runs the base policy and
  every pair at every gamma over the object's Simple-PnP v2 layouts, and writes the paired report.
- `scripts/pps_single_stage/analyse_proxy_pairs.py` is the report on its own, for runs that already exist.

For what the PPS server is and why the base policy has to be measured through it, see
[`scripts/pps_single_stage/README.md`](../scripts/pps_single_stage/README.md). For the protocol of
the earlier campaigns and their results, see
[`pps_proxy_evaluation_results_20261106.md`](pps_proxy_evaluation_results_20261106.md).

## What you need

- **The PPS repo** at `/tmp2/leocheng/pps` (override with `PPS_DIR`) with the converted base
  checkpoint `runs/ckpt/pi05_droid_jointpos_pt`.
- **The trained pairs**, unzipped. Each pair is a folder with `v_ref/` and `v_task/`, each holding
  `final.pt`, `manifest.json` and `tracking.json`, as the training job writes them.
- **Free GPU memory.** The server needs about 9.5 GB; it goes on GPU 1 (the 10 GB card) and fills
  it. Each simulation needs about 6.5 GB on GPU 0; three at once use about 20 of its 24 GB. The
  machine is shared: the script waits until the memory it needs is free and never stops other jobs.

## Run it

```bash
PAIRS=/path/to/unzipped/pairs
scripts/pps_single_stage/eval_proxy_pairs.py --object utilityjug_a01 \
    --pair T50=$PAIRS/SimplePnp283UtilityjugA01_T50 \
    --pair T100=$PAIRS/SimplePnp283UtilityjugA01_T100 \
    --pair T200=$PAIRS/SimplePnp283UtilityjugA01_T200 \
    --gammas 0.4 0.8
```

`--object` is the object's name in the v2 task files (`pnp_283_utilityjug_a01_00_task.py` gives
`utilityjug_a01`). The name before `=` in `--pair` is the arm name used in folder names and in the
report. Add `--dry-run` to print the server and client commands without starting anything.

What it does, in order:

1. **Checks the pairs.** Each `final.pt` is hashed and compared with the hash recorded at training
   time, each `v_task` must have been trained from its own `v_ref`, and all pairs must be for the
   same task. It prints each pair's task, training steps, dataset size and prompt.
2. **Starts one server** on `--server-gpu` (default 1) and `--port` (default 8018), with the base
   policy as its default mode and one `--arm` per pair. Each request then names its own arm and
   gamma. (Started in steering mode without a default arm, the server refuses to start.)
3. **Runs the base policy** (gamma 0) over the layouts, then **every pair at every gamma**, one
   episode per layout, up to `--parallel` (default 3) simulations at a time on `--client-gpu`
   (default 0), started 90 s apart.
4. **Writes the report** by calling `analyse_proxy_pairs.py`.

Fifty layouts take roughly 40-60 minutes per batch of three simulations, so three pairs at two
gammas plus the base run take about two hours.

### Outputs

| Path | Content |
|---|---|
| `output/pps_v2_<object>_base_<tag>/` | base-policy run (`episode_results.jsonl`, one row per layout) |
| `output/pps_v2_<object>_<arm>_g04_<tag>/` | one run per pair and gamma (`g04` = 0.4, `g08` = 0.8) |
| `output/pps_v2_<object>_eval_<tag>/report.txt` | totals with 95% intervals, each arm against the base, arm against arm |
| `output/pps_v2_<object>_eval_<tag>/per_layout.csv` | one row per layout, one column per run (1 = success) |
| `output/pps_v2_<object>_eval_<tag>/evaluation.json` | exact commands, pair hashes, layouts, repository revision |
| `output/pps_v2_<object>_eval_<tag>/server.log`, `client_*.log` | logs |

`<tag>` defaults to today's date. A finished run is skipped and an interrupted run resumes, so
repeating the same command (same `--tag`) continues where it stopped.

### Reusing a base run

The base policy does not depend on the pairs, so one base run per object is enough. Pass it with
`--base-run output/pps_v2_<object>_base_<tag>` and the script skips the base run and pairs every
arm with it. The pail's is `output/pps_v2_r3_base_pail_20261111` (40 of 50).

## What makes runs comparable

Two runs can be compared layout by layout only if all of these hold. The script keeps them fixed
within one invocation; they matter when you reuse an older run.

- **Same server code path.** The base policy is served by the PPS server in mode `none`, not by the
  openpi JAX server (see the README above for the measured difference).
- **Same noise keys.** The sampler's noise is derived from the episode, environment and chunk
  indices and the `--pps-task` label. The script uses the pairs' training task
  (`SimplePnp<number><Object>`, for example `SimplePnp283UtilityjugA01`) for every run. A run made
  with another label is not paired with these.
- **Same client flags.** One environment, one run, no video, default open-loop horizon.
- **Same success rule.** Since commit `27429903` (2026-11-11) placement is judged on pose-derived
  speeds. Totals from before that date used the reported-speed rule and are about two successes in
  fifty lower for the pail base policy (38 against 40), so compare against them as totals only.

Pairing removes layout-to-layout variation, not run-to-run variation: repeating a run does not
reproduce every episode exactly (campaign B' in the 2026-11-06 results). With one episode per
layout and 50 layouts, only large differences are resolved; the report gives the exact p-value.

## Reading the report

```
base: 40/50 (80%, 95% CI 67%-89%)
T@0.4: 48/50 (96%, 95% CI 87%-99%)
T@0.4 vs base: T@0.4-only 9, base-only 1, difference +16%, exact McNemar p=0.021
```

- `X-only` counts layouts that X solved and the other run did not. Only these discordant layouts
  carry information about the difference.
- The p-value is the exact two-sided McNemar (sign) test on the discordant layouts.
- `failed layouts` lists layout indices; the line under it counts the failing condition
  (`object_grabbed` not satisfied is a grasp failure, `object_resting_in_container` a place failure).

To report on runs that already exist, or to mix runs from different days:

```bash
scripts/pps_single_stage/analyse_proxy_pairs.py --base output/pps_v2_r3_base_pail_20261111 \
    --arm T=output/pps_v2_pail_T_g04_20261113 --arm L=output/pps_v2_pail_L_g04_20261113 \
    --csv per_layout.csv
```

## Things that have gone wrong before

- **Out of memory on GPU 0.** Another user's policy server and rollout started while a job of ours
  was running there. The script waits for free memory before each start but cannot reserve it.
  Lower `--parallel` when the card is shared, and rerun the same command to resume a run that died.
- **The server does not fit next to anything else on GPU 1.** Use `--server-gpu 0` and a lower
  `--parallel` if GPU 1 is busy.
- **A port left over from another run.** The script stops if `--port` is already in use; pick another.
- **A layout set that also chose the training data.** The 50 v2 layouts of an object informed the
  range the explicit training targets were drawn from, so results on them are not a held-out test.
