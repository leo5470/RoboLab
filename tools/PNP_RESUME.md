# Resuming the five-episode PnP run

Preview the original shard assignments and isolated services:

```sh
.venv/bin/python tools/resume_pnp.py --dry-run
```

Resume with `.venv/bin/python tools/resume_pnp.py`. This uses the saved JSONL
results and current exclusions, with five environments and no videos. Each
evaluator exits after 50 tasks; its worker then starts a fresh Isaac process for
the next batch. An unexpected exit stops the worker and remains visible in the
service journal. Completed batches are retained for the next resume.

Each policy server and evaluator runs in its own systemd user service, outside
the launching terminal's scope. Servers and evaluators have a 64 GiB RAM limit
each. These are per-service ceilings; the host still has 125 GiB total, so
simultaneous growth can still cause global memory pressure. If a service hits
its limit, inspect its journal and memory telemetry before adjusting it.
Other host workloads can still cause global memory pressure.

Monitor with `systemctl --user status 'robolab-pnp-*'` and
`journalctl --user -u robolab-pnp-server-gpu0.service`. Server output is in the
journal; evaluator output remains in `output/pi05_fresh_5eps_logs/eval_gpu*.log`.
Workers record their service's RAM, swap and OOM counters every 30 seconds in
`output/pi05_fresh_5eps_logs/memory_gpu*.jsonl`. Server memory is available through
`systemctl --user show robolab-pnp-server-gpu0 -p MemoryCurrent -p MemoryPeak`.

Stop all four services with `systemctl --user stop 'robolab-pnp-*'`.
Avoid the earlier direct Python/tmux launch commands: starting a new process
session does not move a process out of its original systemd memory scope.

The client lifecycle regression tests run without Isaac or GPU models:

```sh
.venv/bin/python -m pytest --confcutdir=tests/unit tests/unit/test_eval_cleanup.py tests/unit/test_pnp_resume.py -q
```

The unclosed WebSockets are confirmed and fixed. The earlier 113.9 GiB scope
peak cannot be attributed entirely to those connections from the retained data;
batch recycling bounds process lifetime while telemetry helps diagnose any
remaining Isaac or server memory growth.
