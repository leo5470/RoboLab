# Pi0.5 single-stage evaluation scripts

These scripts launch the Pi0.5 server and the two requested RoboLab task
groups. The evaluation scripts are intended to run inside the RoboLab Docker
container.

Start the policy server from an environment containing an OpenPI checkout:

```bash
OPENPI_DIR=/path/to/openpi ./scripts/pi05_single_stage/start_pi05_server.sh
```

Then, inside the RoboLab container, run either task group:

```bash
./scripts/pi05_single_stage/run_container_placement.sh
./scripts/pi05_single_stage/run_surface_placement.sh
```

Configuration can be supplied through environment variables:

```bash
NUM_ENVS=8 NUM_RUNS=2 \
OUTPUT_FOLDER_NAME=my_pi05_container_run \
./scripts/pi05_single_stage/run_container_placement.sh
```

Supported evaluation variables are `REMOTE_HOST` (default `localhost`),
`REMOTE_PORT` (default `8000`), `NUM_ENVS` (default `1`), `NUM_RUNS`
(default `1`), `OUTPUT_FOLDER_NAME`, and `ISAAC_PYTHON`. Additional RoboLab
arguments can be appended directly to either evaluation command.

The server script accepts `OPENPI_DIR`, `POLICY_CONFIG`, `POLICY_DIR`, and
`XLA_MEMORY_FRACTION`. Extra arguments are forwarded to OpenPI's
`serve_policy.py`.
