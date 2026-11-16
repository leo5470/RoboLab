"""Audit Mimic source recordings and inventory task declarations without Isaac Sim."""

import argparse
import json
from pathlib import Path

from robolab.datagen.audit import audit_dataset, audit_tasks, installed_versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=Path("robolab/tasks/benchmark"))
    args = parser.parse_args()
    result = {"schema_version": 1, "installed_versions": installed_versions(),
              "dataset": audit_dataset(args.input), "task_census": audit_tasks(args.tasks)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Never replace an existing review report accidentally.
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"report": str(args.output), "ready_for_replay": result["dataset"].get("ready_for_replay", False),
                      "task_counts": result["task_census"]["counts"]}, indent=2))
    return 0 if result["dataset"].get("ready_for_replay") else 1


if __name__ == "__main__":
    raise SystemExit(main())
