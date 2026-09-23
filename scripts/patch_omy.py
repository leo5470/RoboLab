#!/usr/bin/env python3
"""Apply RoboLab's trigger correction to the pinned upstream package; safe to repeat."""

import importlib.util
from pathlib import Path
import subprocess


def main():
    spec = importlib.util.find_spec("omy_leader_isaaclab")
    if spec is None:
        raise SystemExit("Install requirements-omy.txt first, then rerun this script.")
    root = Path(spec.origin).resolve().parent.parent
    patch = (Path(__file__).resolve().parents[1] / "patches/omy-leader-trigger.patch").read_bytes()
    command = ["patch", "--batch", "--fuzz=0", "-p1", "-d", str(root)]
    def run(change, *flags):
        return subprocess.run(command + list(flags), input=change, capture_output=True)
    changes = [b"diff --git " + part for part in patch.split(b"diff --git ")[1:]]
    pending = []
    for change in changes:
        if run(change, "--dry-run", "--reverse", "--force").returncode == 0:
            continue
        check = run(change, "--dry-run", "--forward")
        if check.returncode:
            raise SystemExit("OMY source differs from the pinned patch; no files changed.\n"
                             + check.stdout.decode() + check.stderr.decode())
        pending.append(change)
    if not pending:
        print(f"OMY trigger fix already installed in {root}")
        return
    for change in pending:
        result = run(change, "--forward")
        if result.returncode:
            raise SystemExit(result.stdout.decode() + result.stderr.decode())
    print(f"Installed OMY trigger fix in {root}")


if __name__ == "__main__":
    main()
