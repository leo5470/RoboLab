#!/usr/bin/env python3
"""Plan or apply a reversible split of a finalized batch by collection session."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robolab.core.omy_postprocess.split_batch import split_batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, default=Path("output/demo_library"))
    parser.add_argument("--apply", action="store_true", help="Apply after stopping the website and video service")
    args = parser.parse_args()
    report = split_batch(args.run, args.state_dir, apply=args.apply)
    print(json.dumps({k: v for k, v in report.items() if k not in ("payloads", "catalog_ids")}, indent=2))
