#!/usr/bin/env python3
"""Encode native-15 success verification videos from a processed dataset on CPU."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="dataset.json or its directory")
    parser.add_argument("--output", type=Path, help="New output directory; default: DATASET/verification_videos")
    args = parser.parse_args()
    from robolab.core.omy_postprocess.videos import export_success_videos

    export_success_videos(args.dataset, args.output)


if __name__ == "__main__":
    main()
