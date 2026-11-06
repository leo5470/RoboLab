#!/usr/bin/env python3
"""Build the per-entry best-case handle area for the visibility gate (no Isaac).

    python3 tools/simple_pnp_v2/visibility_reference.py \
        --sweeps output/simple_pnp_v2_validation/pilot3/sweep \
        --output output/simple_pnp_v2_validation/inputs/visibility_reference.json

Reads visibility.py sweep reports (--yaw-offsets) and keeps, per catalog
entry, the maximum distance-normalised visible area (pixel area x median handle
depth^2) over all successful offsets of all swept layouts. An entry whose
sweeps have fewer than --min-offsets successful offsets is listed under
"insufficient" and gets no reference (sweep another layout for it).
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweeps", type=Path, nargs="+", required=True)
    parser.add_argument("--min-offsets", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    found = {}
    for d in args.sweeps:
        for f in sorted(d.glob("*Task.json")):
            row = json.loads(f.read_text())
            if row.get("status") != "sweep":
                continue
            index = int(Path(row["scene"]).stem.split("_")[1])
            item = found.setdefault(index, {"ok": 0, "best": None, "sources": []})
            item["sources"].append({"task": row["task"], "report_sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
            for m in row["sweep"]:
                if "normalized_area" not in m:
                    continue
                item["ok"] += 1
                if item["best"] is None or m["normalized_area"] > item["best"]["normalized_area"]:
                    item["best"] = {"task": row["task"], **{k: m[k] for k in (
                        "yaw_offset_deg", "normalized_area", "pixel_area", "handle_depth_m", "overlay")}}
    references, insufficient = {}, {}
    for index, item in sorted(found.items()):
        if item["ok"] < args.min_offsets or not item["best"] or item["best"]["normalized_area"] <= 0:
            insufficient[str(index)] = {"successful_offsets": item["ok"], "sources": item["sources"]}
            continue
        references[str(index)] = {"best_normalized_area": item["best"]["normalized_area"], "best": item["best"],
                                  "successful_offsets": item["ok"], "sources": item["sources"]}
    args.output.write_text(json.dumps({"min_offsets": args.min_offsets, "references": references,
                                       "insufficient": insufficient}, indent=1))
    print(f"{len(references)} references, {len(insufficient)} insufficient: {sorted(insufficient)}")


if __name__ == "__main__":
    main()
