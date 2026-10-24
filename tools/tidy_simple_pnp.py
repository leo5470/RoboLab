#!/usr/bin/env python3
"""Apply the launch-filtered benchmark's exact task exclusions (dry run by default)."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "robolab/tasks/single_pnp"
SCENE_DIR = ROOT / "assets/scenes/simple_pnp"
DEFAULT_FILTER = ROOT / "output/pi05_simple_pnp_5eps_combined_launch_filtered"


def load_json(path: Path):
    return json.loads(path.read_text())


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sync_target_asset_paths(scenes: list[dict]) -> None:
    """Record the actual target payload, including repaired model substitutions."""
    for scene in scenes:
        path = SCENE_DIR / scene["scene"]
        pattern = (r'\bdef[^\n]*"' + re.escape(scene["target"]["name"])
                   + r'"\s*\(\s*prepend payload\s*=\s*@([^@]+)@')
        match = re.search(pattern, path.read_text())
        if match is None:
            raise ValueError(f"Target payload not found in {path}")
        payload = (path.parent / match[1]).resolve()
        if not payload.is_file() or not payload.is_relative_to(ROOT):
            raise ValueError(f"Target payload is missing or outside the repository: {payload}")
        scene["target"]["usd_path"] = str(payload.relative_to(ROOT))


def filtered_viewer(text: str, scene_names: set[str], removed_entries: list[dict]) -> str:
    match = re.search(r"(?m)^const DATA = (.+);$", text)
    if match is None:
        raise ValueError("Layout viewer's embedded DATA was not found")
    data = json.loads(match.group(1))
    targets = []
    for target in data["targets"]:
        target["layouts"] = [x for x in target["layouts"] if x["scene"] in scene_names]
        if target["layouts"]:
            targets.append(target)
    data.update(targets=targets, scene_count=len(scene_names), variants_per_target=None)
    data["excluded"].extend(removed_entries)
    layouts = [x["scene"] for target in targets for x in target["layouts"]]
    if len(layouts) != len(scene_names) or set(layouts) != scene_names:
        raise ValueError("Layout viewer does not match the retained scene set")
    return text[:match.start(1)] + json.dumps(data, separators=(",", ":")) + text[match.end(1):]


def readme(tasks: list[dict], scenes: list[dict], source: dict) -> str:
    counts = Counter(x["catalog_index"] for x in scenes)
    targets = {x["catalog_index"]: x["target"]["name"] for x in scenes}
    lines = [
        "# Simple pick-and-place benchmark", "",
        f"{len(tasks):,} tasks and matching scenes across {len(counts)} catalog entries. "
        "Each task grasps one tabletop item and places it into `grey_bin`, with a "
        "50-second episode limit. Success requires a released object supported by an upright, "
        "stationary bin, with no table contact. Both object and bin must remain below "
        "0.05 m/s linear speed and 0.5 rad/s angular speed, with poses staying within "
        "1 mm and 0.5 degrees of their initial settled poses for 0.5 simulated seconds. "
        "Protrusion and stable rim-spanning placements are accepted. Termination and "
        "full subtask credit use the same condition.", "",
        "## Filtering", "",
        f"The original {source['original']['task_count']:,} layouts (50 per catalog entry) "
        f"were filtered using `{DEFAULT_FILTER.relative_to(ROOT)}`. "
        f"All {source['excluded_task_count']} excluded tasks and matching scenes are "
        "absent from this directory and the scene manifest. Exclusions cover launch-audit "
        "falls, replaced asset scales, and the explicitly excluded objects; geometry "
        "warnings alone do not exclude tasks.", "",
        "`exclusions.json` records every excluded task, its reason, and source hashes. "
        "Original task IDs and layout suffixes are preserved, so some objects have fewer "
        "than 50 layouts. Count catalog entries by index: names are not necessarily unique.", "",
        "## Use", "",
        "Run from the RoboLab repository root with an already-running pi05 server:", "",
        "```bash",
        ".venv/bin/python policies/pi0_family/run.py \\",
        "    --policy pi05 --headless --device cuda:0 --remote-port 8000 \\",
        "    --task-dirs single_pnp --task SimplePnp000BlueBlock00Task \\",
        "    --num-envs 1 --num-runs 1 --video-mode none \\",
        "    --output-folder-name simple_pnp_smoke_test",
        "```", "",
        "Tasks register automatically at startup. Omit `--task` to select the whole set; "
        "use explicit task batches to limit registration time and memory. No additional "
        "exclusion filtering is needed for this committed task set.", "",
        "Run `git lfs pull` after cloning to obtain the official assets and the five "
        "repaired models in `assets/objects/repaired/`. Small generated scene layers "
        "are stored as ordinary text files. Existing RoboLab/IsaacLab dependencies "
        "and the policy server must be installed separately. Repaired models use "
        "relative texture paths so the repository can be installed at any location.", "",
        "Open `assets/scenes/simple_pnp/layout_viewer.html` to inspect the retained "
        "layouts without simulation. Both manifests describe only the retained set.", "",
        "## Retained layouts", "",
        "| Catalog index | Object | Layouts |",
        "| --- | --- | ---: |",
    ]
    lines.extend(f"| {index:03d} | {targets[index]} | {counts[index]} |" for index in sorted(counts))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter-dir", type=Path, default=DEFAULT_FILTER)
    parser.add_argument("--apply", action="store_true", help="Archive excluded files and update manifests")
    args = parser.parse_args()
    source_dir = args.filter_dir.resolve()
    source_path = source_dir / "filter_manifest.json"
    source = load_json(source_path)
    excluded = set(source["excluded_tasks"])
    if len(excluded) != source["excluded_task_count"]:
        raise ValueError("Source exclusion count is inconsistent")
    task_path, scene_path = TASK_DIR / "manifest.json", SCENE_DIR / "manifest.json"
    task_manifest, scene_manifest = load_json(task_path), load_json(scene_path)
    tasks = task_manifest["tasks"]
    kept = [task for task in tasks if task["class_name"] not in excluded]
    removed = [task for task in tasks if task["class_name"] in excluded]
    names = {task["class_name"] for task in kept}
    if len(names) != len(kept) or len(kept) != source["filtered"]["task_count"]:
        raise ValueError("Retained task count is inconsistent with filtered results")
    # Validate against the actual recorded episodes, not just the summary counts.
    recorded = Counter()
    with (source_dir / "episode_results.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            recorded[record["task_name"]] += 1
    if set(recorded) != names or sum(recorded.values()) != source["filtered"]["episode_count"]:
        raise ValueError("Retained tasks do not match the filtered episode results")
    if not removed:
        if set(load_json(TASK_DIR / "exclusions.json")["excluded_tasks"]) != excluded:
            raise ValueError("Already-filtered task set has different exclusion provenance")
        print(f"Already filtered: {len(kept)} tasks; no changes made")
        return
    if {task["class_name"] for task in removed} != excluded:
        raise ValueError("Some excluded task definitions are missing")
    if {p.name for p in TASK_DIR.glob("*_task.py")} != {task["task"] for task in tasks}:
        raise ValueError("Task directory differs from its manifest")
    if {p.name for p in SCENE_DIR.glob("*.usda")} != {s["scene"] for s in scene_manifest["scenes"]}:
        raise ValueError("Scene directory differs from its manifest")
    by_scene = {s["scene"]: s for s in scene_manifest["scenes"]}
    if len(by_scene) != len(tasks):
        raise ValueError("Tasks and scenes are not one-to-one")
    kept_scenes = [by_scene[Path(task["scene"]).name] for task in kept]
    sync_target_asset_paths(kept_scenes)
    kept_indices = {s["catalog_index"] for s in kept_scenes}
    if len(kept_indices) != source["filtered"]["object_count"]:
        raise ValueError("Retained catalog count is inconsistent")
    with (source_dir / "excluded_tasks.csv").open(newline="") as stream:
        reasons = {row["task"]: row for row in csv.DictReader(stream)}
    if set(reasons) != excluded:
        raise ValueError("Exclusion CSV differs from the source manifest")
    excluded_records, removed_entries = [], {}
    for task in removed:
        scene = by_scene[Path(task["scene"]).name]
        reason = reasons[task["class_name"]]["reasons"]
        excluded_records.append({**task, "catalog_index": scene["catalog_index"], "reasons": reason})
        if scene["catalog_index"] not in kept_indices:
            removed_entries[scene["catalog_index"]] = {
                "catalog_index": scene["catalog_index"], "name": task["target"],
                "usd_path": scene["target"]["usd_path"], "dims": scene["target"]["dims"],
                "class": "", "reason": reason,
            }
    viewer_path = SCENE_DIR / "layout_viewer.html"
    viewer = filtered_viewer(viewer_path.read_text(), {s["scene"] for s in kept_scenes}, list(removed_entries.values()))
    provenance = {
        "source": str(source_dir.relative_to(ROOT)) if source_dir.is_relative_to(ROOT) else source_dir.name,
        "filter_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "episode_results_sha256": hashlib.sha256((source_dir / "episode_results.jsonl").read_bytes()).hexdigest(),
        "exclusion_rule": source["exclusion_rule"],
        "original_task_count": source["original"]["task_count"],
        "excluded_task_count": len(excluded), "retained_task_count": len(kept),
    }
    old_exclusions = load_json(TASK_DIR / "exclusions.json")
    exclusions = {
        "target_patterns": sorted(set(old_exclusions["target_patterns"]) | set(source["excluded_targets"]) | {x["name"] for x in removed_entries.values()}),
        "reason": source["exclusion_rule"], "provenance": provenance,
        "excluded_tasks": sorted(excluded), "tasks": excluded_records,
    }
    task_manifest.update(task_count=len(kept), tasks=kept, filtering=provenance)
    counts = Counter(s["catalog_index"] for s in kept_scenes)
    scene_manifest.update(
        description="Launch-filtered two-object tabletop PnP layouts; original IDs preserved",
        target_count=len(kept_indices), scene_count=len(kept_scenes), variants_per_target=None,
        generated_variants_per_target=scene_manifest["variants_per_target"],
        variants_per_target_range=[min(counts.values()), max(counts.values())],
        scenes=kept_scenes, filtering=provenance,
    )
    scene_manifest["excluded_targets"].extend(removed_entries.values())
    print(f"Keep {len(kept)} tasks/scenes across {len(kept_indices)} catalog entries; archive {len(removed)} pairs")
    if not args.apply:
        print("Dry run; pass --apply to perform cleanup")
        return
    # Keep all removed definitions and original metadata recoverable outside discovery.
    backup = Path(tempfile.mkdtemp(prefix="simple_pnp_cleanup_", dir=ROOT / "output"))
    for path in (task_path, scene_path, TASK_DIR / "exclusions.json", TASK_DIR / "README.md", viewer_path):
        destination = backup / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    for task in removed:
        for path in (TASK_DIR / task["task"], ROOT / "assets/scenes" / task["scene"]):
            destination = backup / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), destination)
    save_json(task_path, task_manifest)
    save_json(scene_path, scene_manifest)
    save_json(TASK_DIR / "exclusions.json", exclusions)
    viewer_path.write_text(viewer)
    (TASK_DIR / "README.md").write_text(readme(kept, kept_scenes, source))
    print(f"Archived excluded files and original metadata in {backup}")


if __name__ == "__main__":
    main()
