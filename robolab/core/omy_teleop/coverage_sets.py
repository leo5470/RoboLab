"""Nested coverage datasets of 50 / 100 / 200 demonstrations for one collected object.

Each set keeps the recipe of the T arm in ``layout_arms`` (the humans, then
generated demonstrations split evenly between local perturbations at 2:1:1
near / moderate / wide and explicit targets across the evaluation range, one
per distinct layout) and contains the next smaller set, so a larger set only
adds demonstrations.

Rendering is the expensive step in time and disk, so the sets are rendered
once, in disjoint chunks of at most ``CHUNK`` new demonstrations: each chunk
is a small pool with its own training file and PPS export, and a set's PPS
file is assembled from the chunks it contains (``merge_pps``). No simulator
imports here.
"""

import json
import math
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

import h5py

from robolab.core.omy_postprocess import pps_hdf5
from robolab.core.omy_postprocess.io import fsync_directory, publish_directory, sha256_file

from . import layout_arms
from .final_dataset import choose, parent_key

SIZES = (50, 100, 200)
CHUNK = 50
SURPLUS = 1.5  # candidates per selected demonstration of the largest set, per stratum


def set_quotas(size, humans=10):
    """Stratum counts of a set: T's 10 / 5 / 5 / 20 at 50 with ten humans, scaled."""
    generated = size - humans
    if generated < 0:
        raise ValueError(f"A set of {size} cannot hold {humans} humans")
    explicit = generated // 2
    local = generated - explicit
    near = (local + 1) // 2
    moderate = (local - near + 1) // 2
    return {"collected": humans, "near": near, "moderate": moderate, "wide": local - near - moderate,
            "explicit": explicit}


def candidate_needs(largest, humans=10, surplus=SURPLUS):
    """Local candidates (a multiple of four, split 2:1:1 by the pipeline) and accepted explicit layouts to seek."""
    quotas = set_quotas(largest, humans)
    local = quotas["near"] + quotas["moderate"] + quotas["wide"]
    return 4 * math.ceil(surplus * local / 4), math.ceil(surplus * quotas["explicit"])


def pick_humans(rows, preferred=(), limit=10):
    """Fingerprints of the humans every set keeps: ``preferred`` if all are present, else farthest-first."""
    collected = [r for r in rows if r["kind"] == "collected"]
    present = {r["fingerprint"] for r in collected}
    if preferred and set(preferred) <= present:
        return list(preferred)[:limit], "preferred"
    count = min(limit, len(collected))
    if count == 0:
        return [], "none"
    selected, _ = choose(collected, count, count)
    return [r["fingerprint"] for r in selected], "farthest_first"


def select_nested(rows, tiers, qualified, layouts, names, sizes=SIZES, *, base=(), preferred_humans=()):
    """Per size, the selected rows (each containing the previous size) or the reason it cannot be filled.

    Generated candidates starting within ``layout_arms.COINCIDENT_M`` of an
    evaluation layout are dropped; humans are kept and only reported. ``base``
    lists the fingerprints of an already published smaller set to build on.
    """
    eligible = [r for r in rows if r["kind"] == "collected"
                or (r["kind"] == "generated" and parent_key(r) in qualified)]
    hits = layout_arms.coincident(eligible, layouts, names)
    kept = [r for i, r in enumerate(eligible) if i not in hits or r["kind"] == "collected"]
    strata = [layout_arms.stratum_of(r, tiers) for r in kept]
    by_fingerprint = {r["fingerprint"]: (r, s) for r, s in zip(kept, strata)}
    if base:
        missing = [f for f in base if f not in by_fingerprint]
        if missing:
            raise ValueError(f"{len(missing)} demonstrations of the base set are not among the candidates")
        humans, rule = [f for f in base if by_fingerprint[f][1] == "collected"], "base_set"
    else:
        humans, rule = pick_humans(kept, preferred_humans)
    previous, out = list(base), {}
    for size in sorted(sizes):
        quotas = set_quotas(size, len(humans))
        try:
            selected, normalization = layout_arms.choose_strata(
                kept, quotas, strata, distinct={"explicit": lambda r: r["generation_target"]},
                preselected=previous, humans=humans)
        except ValueError as error:
            out[size] = {"error": str(error), "quotas": quotas}
            continue
        out[size] = {"rows": selected, "normalization": normalization, "quotas": quotas,
                     "counts": dict(Counter(by_fingerprint[r["fingerprint"]][1] for r in selected)),
                     "parents": dict(Counter(f"{Path(k[0]).parent.name}:{k[1]}" for k in
                                             (parent_key(r) for r in selected if r["kind"] == "generated")))}
        previous = [r["fingerprint"] for r in selected]
    report = {"available_by_stratum": dict(Counter(strata)), "humans_rule": rule, "humans": len(humans),
              "dropped_near_evaluation_layouts": sum(eligible[i]["kind"] == "generated" for i in hits),
              "dropped": [{"file": eligible[i]["file"], "episode": eligible[i]["episode"], "layout": layout}
                          for i, layout in sorted(hits.items()) if eligible[i]["kind"] == "generated"],
              "humans_near_evaluation_layouts": [{"episode": eligible[i]["episode"], "task": eligible[i]["task"],
                                                  "layout": layout} for i, layout in sorted(hits.items())
                                                 if eligible[i]["kind"] == "collected"],
              "unfilled": {str(size): value["error"] for size, value in out.items() if "error" in value}}
    return out, report


def render_chunks(selected, base=(), chunk=CHUNK):
    """Disjoint row lists to render, in nesting order: each set's additions, at most ``chunk`` rows each.

    ``selected`` maps size -> rows for the sizes that were filled; ``base``
    fingerprints are already rendered elsewhere and are left out.
    """
    seen, chunks = set(base), []
    for size in sorted(selected):
        new = [r for r in selected[size] if r["fingerprint"] not in seen]
        seen.update(r["fingerprint"] for r in new)
        new.sort(key=lambda r: (r["kind"] != "collected", r["file"], r["episode"]))
        chunks += [new[i:i + chunk] for i in range(0, len(new), chunk)]
    return chunks


def merge_pps(pool_dir, part_dirs, output_root, name=None):
    """Assemble OUTPUT_ROOT/SESSION/{pps.hdf5, manifest.json, validation.json} for a pool from chunk exports.

    ``part_dirs`` are ``pps_hdf5.export_session`` outputs whose episodes are
    disjoint and together hold the pool's episodes. Episode groups are copied
    unchanged and numbered in the pool's order, as a direct export would.
    """
    pool_dir, output_root = Path(pool_dir), Path(output_root)
    pool = json.loads((pool_dir / "dataset.json").read_text())
    target = output_root / (name or pool["session"])
    if target.exists():
        raise FileExistsError(f"Refusing existing output: {target}")
    parts, where = [], {}
    for directory in map(Path, part_dirs):
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("format") != pps_hdf5.FORMAT or sha256_file(directory / pps_hdf5.FILE_NAME) != manifest["sha256"]:
            raise ValueError(f"PPS part {directory} does not match its manifest")
        parts.append((directory, manifest))
        for episode in manifest["episodes"]:
            if episode["fingerprint"] in where:
                raise ValueError(f"Episode {episode['fingerprint'][:12]} is in two parts")
            where[episode["fingerprint"]] = (len(parts) - 1, episode)
    wanted = [r["fingerprint"] for r in pool["episodes"]]
    if set(wanted) - set(where):
        raise ValueError(f"{len(set(wanted) - set(where))} pool episodes are in no part")
    if len({(m["task"], m["prompt"]) for _, m in parts}) != 1:
        raise ValueError("Parts disagree on task or prompt")
    task, prompt = parts[0][1]["task"], parts[0][1]["prompt"]
    output_root.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=output_root))
    episodes = []
    sources = [h5py.File(directory / pps_hdf5.FILE_NAME, "r") for directory, _ in parts]
    try:
        with h5py.File(partial / pps_hdf5.FILE_NAME, "x") as out:
            out.attrs.update(fps=pps_hdf5.FPS, task=task, prompt=prompt, training_ready=False,
                             format=pps_hdf5.FORMAT, wire_convention=pps_hdf5.WIRE_CONVENTION, session=pool["session"])
            data = out.create_group("data")
            for index, row in enumerate(pool["episodes"]):
                part, episode = where[row["fingerprint"]]
                group = f"episode_{index:03d}"
                sources[part].copy(sources[part][episode["group"]], data, group)
                if data[group].attrs["fingerprint"] != row["fingerprint"]:
                    raise ValueError(f"{episode['group']} of part {part} is not {row['fingerprint'][:12]}")
                episodes.append({**episode, "group": f"data/{group}"})
    finally:
        for source in sources:
            source.close()
    validation = pps_hdf5.verify_file(partial / pps_hdf5.FILE_NAME)
    (partial / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    used = sorted({where[f][0] for f in wanted})
    manifest = {
        "schema_version": 1, "format": pps_hdf5.FORMAT, "wire_convention": pps_hdf5.WIRE_CONVENTION,
        "file": pps_hdf5.FILE_NAME, "sha256": sha256_file(partial / pps_hdf5.FILE_NAME), "training_ready": False,
        "session": pool["session"], "family": pool["family"], "task": task, "prompt": prompt,
        "pool_manifest_sha256": sha256_file(pool_dir / "dataset.json"),
        "assembled_from": [{"directory": str(parts[i][0]), "sha256": parts[i][1]["sha256"],
                            "pool_manifest_sha256": parts[i][1]["pool_manifest_sha256"],
                            "training_manifest_sha256": parts[i][1].get("training_manifest_sha256"),
                            "training_file_sha256": parts[i][1].get("training_file_sha256"),
                            "episodes": sum(where[f][0] == i for f in wanted)} for i in used],
        **pps_hdf5.revision(), "created_unix_s": time.time(), "selection_rule": pool.get("method"),
        "episodes": episodes, "conventions": parts[0][1].get("conventions"), "held_joint": validation["held_joint"],
    }
    (partial / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in partial.iterdir():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    fsync_directory(partial)
    publish_directory(partial, target)
    return target


def verify_merged(directory, pool_dir):
    """A published assembled export still matches its pool and file, episode for episode."""
    directory, pool_dir = Path(directory), Path(pool_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    pool = json.loads((pool_dir / "dataset.json").read_text())
    if (manifest.get("format") != pps_hdf5.FORMAT
            or manifest.get("pool_manifest_sha256") != sha256_file(pool_dir / "dataset.json")
            or manifest.get("sha256") != sha256_file(directory / pps_hdf5.FILE_NAME)
            or [e["fingerprint"] for e in manifest["episodes"]] != [r["fingerprint"] for r in pool["episodes"]]):
        raise ValueError(f"PPS export {directory} no longer matches its pool or file")
    with h5py.File(directory / pps_hdf5.FILE_NAME, "r") as stream:
        if [stream[e["group"]].attrs["fingerprint"] for e in manifest["episodes"]] != \
                [r["fingerprint"] for r in pool["episodes"]]:
            raise ValueError(f"PPS file {directory} holds other episodes than its pool")
    return manifest
