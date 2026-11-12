"""Matched L/T datasets for the layout-coverage experiment (docs/omy_mimicgen_diversity_plan_20261108.md).

Both arms keep the same ten human recordings and add forty generated
demonstrations from the same qualified parents and generator profile:

- L (local): 20 / 10 / 10 near / moderate / wide perturbations of the humans' layouts.
- T (target): 10 / 5 / 5 local perturbations plus 20 explicit targets, one per distinct layout.

Local candidates come from the single-source pipeline (``Pipeline`` with a
candidate factor of three, i.e. 60 / 30 / 30). Explicit candidates come from
``ExplicitTargets``: one job per target, using the best-ranked feasible source
from ``path_screen.rank``, falling back to the next one when a job yields no
accepted demonstration. Selection (``choose_strata``) enforces each arm's
stratum counts exactly, farthest-first within each stratum, balanced across
parents; no selected demonstration may start within ``COINCIDENT_M`` of an
evaluation layout.
"""

import threading
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from .diversity import feature_matrix

from .dataset_paths import job_recording
from .final_dataset import gather_candidates, parent_cap, parent_key, read_json
from .pipeline import LIMITS, Interrupted, atomic_json, process_job, source_key, tier_of

ARMS = {
    "L": {"collected": 10, "near": 20, "moderate": 10, "wide": 10},
    "T": {"collected": 10, "near": 10, "moderate": 5, "wide": 5, "explicit": 20},
}
STRATA = ("near", "moderate", "wide", "explicit")
EXPLICIT_PREFIX = "tx"
COINCIDENT_M = 0.02  # joint-XY distance below which a start counts as an evaluation layout


def explicit_need(arms=ARMS, factor=3):
    """Distinct accepted explicit layouts to seek before selection (the plan's spatial surplus)."""
    return factor * max(arm.get("explicit", 0) for arm in arms.values())


def start_poses(file, episode, names):
    """Recorded start (initial-state) root poses of the named rigid objects."""
    with h5py.File(file, "r") as stream:
        group = stream["data"][episode]["initial_state"]["rigid_object"]
        return {name: np.asarray(group[name]["root_pose"][0], dtype=float) for name in names}


def joint_xy(first, second, names):
    return max(float(np.linalg.norm(np.asarray(first[n][:2]) - np.asarray(second[n][:2]))) for n in names)


def coincident(rows, layouts, names, threshold=COINCIDENT_M):
    """Rows whose recorded start lies within ``threshold`` joint-XY of an evaluation layout: {index: layout id}."""
    hits = {}
    for index, row in enumerate(rows):
        start = row.get("start_poses") or start_poses(row["file"], row["episode"], names)
        for layout in layouts:
            if joint_xy(start, layout["poses"], names) < threshold:
                hits[index] = layout["id"]
                break
    return hits


def stratum_of(row, job_tiers):
    """Stratum of a candidate row: collected, a local tier, explicit, or None (unknown)."""
    if row["kind"] == "collected":
        return "collected"
    if row.get("generation_target"):
        return "explicit"
    tier = job_tiers.get(str(Path(row["file"]).resolve()))
    return tier if tier in STRATA else tier_of(row.get("generation_ranges"))


def choose_strata(rows, quotas, strata, *, distinct=None):
    """Select exactly ``quotas[stratum]`` rows per stratum.

    Humans (``collected``) are taken first; each generated stratum is then
    filled farthest-first in the shared descriptor space, against everything
    already selected, with its parents balanced as in ``final_dataset.choose``.
    ``distinct`` maps a stratum to a row key that may appear only once in it
    (explicit targets: one demonstration per layout).
    """
    distinct = distinct or {}
    by = {s: [i for i, r in enumerate(rows) if strata[i] == s] for s in quotas}
    for s, n in quotas.items():
        available = len({distinct[s](rows[i]) for i in by[s]}) if s in distinct else len(by[s])
        if available < n:
            raise ValueError(f"Stratum {s!r} needs {n}; {available} available")
    features, normalization = feature_matrix([r["blocks"] for r in rows])
    chosen, distances = [], np.full(len(rows), np.inf)

    def add(index):
        chosen.append(index)
        np.minimum(distances, np.linalg.norm(features - features[index], axis=1), out=distances)

    for s in ("collected", *[s for s in STRATA if s in quotas]):
        pool, n = by.get(s, []), quotas.get(s, 0)
        if s == "collected":
            for index in sorted(pool, key=lambda i: (rows[i]["task"], rows[i]["episode"]))[:n]:
                add(index)
            continue
        parents = {i: parent_key(rows[i]) for i in pool}
        cap = parent_cap(Counter(parents.values()), n) if n else 0
        used, keys = Counter(), set()
        for _ in range(n):
            eligible = [i for i in pool if i not in chosen and used[parents[i]] < cap
                        and (s not in distinct or distinct[s](rows[i]) not in keys)]
            uncovered = [i for i in eligible if used[parents[i]] == 0]
            eligible = uncovered or eligible
            if not eligible:
                raise ValueError(f"Stratum {s!r}: no eligible candidate left after {used.total()} of {n}")
            if not chosen:
                center = np.median(features[pool], axis=0)
                index = min(eligible, key=lambda i: (float(np.linalg.norm(features[i] - center)), i))
            else:
                index = min(eligible, key=lambda i: (-float(distances[i]), i))
            add(index)
            used[parents[index]] += 1
            if s in distinct:
                keys.add(distinct[s](rows[index]))
    return [rows[i] for i in chosen], normalization


class ExplicitTargets:
    """Generate at explicit targets with screen-ranked sources, resumably, on the given GPUs.

    ``ranking`` is ``path_screen.rank`` output; its source keys are
    ``"<task>:<episode>"``. Only qualified sources are used, at most
    ``max_sources`` per target, and a target is done once one job is accepted.
    """

    def __init__(self, work, manifest_path, ranking, gpus, generator_profile, *, need, max_sources=3,
                 attempts=2, log=print):
        self.work, self.manifest_path, self.gpus = work, str(Path(manifest_path).resolve()), gpus
        self.profile, self.need, self.max_sources, self.attempts, self.log = (
            generator_profile, need, max_sources, attempts, log)
        self.targets = [t["id"] for t in read_json(manifest_path)["targets"]]
        by_key = {f"{Path(s['input']).parent.name}:{s['episode']}": s for s in work.sources
                  if source_key(s) in work.qualified}
        self.ranked = {row["id"]: [by_key[k] for k in row["ranked_sources"] if k in by_key][:max_sources]
                       for row in ranking["targets"]}
        self.lock, self.stop = threading.Condition(), threading.Event()
        self.results = {}  # target id -> list of results
        for path in sorted((work.folder / "jobs").glob(f"{EXPLICIT_PREFIX}*/result.json")):
            result = read_json(path)
            self.results.setdefault(result["target_id"], []).append(result)
        self.active = set()

    def accepted(self):
        return sorted(t for t, results in self.results.items()
                      if any(r["accepted"] and not r["errors"] for r in results))

    def next_job(self):
        if len(self.accepted()) + len(self.active) >= self.need:
            return None
        for index, target in enumerate(self.targets):
            if target in self.active or target in self.accepted():
                continue
            tried = {source_key(r) for r in self.results.get(target, [])}
            remaining = [s for s in self.ranked.get(target, []) if source_key(s) not in tried]
            if not remaining:
                continue
            source = remaining[0]
            attempt = len(tried)
            job_id = f"{EXPLICIT_PREFIX}{index:05d}_{attempt}_{target}_{source['task']}"
            return {"id": job_id, "index": index, "family": self.work.family, "task": source["task"],
                    "input": source["input"], "episode": source["episode"], "stage": "generate",
                    "tier": "explicit", "num_demos": 1, "max_attempts": self.attempts,
                    "seed": 4_000_000 + index * 1000, "ranges": {}, "target_manifest": self.manifest_path,
                    "target_id": target, "source_rank": attempt, "generator_profile": self.profile,
                    "output_dir": str(Path(source["task"]) / job_id / Path(source["input"]).name)}
        return None

    def worker(self, gpu):
        while not self.stop.is_set():
            with self.lock:
                job = self.next_job()
                if job is None:
                    return
                self.active.add(job["target_id"])
            try:
                result = process_job(self.work.folder, job, gpu, self.stop)
            except Interrupted:
                return
            finally:
                with self.lock:
                    self.active.discard(job["target_id"])
            with self.lock:
                self.results.setdefault(job["target_id"], []).append(result)
                self.log(f"{job['id']} on GPU {gpu}: {len(result['accepted'])} accepted / {result['attempts']} "
                         f"attempts; {len(self.accepted())}/{self.need} targets", flush=True)

    def run(self):
        threads = [threading.Thread(target=self.worker, args=(gpu,), daemon=True) for gpu in self.gpus]
        for thread in threads:
            thread.start()
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            self.stop.set()
            raise
        return self.report()

    def report(self):
        tried = sum(1 for t in self.targets if self.results.get(t))
        unranked = [t for t in self.targets if not self.ranked.get(t)]
        return {"need": self.need, "accepted_targets": len(self.accepted()), "targets_tried": tried,
                "targets_without_feasible_source": len(unranked),
                "jobs": sum(len(v) for v in self.results.values()),
                "accepted_by_source_rank": dict(Counter(
                    r["source_rank"] for v in self.results.values() for r in v if r["accepted"] and not r["errors"]))}


def candidate_rows(roots, work):
    """Quality-passing rows of the session (humans and every generated job) and each job recording's tier."""
    from robolab.core.omy_postprocess.automatic_dataset import discover_recordings

    recordings, family, _ = discover_recordings(roots["collected"], roots["generated"], work.name)
    rows, rejected = gather_candidates(recordings, family, LIMITS)
    tiers = {}
    for path in sorted((work.folder / "jobs").glob("*/result.json")):
        result = read_json(path)
        if result.get("stage") == "generate":
            tiers[str(job_recording(work.folder, result).resolve())] = result.get("tier")
    return rows, rejected, tiers


def select_arms(rows, tiers, qualified, layouts, names, arms=ARMS):
    """Per arm, the selected rows and a report; rows near evaluation layouts are dropped first."""
    eligible = [r for r in rows if r["kind"] == "collected"
                or (r["kind"] == "generated" and parent_key(r) in qualified)]
    hits = coincident(eligible, layouts, names)
    kept = [r for i, r in enumerate(eligible) if i not in hits]
    strata = [stratum_of(r, tiers) for r in kept]
    available = Counter(strata)
    out = {}
    for arm, quotas in arms.items():
        selected, normalization = choose_strata(kept, quotas, strata,
                                                distinct={"explicit": lambda r: r["generation_target"]})
        out[arm] = {"rows": selected, "normalization": normalization,
                    "counts": dict(Counter(stratum_of(r, tiers) for r in selected)),
                    "parents": dict(Counter(f"{Path(k[0]).parent.name}:{k[1]}" for k in
                                            (parent_key(r) for r in selected if r["kind"] == "generated")))}
    shared = {r["fingerprint"] for r in out["L"]["rows"]} & {r["fingerprint"] for r in out["T"]["rows"]}
    report = {"available_by_stratum": dict(available), "dropped_near_evaluation_layouts": len(hits),
              "dropped": [{"file": eligible[i]["file"], "episode": eligible[i]["episode"], "layout": layout}
                          for i, layout in sorted(hits.items())],
              "shared_between_arms": len(shared)}
    return out, report
