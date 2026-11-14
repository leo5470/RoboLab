"""Nested coverage sets: quotas, nesting, render chunks, PPS assembly and derived target profiles (no Isaac Sim)."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pytest

from robolab.core.omy_postprocess import pps_hdf5
from robolab.core.omy_teleop import coverage_profile, coverage_sets
from robolab.core.omy_teleop.coverage_sets import candidate_needs, merge_pps, render_chunks, select_nested, set_quotas, verify_merged
from robolab.core.omy_teleop.final_dataset import parent_key

NAMES = ("obj", "bin")


def _row(kind, i, parent=None, target=None, start=(0.5, 0.2, 0.5, -0.2)):
    rng = np.random.default_rng(i)
    row = {"kind": kind, "task": f"T{i % 7}", "episode": f"demo_{i}", "file": f"/runs/{kind}/{i}/omy_demos.hdf5",
           "fingerprint": f"f{kind}{i}", "blocks": [rng.normal(size=4), rng.normal(size=3), rng.normal(size=2), rng.normal(size=2)],
           "generation_target": target, "generation_ranges": None,
           "start_poses": {"obj": [start[0], start[1], 0, 1, 0, 0, 0], "bin": [start[2], start[3], 0, 1, 0, 0, 0]}}
    if kind == "generated":
        row["generation_source"] = {"file": f"/collected/P{parent}/part_000/omy_demos.hdf5", "episode": "demo_0"}
    return row


def _pool(humans=12, near=72, explicit=143, parents=9):
    rows, tiers, i = [_row("collected", i) for i in range(humans)], {}, 100
    for tier, n in (("near", near), ("moderate", near // 2), ("wide", near // 2)):
        for k in range(n):
            row = _row("generated", i, parent=k % parents)
            tiers[str(Path(row["file"]).resolve())] = tier
            rows.append(row)
            i += 1
    for k in range(explicit):
        rows.append(_row("generated", i, parent=k % parents, target=f"t{k:03d}"))
        i += 1
    return rows, tiers


def test_quotas_scale_the_t_recipe_and_nest():
    assert set_quotas(50) == {"collected": 10, "near": 10, "moderate": 5, "wide": 5, "explicit": 20}
    assert set_quotas(100) == {"collected": 10, "near": 23, "moderate": 11, "wide": 11, "explicit": 45}
    assert set_quotas(200) == {"collected": 10, "near": 48, "moderate": 24, "wide": 23, "explicit": 95}
    assert all(sum(set_quotas(n, h).values()) == n for n in (50, 100, 200) for h in (10, 7, 0))
    assert candidate_needs(200) == (144, 143)


def test_each_set_contains_the_smaller_one_with_exact_strata_and_fixed_humans():
    rows, tiers = _pool()
    qualified = {parent_key(r) for r in rows if r["kind"] == "generated"}
    preferred = [f"fcollected{i}" for i in (11, 3, 5, 7, 0, 1, 2, 4, 6, 8)]
    selected, report = select_nested(rows, tiers, qualified, [], NAMES, preferred_humans=preferred)
    assert report["humans_rule"] == "preferred" and not report["unfilled"]
    previous = set()
    for size in (50, 100, 200):
        chosen = selected[size]["rows"]
        fingerprints = {r["fingerprint"] for r in chosen}
        assert len(fingerprints) == size and previous <= fingerprints
        assert selected[size]["counts"] == {k: v for k, v in set_quotas(size).items()}
        assert {r["fingerprint"] for r in chosen if r["kind"] == "collected"} == set(preferred)
        targets = [r["generation_target"] for r in chosen if r["generation_target"]]
        assert len(targets) == len(set(targets))
        previous = fingerprints
    chunks = render_chunks({size: value["rows"] for size, value in selected.items()})
    assert [len(c) for c in chunks] == [50, 50, 50, 50]
    assert {r["fingerprint"] for r in chunks[0]} == {r["fingerprint"] for r in selected[50]["rows"]}
    assert len({r["fingerprint"] for c in chunks for r in c}) == 200
    # Building on a published base: only the additions are rendered.
    base = [r["fingerprint"] for r in selected[50]["rows"]]
    again, report = select_nested(rows, tiers, qualified, [], NAMES, [100], base=base)
    assert report["humans_rule"] == "base_set" and set(base) <= {r["fingerprint"] for r in again[100]["rows"]}
    assert [len(c) for c in render_chunks({100: again[100]["rows"]}, base)] == [50]


def test_a_size_that_cannot_be_filled_is_reported_and_generated_starts_at_evaluation_layouts_dropped():
    rows, tiers = _pool(humans=10, explicit=50)
    rows.append(_row("generated", 999, parent=0, target="t999", start=(0.3, 0.1, 0.6, -0.2)))
    rows[0]["start_poses"] = {"obj": [0.3, 0.1, 0, 1, 0, 0, 0], "bin": [0.6, -0.2, 0, 1, 0, 0, 0]}  # a human there stays
    qualified = {parent_key(r) for r in rows if r["kind"] == "generated"}
    layouts = [{"id": "v2_00", "poses": {"obj": [0.3, 0.1, 0], "bin": [0.6, -0.2, 0]}}]
    selected, report = select_nested(rows, tiers, qualified, layouts, NAMES)
    assert "rows" in selected[50] and "rows" in selected[100] and "explicit" in selected[200]["error"]
    assert report["unfilled"] == {"200": selected[200]["error"]} and report["dropped_near_evaluation_layouts"] == 1
    assert report["humans"] == 10 and len(report["humans_near_evaluation_layouts"]) == 1
    assert "fgenerated999" not in {r["fingerprint"] for r in selected[100]["rows"]}


def _part(folder, fingerprints, monkey_sha):
    folder.mkdir(parents=True)
    with h5py.File(folder / pps_hdf5.FILE_NAME, "w") as stream:
        data = stream.create_group("data")
        for i, fingerprint in enumerate(fingerprints):
            group = data.create_group(f"episode_{i:03d}")
            group.create_dataset("actions", data=np.full((3, 2), i, dtype=np.float32))
            group.attrs.update(fingerprint=fingerprint, num_samples=3)
    episodes = [{"group": f"data/episode_{i:03d}", "fingerprint": f, "kind": "generated", "num_samples": 3}
                for i, f in enumerate(fingerprints)]
    manifest = {"format": pps_hdf5.FORMAT, "sha256": monkey_sha(folder / pps_hdf5.FILE_NAME), "task": "T", "prompt": "p",
                "pool_manifest_sha256": "x", "episodes": episodes, "conventions": {}}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


def test_a_set_file_is_assembled_from_chunk_exports_in_pool_order(tmp_path, monkeypatch):
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    monkeypatch.setattr(pps_hdf5, "verify_file", lambda path: {"held_joint": {}})
    monkeypatch.setattr(pps_hdf5, "revision", lambda: {})
    parts = [_part(tmp_path / "c0", ["a", "b"], sha), _part(tmp_path / "c1", ["c", "d"], sha)]
    pool = tmp_path / "pool"
    pool.mkdir()
    order = ["c", "a", "b"]
    (pool / "dataset.json").write_text(json.dumps({"session": "s", "family": "f", "episodes": [{"fingerprint": f} for f in order]}))
    target = merge_pps(pool, parts, tmp_path / "out")
    with h5py.File(target / pps_hdf5.FILE_NAME, "r") as stream:
        assert [stream[f"data/episode_{i:03d}"].attrs["fingerprint"] for i in range(3)] == order
        assert stream["data/episode_000/actions"][0, 0] == 0 and stream["data/episode_002/actions"][0, 0] == 1
    manifest = verify_merged(target, pool)
    assert [e["fingerprint"] for e in manifest["episodes"]] == order
    assert Counter(p["episodes"] for p in manifest["assembled_from"]) == {2: 1, 1: 1}
    with pytest.raises(FileExistsError):
        merge_pps(pool, parts, tmp_path / "out")
    (pool / "dataset.json").write_text(json.dumps({"session": "s", "family": "f", "episodes": [{"fingerprint": "zz"}]}))
    with pytest.raises(ValueError, match="in no part"):
        merge_pps(pool, parts, tmp_path / "other")


def test_yaw_arc_and_a_derived_profile():
    assert coverage_profile.yaw_arc([350, 10, 20]) == [350.0, 380.0]
    assert coverage_profile.yaw_arc([70.4, 100, 196.2]) == [70.0, 197.0]
    assert coverage_profile.yaw_arc(np.linspace(0, 359, 50)) == [0.0, 360.0]
    box = {"lower": [-0.04, -0.04, 0], "upper": [0.04, 0.04, 0.1], "scale": [1, 1, 1]}
    geometry = {"obj": {**box, "pose": [0.5, 0, 0.05, 1, 0, 0, 0]},
                "bin": {"lower": [-0.1, -0.1, 0], "upper": [0.1, 0.1, 0.1], "scale": [1, 1, 1], "pose": [0.5, 0.3, 0, 1, 0, 0, 0]},
                "table": {"lower": [-0.5, -0.6, -0.05], "upper": [0.5, 0.6, 0], "scale": [1, 1, 1], "pose": [0.55, 0, 0, 1, 0, 0, 0]}}
    rng = np.random.default_rng(0)
    layouts = []
    for _ in range(50):
        yaw = np.radians(rng.uniform(80, 190))
        layouts.append({"poses": {"obj": [rng.uniform(0.28, 0.72), rng.uniform(-0.4, 0.4), 0, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
                                  "bin": [rng.uniform(0.43, 0.66), rng.uniform(-0.33, 0.33), 0, 1, 0, 0, 0]}})
    profile = coverage_profile.derive_profile("obj", "bin", layouts, geometry, draws=1500)
    assert profile["container_yaw_deg"] == [0.0, 0.0] and 79 <= profile["object_yaw_deg"][0] <= profile["object_yaw_deg"][1] <= 191
    strata = profile["object_x_strata"]
    assert len(strata["weights"]) == len(strata["edges"]) - 1 == 5 and abs(sum(strata["weights"]) - 1) < 1e-3
    assert sum(strata["layout_counts"]) == 50 and strata["edges"][0] == profile["object_xy"][0][0]


def test_bounds_that_follow_the_scene_yaw_are_replaced_by_local_bounds(monkeypatch):
    def scene(lower, upper):
        return {"obj": {"lower": lower, "upper": upper, "scale": [1, 1, 1], "pose": [0, 0, 0.1, 1, 0, 0, 0]},
                "table": {"lower": [-1, -1, 0], "upper": [1, 1, 0], "scale": [1, 1, 1], "pose": [0, 0, 0, 1, 0, 0, 0]}}

    legacy = {"a": scene([-0.05, -0.09, 0], [0.05, 0.09, 0.2]), "b": scene([-0.11, -0.11, 0], [0.11, 0.11, 0.2])}
    local = {"obj": ([-0.04, -0.08, 0], [0.04, 0.08, 0.2]), "table": ([-1, -1, 0], [1, 1, 0])}
    monkeypatch.setattr(coverage_profile, "scene_geometry", lambda path, names: json.loads(json.dumps(legacy[path])))
    monkeypatch.setattr(coverage_profile, "local_bounds", lambda path, names: local)
    geometry, replaced = coverage_profile.consistent_geometry(["a", "b"], ("obj", "table"))
    assert replaced == ["obj"]
    assert (geometry["obj"]["lower"], geometry["obj"]["upper"]) == local["obj"] and geometry["table"] == legacy["a"]["table"]

    # Bounds that agree with the geometry's own are left to common_geometry, unchanged.
    monkeypatch.setattr(coverage_profile, "scene_geometry", lambda path, names: json.loads(json.dumps(legacy["a"])))
    monkeypatch.setattr(coverage_profile, "local_bounds",
                        lambda path, names: {n: (legacy["a"][n]["lower"], legacy["a"][n]["upper"]) for n in names})
    monkeypatch.setattr(coverage_profile, "common_geometry", lambda scenes, names, atol: "common")
    assert coverage_profile.consistent_geometry(["a", "b"], ("obj", "table")) == ("common", [])

    # Local bounds that still differ between scenes are refused.
    monkeypatch.setattr(coverage_profile, "scene_geometry", lambda path, names: json.loads(json.dumps(legacy[path])))
    monkeypatch.setattr(coverage_profile, "local_bounds", lambda path, names: {
        "obj": ([-0.04, -0.08, 0], [0.04 if path == "a" else 0.06, 0.08, 0.2]), "table": local["table"]})
    with pytest.raises(ValueError, match="obj upper differs"):
        coverage_profile.consistent_geometry(["a", "b"], ("obj", "table"))
