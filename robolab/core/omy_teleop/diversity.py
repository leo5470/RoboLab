"""Quality-gated selection of generated OMY trajectories; no simulator imports."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

from .dataset_paths import job_recording, named_output, source_names


def resample(rows, count=20):
    rows = np.asarray(rows, dtype=float)
    if not len(rows):
        raise ValueError("Cannot compare an empty trajectory")
    axis = np.linspace(0, 1, len(rows))
    target = np.linspace(0, 1, count)
    return np.stack([np.interp(target, axis, column) for column in rows.T], axis=1)


def yaw_features(quat):
    w, x, y, z = np.asarray(quat, dtype=float)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([np.sin(yaw), np.cos(yaw)])


def descriptor(episode, object_name, container_name):
    """Compare arrangements, phase-aligned achieved motion, grasp and release."""
    actions = episode["actions"][:]
    initial = episode["initial_state/rigid_object"]
    states = episode["states/rigid_object"]
    obj = np.concatenate((initial[object_name]["root_pose"][:1], states[object_name]["root_pose"][:-1]))
    container = np.concatenate((initial[container_name]["root_pose"][:1],
                                states[container_name]["root_pose"][:-1]))
    held = np.flatnonzero((actions[:, -1] < 0) & (obj[:, 2] - obj[0, 2] > .02))
    if not len(held):
        raise ValueError("Generated success has no closed-gripper lift")
    grasp = int(held[0])
    opened = np.flatnonzero(actions[grasp:, -1] > 0)
    release = grasp + int(opened[0]) if len(opened) else len(actions) - 1
    eef = episode["mimic/eef_pose"][:]
    if eef.shape != (len(actions), 7):
        raise ValueError("Missing aligned achieved end-effector trajectory")
    joints = episode["obs/proprio_obs/arm_joint_pos"][:]
    scene = np.r_[obj[0, :2], container[0, :2], yaw_features(obj[0, 3:]), yaw_features(container[0, 3:])]
    approach = resample(eef[:grasp + 1, :3] - obj[0, :3])
    transport = resample(eef[grasp:, :3] - container[0, :3])
    arm = np.r_[resample(joints[:grasp + 1]).ravel(), resample(joints[grasp:]).ravel()]
    pose = np.r_[eef[grasp, :3] - obj[grasp, :3], eef[release, :3] - container[release, :3],
                 yaw_features(eef[grasp, 3:]), yaw_features(eef[release, 3:]),
                 obj[:, 2].max() - obj[0, 2]]
    blocks = [scene, np.r_[approach.ravel(), transport.ravel()], arm, pose]
    if any(not np.isfinite(block).all() for block in blocks):
        raise ValueError("Nonfinite diversity features")
    return blocks


def feature_matrix(blocks):
    """Robust, dimension-normalized blocks; tiny numerical drift has a floor."""
    weights = [.5, .3, .1, .1]
    floors = [.02, .02, .1, .02]
    output, recipe = [], []
    for group, weight, floor in zip(zip(*blocks), weights, floors):
        values = np.asarray(group)
        center = np.median(values, axis=0)
        scale = np.maximum(np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0), floor)
        output.append((values - center) / scale * np.sqrt(weight / values.shape[1]))
        recipe.append({"weight": weight, "center": center.tolist(), "scale": scale.tolist()})
    return np.concatenate(output, axis=1), recipe


def select_indices(features, sources, count=100, per_source_min=2, max_share=.2):
    """Seed representative pairs by source, then add farthest eligible candidates."""
    features = np.asarray(features, dtype=float)
    sources = list(sources)
    if len(features) < count:
        raise ValueError(f"Need {count} quality-passing candidates; have {len(features)}")
    groups = defaultdict(list)
    for index, source in enumerate(sources):
        groups[source].append(index)
    chosen = []
    for source in sorted(groups):
        indices = groups[source]
        center = np.median(features[indices], axis=0)
        first = min(indices, key=lambda i: (float(np.linalg.norm(features[i] - center)), i))
        chosen.append(first)
        if per_source_min > 1 and len(indices) > 1:
            # A second representative away from the medoid, within this valid source pool.
            chosen.append(max((i for i in indices if i != first),
                              key=lambda i: (float(np.linalg.norm(features[i] - features[first])), -i)))
    if len(chosen) > count:
        raise ValueError("Source representation quota exceeds selection size")
    if not chosen:
        raise ValueError("No sources")
    distances = np.linalg.norm(features[:, None] - features[chosen][None], axis=2).min(axis=1)
    counts = Counter(sources[i] for i in chosen)
    cap = max(int(np.ceil(count * max_share)), int(np.ceil(count / len(groups))))
    while len(chosen) < count:
        eligible = [i for i, source in enumerate(sources) if i not in chosen and counts[source] < cap]
        if not eligible:
            # Missing/rejected sources can make the nominal cap infeasible; record effective counts.
            cap += 1
            continue
        index = max(eligible, key=lambda i: (float(distances[i]), -i))
        chosen.append(index)
        counts[sources[index]] += 1
        distances = np.minimum(distances, np.linalg.norm(features - features[index], axis=1))
    return chosen


def coverage(features, selected):
    distances = np.linalg.norm(features[:, None] - features[selected][None], axis=2).min(axis=1)
    return {"mean": float(distances.mean()), "p90": float(np.percentile(distances, 90)),
            "max": float(distances.max())}


def plot_selection(root, family, rows, chosen, features, random_selection):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene = np.asarray([row["blocks"][0] for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, offset, label in zip(axes[:2], (0,2), ("Object starts", "Bin starts")):
        axis.scatter(scene[:,offset],scene[:,offset+1],s=10,c="lightgray",label="Candidates")
        axis.scatter(scene[chosen,offset],scene[chosen,offset+1],s=16,c="#1769aa",label="Selected")
        axis.set(xlabel="x (m)",ylabel="y (m)",title=label)
        axis.set_aspect("equal",adjustable="datalim")
    axes[0].legend(loc="best")
    for indices,label in ((chosen,"Selected"),(random_selection,"Random, same source counts")):
        distances = np.linalg.norm(features[:,None]-features[indices][None],axis=2).min(axis=1)
        ordered = np.sort(distances)
        axes[2].plot(ordered,np.linspace(0,1,len(ordered)),label=label)
    axes[2].set(xlabel="Distance to nearest selected demo",ylabel="Fraction of candidate pool",
                title="Coverage in normalized feature space")
    axes[2].legend(fontsize=8)
    figure.suptitle(f"{family}: {len(chosen)} selected from {len(rows)} candidates")
    figure.tight_layout()
    output = root / "reports"
    output.mkdir(exist_ok=True)
    figure.savefig(output / f"{family}_diversity.png",dpi=160)
    plt.close(figure)


def curate(root, count=100):
    """Select per family and copy source-consistent HDF5 shards with an explicit index."""
    root = Path(root)
    records = []
    for result_file in sorted((root / "jobs").glob("*/result.json")):
        result = json.loads(result_file.read_text())
        if result.get("stage") != "generate":
            continue
        path = job_recording(root, result)
        if not path.is_file():
            continue
        with h5py.File(path, "r") as stream:
            generated = json.loads(stream["data"].attrs["robolab_omy_mimicgen"])
            for name in result["accepted"]:
                blocks = descriptor(stream["data"][name], generated["object"], generated["container"])
                records.append({"family": result["family"], "task": result["task"], "tier": result["tier"],
                                "file": str(path.resolve()), "episode": name, "blocks": blocks,
                                "job_id": result["id"], "input": result["input"],
                                "session": source_names(result["input"], result["task"])[0],
                                "named_output": "output_dir" in result,
                                "seed": int(stream["data"][name].attrs["seed"])})
    families = sorted({r["family"] for r in records})
    report, datasets = {}, []
    for family in families:
        rows = [r for r in records if r["family"] == family]
        if len(rows) < count:
            report[family] = {"status": "insufficient_candidates", "candidates": len(rows)}
            continue
        features, recipe = feature_matrix([r["blocks"] for r in rows])
        chosen = select_indices(features, [r["task"] for r in rows], count)
        rng = np.random.default_rng(0)
        source_counts = Counter(rows[i]["task"] for i in chosen)
        random_coverages = []
        for _ in range(20):
            random_selection = []
            for source, amount in sorted(source_counts.items()):
                available = [i for i,row in enumerate(rows) if row["task"] == source]
                random_selection.extend(rng.choice(available, amount, replace=False).tolist())
            random_coverages.append(coverage(features, random_selection))
        report[family] = {"status": "selected", "candidates": len(rows), "selected": len(chosen),
                          "source_counts": dict(Counter(rows[i]["task"] for i in chosen)),
                          "tier_counts": dict(Counter(rows[i]["tier"] for i in chosen)),
                          "coverage": coverage(features, chosen), "random_100_coverage": random_coverages,
                          "normalization": recipe}
        plot_selection(root,family,rows,chosen,features,random_selection)
        shards = defaultdict(list)
        for i in chosen:
            shards[rows[i]["file"]].append(rows[i])
        for source, selected in sorted(shards.items()):
            source = Path(source)
            row = selected[0]
            relative = (named_output(row["input"], row["task"], row["job_id"]) if row["named_output"]
                        else Path(family) / row["job_id"])
            if root.name == row["session"]:
                relative = Path(*named_output(row["input"], row["task"], row["job_id"]).parts[1:])
            output = root / "selected" / relative
            output.mkdir(parents=True, exist_ok=True)
            target = output / "omy_demos.hdf5"
            if target.exists():
                with h5py.File(target, "r") as existing:
                    if set(existing["data"]) != {r["episode"] for r in selected}:
                        raise ValueError(f"Existing selection conflicts with this selection: {target}")
            if not target.exists():
                temporary = target.with_suffix(".tmp.hdf5")
                with h5py.File(source, "r") as src, h5py.File(temporary, "w") as dst:
                    data = dst.create_group("data")
                    for key, value in src["data"].attrs.items():
                        data.attrs[key] = value
                    for row in selected:
                        src.copy(src["data"][row["episode"]], data, row["episode"])
                    data.attrs["total"] = sum(int(data[r["episode"]].attrs["num_samples"]) for r in selected)
                    data.attrs["robolab_omy_diversity"] = json.dumps({"source": str(source), "family": family,
                        "method": "source-constrained farthest-first", "selected_episodes": [r["episode"] for r in selected]})
                shutil.copy2(source.parent / "env_cfg.json", output / "env_cfg.json")
                shutil.copy2(source.parent / "mimicgen_summary.json", output / "mimicgen_summary.json")
                temporary.replace(target)
            for row in selected:
                datasets.append({**{k:v for k,v in row.items() if k != "blocks"},
                                 "file": str(target.resolve()), "candidate_file": str(source)})
    manifest = {"schema_version": 1, "selected_count": len(datasets), "episodes": datasets,
                "families": report, "training_ready": False}
    selection = root / "selection.json"
    if selection.exists():
        previous = json.loads(selection.read_text())
        identities = {(r["candidate_file"], r["episode"]): r["catalog_id"]
                      for r in previous["episodes"] if r.get("catalog_id")}
        for row in datasets:
            if (row["candidate_file"], row["episode"]) in identities:
                row["catalog_id"] = identities[row["candidate_file"], row["episode"]]
        if previous.get("quality_review"):
            manifest["quality_review"] = previous["quality_review"]
    temporary = root / ".selection.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(selection)
    return manifest
