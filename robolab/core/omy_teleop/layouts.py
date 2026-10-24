"""Select compatible simple-PnP layouts without importing simulator code.

Only the generated USDA subset is supported. Keeping this deliberately strict
lets the collector reject geometry or task changes before launching Isaac Sim.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

from robolab.core.task.task_index import ResolvedTask, resolve_selected_task


_ROOT = Path(__file__).resolve().parents[2]
_LAYOUT_FILE = re.compile(r"pnp_(\d+)_([a-zA-Z0-9_]+)_(\d+)_task\.py\Z")


@dataclass(frozen=True)
class Layout:
    task: str
    resolution: ResolvedTask
    scene_file: Path
    scene_relative: str
    poses: dict[str, tuple[float, ...]]
    scene_sha256: str
    layout_index: int
    target: str
    destination: str

    def metadata(self) -> dict:
        return {
            "task": self.task,
            "task_source": self.resolution.metadata(),
            "scene": self.scene_relative,
            "scene_sha256": self.scene_sha256,
            "poses": {name: list(pose) for name, pose in self.poses.items()},
            "layout_index": self.layout_index,
            "target": self.target,
            "destination": self.destination,
        }

    def verify_current(self) -> None:
        self.resolution.verify_current()
        if hashlib.sha256(self.scene_file.read_bytes()).hexdigest() != self.scene_sha256:
            raise ValueError(f"Scene changed after lookup: {self.scene_file}")


@dataclass(frozen=True)
class LayoutPlan:
    object_name: str
    catalog_index: int
    layouts: tuple[Layout, ...]

    def metadata(self) -> dict:
        return {
            "schema_version": 1,
            "object_name": self.object_name,
            "catalog_index": self.catalog_index,
            "layouts": [layout.metadata() for layout in self.layouts],
        }


def _read_manifest(path: Path, key: str) -> list[dict]:
    try:
        contents = json.loads(path.read_text())
        rows = contents[key]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Cannot read {key} manifest {path}: {exc}") from exc
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Manifest {path} must contain a {key} list of objects")
    return rows


def _scoped_file(root: Path, relative: str, *, suffix: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"Invalid manifest file path: {relative!r}")
    path = Path(relative)
    resolved = (root / path).resolve()
    if (path.is_absolute() or ".." in path.parts or path.suffix != suffix
            or not resolved.is_relative_to(root) or not resolved.is_file()):
        raise ValueError(f"Manifest file must exist inside {root}: {relative!r}")
    return resolved


class _TaskSignature(ast.NodeTransformer):
    def __init__(self, task: str, scene: str):
        prefix = task.removesuffix("Task")
        self.names = {task: "LayoutTask", prefix + "Terminations": "LayoutTerminations"}
        self.scene = scene
        self.scene_calls = 0

    def visit_ClassDef(self, node):
        node.name = self.names.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_Name(self, node):
        node.id = self.names.get(node.id, node.id)
        return node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "import_scene":
            if not node.args or not isinstance(node.args[0], ast.Constant) or node.args[0].value != self.scene:
                raise ValueError("Task scene does not match its manifest")
            self.scene_calls += 1
            node.args[0] = ast.Constant(value="<layout-scene>")
        return self.generic_visit(node)


def _task_signature(layout: Layout) -> str:
    tree = ast.parse(layout.resolution.task_file.read_bytes(), filename=str(layout.resolution.task_file))
    normalizer = _TaskSignature(layout.task, layout.scene_relative)
    tree = normalizer.visit(tree)
    if normalizer.scene_calls != 1:
        raise ValueError(f"Expected exactly one authored import_scene call: {layout.task}")
    return ast.dump(tree, include_attributes=False)


def _numeric_tuple(source: str, size: int, *, path: Path) -> tuple[float, ...]:
    try:
        value = ast.literal_eval(source)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Unsupported pose in scene {path}") from exc
    if (not isinstance(value, tuple) or len(value) != size
            or any(type(item) not in (int, float) or not math.isfinite(item) for item in value)):
        raise ValueError(f"Expected {size} finite pose components in scene {path}")
    return tuple(float(item) for item in value)


def _scene_poses(path: Path, names: tuple[str, str]) -> tuple[dict[str, tuple[float, ...]], str]:
    source = path.read_text()
    world = list(re.finditer(r'^def Xform "world"\n\{\n((?:    [^\n]*\n)+)', source, flags=re.M))
    if len(world) != 1:
        raise ValueError(f"Expected one generated world frame: {path}")
    world_order = re.findall(r"^    uniform token\[\] xformOpOrder = (.*)$", world[0].group(1), flags=re.M)
    if world_order != ['["xformOp:translate", "xformOp:orient", "xformOp:scale"]']:
        raise ValueError(f"Unsupported world transform order: {path}")
    # These poses are in the environment frame only because /world is identity.
    # Reject unsupported composition instead of silently dropping a parent pose.
    for op, usd_type, expected in (("translate", "double3", (0.0, 0.0, 0.0)),
                                   ("orient", "quat[fd]", (1.0, 0.0, 0.0, 0.0)),
                                   ("scale", "(?:float|double)3", (1.0, 1.0, 1.0))):
        values = re.findall(r"^    " + usd_type + " xformOp:" + op + r" = (\([^\n]+\))$",
                            world[0].group(1), flags=re.M)
        if len(values) != 1 or _numeric_tuple(values[0], len(expected), path=path) != expected:
            raise ValueError(f"Unsupported nonidentity world frame: {path}")
    poses = {}
    for name in names:
        # Restrict this parser to a flat, generated child prim of /world.
        pattern = re.compile(
            r'(^    def(?: Xform)? "' + re.escape(name)
            + r'"[^\n]*\(\n[^{}]*?^    \)\n    \{\n)([^{}]*?)(^    \})',
            re.M,
        )
        matches = list(pattern.finditer(source))
        if len(matches) != 1:
            raise ValueError(f"Unsupported or ambiguous scene prim {name!r}: {path}")
        match = matches[0]
        body = match.group(2)
        values = {}
        for op, usd_type, count in (("translate", "double3", 3), ("orient", "quat[fd]", 4)):
            pose_pattern = re.compile(r"(^        " + usd_type + " xformOp:" + op + r" = )(\([^\n]+\))$", re.M)
            pose_matches = list(pose_pattern.finditer(body))
            if len(pose_matches) != 1:
                raise ValueError(f"Expected one authored {op} for {name!r}: {path}")
            values[op] = _numeric_tuple(pose_matches[0].group(2), count, path=path)
            body = pose_pattern.sub(r"\1<layout-pose>", body)
        order = re.findall(r'^        uniform token\[\] xformOpOrder = (.*)$', body, flags=re.M)
        if order != ['["xformOp:translate", "xformOp:orient", "xformOp:scale"]']:
            raise ValueError(f"Unsupported transform order for {name!r}: {path}")
        if abs(sum(value * value for value in values["orient"]) - 1.0) > 1e-5:
            raise ValueError(f"Non-unit orientation for {name!r}: {path}")
        poses[name] = values["translate"] + values["orient"]
        source = source[:match.start(2)] + body + source[match.end(2):]
    return poses, source


def select_layouts(object_name=None, object_id=None, *, task_root=None, scene_root=None) -> LayoutPlan:
    """Load retained layouts for one catalog entry, rejecting incompatible rows.

    ``task_root`` is the robolab/tasks directory; ``scene_root`` is assets/scenes.
    Neither task modules nor USD/Isaac packages are imported by this function.
    """
    if object_name is None and object_id is None:
        raise ValueError("Select an object name or catalog ID")
    if object_name is not None and (not isinstance(object_name, str) or not object_name.strip()):
        raise ValueError("Object name must be a nonempty string")
    catalog_id = None
    if object_id is not None:
        if isinstance(object_id, bool) or not re.fullmatch(r"\d+", str(object_id)):
            raise ValueError("Object ID must be a nonnegative catalog integer")
        catalog_id = int(object_id)
    tasks_root = Path(task_root or _ROOT / "tasks").resolve()
    scenes_root = Path(scene_root or _ROOT.parent / "assets/scenes").resolve()
    task_rows = _read_manifest(tasks_root / "single_pnp/manifest.json", "tasks")
    scene_rows = _read_manifest(scenes_root / "simple_pnp/manifest.json", "scenes")
    scenes = {}
    for row in scene_rows:
        key = row.get("scene")
        if not isinstance(key, str) or key in scenes:
            raise ValueError("Scene manifest contains invalid or duplicate scene names")
        scenes[key] = row
    selected = []
    for row in task_rows:
        filename = row.get("task")
        match = _LAYOUT_FILE.fullmatch(filename) if isinstance(filename, str) else None
        if match is None:
            raise ValueError(f"Unsupported simple-PnP task manifest entry: {filename!r}")
        index, _, layout_index = match.groups()
        if object_name is not None and row.get("target") != object_name:
            continue
        if catalog_id is not None and int(index) != catalog_id:
            continue
        selected.append((int(index), int(layout_index), row))
    if not selected:
        raise ValueError(f"No retained layouts for object {object_name!r}, catalog ID {object_id!r}")
    indices = {index for index, _, _ in selected}
    if len(indices) != 1:
        raise ValueError(f"Object name selects multiple catalog entries {sorted(indices)}; use --object-id")
    selected.sort(key=lambda item: item[1])
    seen_indices = set()
    seen_tasks = set()
    layouts = []
    task_signature = None
    scene_signature = None
    selected_name = selected[0][2].get("target")
    for index, layout_index, row in selected:
        task, target, destination, relative = (row.get(name) for name in ("class_name", "target", "destination", "scene"))
        if (not all(isinstance(value, str) and value for value in (task, target, destination, relative))
                or target != selected_name or target == destination):
            raise ValueError("Selected task manifest entries have inconsistent object identities")
        if layout_index in seen_indices or task in seen_tasks:
            raise ValueError(f"Duplicate selected layout/task in manifest: {task}")
        seen_indices.add(layout_index)
        seen_tasks.add(task)
        if relative != "simple_pnp/" + row["task"].removesuffix("_task.py") + ".usda":
            raise ValueError(f"Task and scene filenames disagree: {task}")
        scene_row = scenes.get(Path(relative).name)
        if (scene_row is None or scene_row.get("catalog_index") != index
                or not isinstance(scene_row.get("target"), dict)
                or not isinstance(scene_row.get("destination"), dict)
                or scene_row.get("target", {}).get("name") != target
                or scene_row.get("destination", {}).get("name") != destination):
            raise ValueError(f"Task and scene manifests disagree: {task}")
        _scoped_file(tasks_root / "single_pnp", row["task"], suffix=".py")
        scene_file = _scoped_file(scenes_root, relative, suffix=".usda")
        resolution = resolve_selected_task(task, ["single_pnp"], task_root=tasks_root)
        poses, signature = _scene_poses(scene_file, (target, destination))
        layout = Layout(task, resolution, scene_file, relative, poses,
                        hashlib.sha256(scene_file.read_bytes()).hexdigest(), layout_index, target, destination)
        definition = _task_signature(layout)
        if task_signature is not None and definition != task_signature:
            raise ValueError(f"Selected layouts have different task definitions: {task}")
        if scene_signature is not None and signature != scene_signature:
            raise ValueError(f"Scenes differ beyond target/bin translation and orientation: {scene_file}")
        task_signature, scene_signature = definition, signature
        layouts.append(layout)
    return LayoutPlan(selected_name, next(iter(indices)), tuple(layouts))
