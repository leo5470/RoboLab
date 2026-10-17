"""Metadata-only, scoped task lookup. Never import tasks, USD, or Isaac Lab."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time


def normalize_task_dirs(task_dirs=None) -> tuple[str, ...]:
    """Keep the evaluation CLI's root-relative directory convention."""
    if task_dirs is None:
        from robolab.constants import DEFAULT_TASK_SUBFOLDERS

        task_dirs = DEFAULT_TASK_SUBFOLDERS
    if not isinstance(task_dirs, (list, tuple)) or not task_dirs:
        raise ValueError("task_dirs must be a nonempty list of task directories")
    result = []
    for name in task_dirs:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Task directory names must be nonempty strings")
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or str(path) == "." or "\\" in name:
            raise ValueError(f"Task directory must be relative to robolab/tasks: {name!r}")
        normalized = path.as_posix()
        if normalized not in result:
            result.append(normalized)
    return tuple(result)


def validate_task_source(source, *, task: str) -> dict | None:
    """Validate portable optional provenance without accessing the filesystem."""
    if source is None:
        return None
    if not isinstance(source, dict) or type(source.get("schema_version")) is not int or source["schema_version"] != 1:
        raise ValueError("Invalid task_source schema; expected version 1")
    path = source.get("task_file")
    if (not isinstance(path, str) or not path or Path(path).is_absolute()
            or ".." in Path(path).parts or "\\" in path or Path(path).suffix != ".py"):
        raise ValueError("task_source.task_file must be a root-relative Python task file")
    if source.get("class_name") != task:
        raise ValueError("task_source class_name conflicts with the recorded task")
    digest = source.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("task_source.sha256 must be a lowercase SHA-256 digest")
    return dict(source)


@dataclass(frozen=True)
class ResolvedTask:
    class_name: str
    task_file: Path
    task_root: Path
    task_dirs: tuple[str, ...]
    sha256: str
    method: str
    lookup_seconds: float

    def metadata(self) -> dict:
        return {
            "schema_version": 1,
            "class_name": self.class_name,
            "task_file": self.task_file.relative_to(self.task_root).as_posix(),
            "sha256": self.sha256,
            "lookup_method": self.method,
        }

    def verify_current(self) -> None:
        if hashlib.sha256(self.task_file.read_bytes()).hexdigest() != self.sha256:
            raise ValueError(f"Task definition changed after lookup: {self.task_file}")

    def verify_expected_source(self, source) -> None:
        source = validate_task_source(source, task=self.class_name)
        if source is not None:
            actual = self.metadata()
            for key in ("task_file", "class_name", "sha256"):
                if source[key] != actual[key]:
                    raise ValueError(f"Recorded task source {key} does not match the selected definition; "
                                     "restore the recorded task file before rendering")


def _class_names(contents: bytes, path: Path) -> list[str]:
    try:
        tree = ast.parse(contents, filename=str(path))
    except (SyntaxError, UnicodeError) as exc:
        raise ValueError(f"Cannot statically inspect task file {path}: {exc}") from exc
    # An ordinary declaration, including Task imported under an alias, is
    # discoverable without executing decorators or class-level scene imports.
    return [node.name for node in tree.body if isinstance(node, ast.ClassDef)]


def _task_declarations(contents: bytes, path: Path) -> list[str]:
    tree = ast.parse(contents, filename=str(path))
    bases = {"Task"}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "robolab.core.task.task":
            bases.update(alias.asname or alias.name for alias in node.names if alias.name == "Task")
    result = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if any((isinstance(base, ast.Name) and base.id in bases)
               or (isinstance(base, ast.Attribute) and base.attr == "Task") for base in node.bases):
            result.append(node.name)
            bases.add(node.name)
    return result


def _python_files(folder: Path):
    for path in sorted(folder.rglob("*.py")):
        parts = path.relative_to(folder).parts
        if (path.name == "__init__.py" or any(part.startswith("_") or part in ("generated", "tmp", "not_used")
                                                for part in parts[:-1])):
            continue
        yield path


def resolve_selected_task(task: str, task_dirs=None, *, task_root=None,
                          expected_source=None) -> ResolvedTask:
    """Resolve exactly one task using manifests or static declarations.

    A manifest-backed pack is authoritative: an unlisted class is an error,
    not permission to execute every Python file. Only the selected file is
    validated/read beyond manifest metadata. Explicit files remain scoped.
    No persistent index is used, so regenerated manifests are read on launch.
    """
    started = time.perf_counter()
    if not isinstance(task, str) or not task.strip():
        raise ValueError("A nonempty task name or Python file is required")
    if task_root is None:
        from robolab.constants import TASK_DIR

        task_root = TASK_DIR
    root = Path(task_root).resolve()
    directories = normalize_task_dirs(task_dirs)
    folders = []
    for name in directories:
        folder = (root / name).resolve()
        if not folder.is_relative_to(root) or not folder.is_dir():
            raise ValueError(f"Task directory does not exist or escapes robolab/tasks: {name!r}")
        folders.append(folder)

    matches = {}
    is_path = "/" in task or "\\" in task
    is_filename = not is_path and task.endswith(".py")
    if is_path:
        requested = Path(task)
        if not requested.is_absolute() and not requested.is_file():
            requested = root / requested
        path = requested.resolve()
        if (not path.is_file() or path.suffix != ".py"
                or not any(path.is_relative_to(folder) for folder in folders)):
            raise ValueError(f"Task file must exist within --task-dirs {list(directories)}: {task}")
        matches[path] = (None, "explicit-file")
    else:
        for folder in folders:
            manifest_path = folder / "manifest.json"
            if manifest_path.exists():
                if not manifest_path.resolve().is_relative_to(folder):
                    raise ValueError(f"Task manifest escapes its directory: {manifest_path}")
                try:
                    manifest = json.loads(manifest_path.read_bytes())
                except (ValueError, UnicodeError) as exc:
                    raise ValueError(f"Invalid task manifest {manifest_path}: {exc}") from exc
                rows = manifest.get("tasks") if isinstance(manifest, dict) else None
                if not isinstance(rows, list):
                    raise ValueError(f"Task manifest must contain a tasks list: {manifest_path}")
                selected = []
                for row in rows:
                    if (not isinstance(row, dict) or not isinstance(row.get("class_name"), str)
                            or not row["class_name"] or not isinstance(row.get("task"), str) or not row["task"]):
                        raise ValueError(f"Invalid task entry in {manifest_path}")
                    if row["class_name"] == task or (is_filename and Path(row["task"]).name == task):
                        selected.append(row)
                if len(selected) > 1:
                    raise ValueError(f"Ambiguous task {task!r} in manifest {manifest_path}")
                for row in selected:
                    relative = Path(row["task"])
                    path = (folder / relative).resolve()
                    if (relative.is_absolute() or ".." in relative.parts or "\\" in row["task"]
                            or relative.suffix != ".py" or not path.is_relative_to(folder)):
                        raise ValueError(f"Task manifest entry escapes its directory: {row['task']!r}")
                    if not path.is_file():
                        raise ValueError(f"Task manifest references a missing file: {path}")
                    value = (row["class_name"], "manifest")
                    if path in matches and matches[path][0] != value[0]:
                        raise ValueError(f"Conflicting class names for task file {path}")
                    matches[path] = value
            else:
                for path in _python_files(folder):
                    if is_filename:
                        if path.name == task:
                            matches[path.resolve()] = (None, "static-file")
                    else:
                        try:
                            names = _class_names(path.read_bytes(), path)
                        except ValueError:
                            # An unrelated malformed module must not prevent a
                            # valid task from being selected, nor be executed.
                            continue
                        if task in names:
                            matches[path.resolve()] = (task, "static-class")

    if not matches:
        raise ValueError(f"Task {task!r} not found in --task-dirs {list(directories)}; "
                         "check the selected folder and regenerate its manifest if needed")
    if len(matches) != 1:
        raise ValueError(f"Ambiguous task {task!r}; matches: {', '.join(str(p) for p in matches)}")
    path, (expected_class, method) = next(iter(matches.items()))
    if not path.is_relative_to(root) or not any(path.is_relative_to(folder) for folder in folders):
        raise ValueError(f"Task file escapes the selected directories: {path}")
    contents = path.read_bytes()
    names = _class_names(contents, path)
    declarations = _task_declarations(contents, path)
    if len(declarations) > 1:
        raise ValueError(f"Multiple Task subclasses in {path}; single-task files are required")
    if expected_class is None:
        if len(declarations) != 1:
            raise ValueError(f"Cannot identify one Task subclass statically in {path}; select its class name")
        expected_class = declarations[0]
    if expected_class not in names or (declarations and declarations != [expected_class]):
        raise ValueError(f"Task manifest/class mismatch: {expected_class!r} is not the selected task in {path}")
    result = ResolvedTask(expected_class, path, root, directories, hashlib.sha256(contents).hexdigest(),
                          method, time.perf_counter() - started)
    result.verify_expected_source(expected_source)
    return result
