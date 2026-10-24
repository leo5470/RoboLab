"""Crash-resilient progress for a sequence of compatible object layouts.

The JSON manifest is updated only after attempt export. Resume additionally
discovers finalized HDF5 episodes to recover an export that preceded a crash.
Every process writes new part directories and holds an exclusive session lock.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile

from .layouts import Layout, LayoutPlan, select_layouts


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _portable_plan(metadata: dict) -> dict:
    """Lookup strategy is diagnostic and does not change a task definition."""
    result = _json_copy(metadata)
    for layout in result["layouts"]:
        layout["task_source"].pop("lookup_method", None)
    return result


class LayoutSession:
    """One object collection pass. Use ``close()`` to release its file lock."""

    def __init__(self, root: Path, plan: LayoutPlan, state: dict, lock):
        self.root = root
        self.plan = plan
        self._state = state
        self._lock = lock
        self._layouts = {layout.task: layout for layout in plan.layouts}

    @staticmethod
    def _acquire(root: Path):
        lock_path = root / ".session.lock"
        if lock_path.is_symlink():
            raise ValueError(f"Session lock cannot be a symbolic link: {lock_path}")
        stream = lock_path.open("a+")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise ValueError(f"Collection session is already open in another process: {root}") from exc
        return stream

    @classmethod
    def create(cls, root: Path, plan: LayoutPlan, target_successes: int, run_metadata: dict,
               *, target_total_successes: int | None = None):
        if type(target_successes) is not int or target_successes < 1:
            raise ValueError("Successes per layout must be a positive integer")
        if not plan.layouts or len({layout.task for layout in plan.layouts}) != len(plan.layouts):
            raise ValueError("A collection plan must have unique, nonempty layouts")
        if not isinstance(run_metadata, dict):
            raise ValueError("Session run_metadata must be a dictionary")
        metadata = _json_copy(run_metadata)
        if target_total_successes is not None:
            metadata["target_total_successes"] = target_total_successes
        cls.validate_total_target(metadata.get("target_total_successes"), target_successes, len(plan.layouts))
        for layout in plan.layouts:
            layout.verify_current()
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        lock = cls._acquire(root)
        try:
            if any(path.name != ".session.lock" for path in root.iterdir()):
                raise ValueError(f"New collection session directory must be empty: {root}")
            state = {
                "schema_version": 1,
                "plan": plan.metadata(),
                "target_successes": target_successes,
                "run_metadata": metadata,
                "created_at": _now(),
                "pass": 1,
                "queue": [layout.task for layout in plan.layouts],
                "position": 0,
                "layouts": {layout.task: {"attempts": {}, "parts": [], "skipped": False}
                            for layout in plan.layouts},
            }
            session = cls(root, plan, state, lock)
            session._persist("session_created")
            return session
        except BaseException:
            lock.close()
            raise

    @classmethod
    def _read(cls, root: Path, plan=None) -> tuple[LayoutPlan, dict]:
        try:
            state = json.loads((root / "session.json").read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read collection session {root}: {exc}") from exc
        if not isinstance(state, dict) or type(state.get("schema_version")) is not int or state["schema_version"] != 1:
            raise ValueError("Unsupported collection session schema")
        stored_plan = state.get("plan")
        if (not isinstance(stored_plan, dict) or type(stored_plan.get("schema_version")) is not int
                or stored_plan["schema_version"] != 1):
            raise ValueError("Invalid recorded layout plan")
        if plan is None:
            plan = select_layouts(stored_plan.get("object_name"), stored_plan.get("catalog_index"))
        try:
            same_plan = _portable_plan(stored_plan) == _portable_plan(plan.metadata())
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("Invalid recorded layout source metadata") from exc
        if not same_plan:
            raise ValueError("Recorded layout sources or retained layout sequence changed; restore the recorded sources before resuming")
        for layout in plan.layouts:
            layout.verify_current()
        quota = state.get("target_successes")
        if type(quota) is not int or quota < 1 or not isinstance(state.get("run_metadata"), dict):
            raise ValueError("Invalid recorded collection quota or configuration")
        cls.validate_total_target(state["run_metadata"].get("target_total_successes"), quota, len(plan.layouts))
        tasks = {layout.task for layout in plan.layouts}
        progress = state.get("layouts")
        queue, position = state.get("queue"), state.get("position")
        if (not isinstance(progress, dict) or set(progress) != tasks
                or not isinstance(queue, list) or not all(isinstance(task, str) and task in tasks for task in queue)
                or len(set(queue)) != len(queue) or type(position) is not int or not 0 <= position <= len(queue)):
            raise ValueError("Invalid recorded collection progress")
        if type(state.get("pass")) is not int or state["pass"] < 1:
            raise ValueError("Invalid recorded collection pass")
        for task, row in progress.items():
            if (not isinstance(row, dict) or not isinstance(row.get("attempts"), dict)
                    or not isinstance(row.get("parts"), list) or type(row.get("skipped")) is not bool):
                raise ValueError(f"Invalid recorded layout progress: {task}")
            for relative in row["parts"]:
                cls._part_path(root, task, relative, must_exist=False)
            for key, attempt in row["attempts"].items():
                if (not isinstance(attempt, dict) or type(attempt.get("success")) is not bool
                        or not isinstance(attempt.get("reason"), str)
                        or type(attempt.get("episode_index")) is not int or attempt["episode_index"] < 0):
                    raise ValueError(f"Invalid recorded attempt: {task}/{key}")
                cls._part_path(root, task, attempt.get("part"), must_exist=False)
                if key != cls._attempt_key(attempt["part"], attempt["episode_index"]):
                    raise ValueError(f"Invalid recorded attempt identity: {task}/{key}")
        return plan, state

    @classmethod
    def inspect(cls, root: Path, plan=None) -> tuple[LayoutPlan, int, dict, Layout | None]:
        """Read-only preflight; returns plan, quota, configuration, first unmet layout.

        This inspects persisted counters. ``resume`` also reconciles completed
        HDF5 exports, so its first unmet layout can be later in the sequence.
        """
        plan, state = cls._read(Path(root).resolve(), plan)
        quota = state["target_successes"]
        total = sum(cls._success_count(row) for row in state["layouts"].values())
        target = state["run_metadata"].get("target_total_successes")
        current = None if target is not None and total >= target else next(
            (layout for layout in plan.layouts
             if cls._success_count(state["layouts"][layout.task]) < quota), None)
        return plan, quota, _json_copy(state["run_metadata"]), current

    @classmethod
    def resume(cls, root: Path, plan=None, run_metadata=None):
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(f"Collection session directory does not exist: {root}")
        lock = cls._acquire(root)
        try:
            plan, state = cls._read(root, plan)
            if run_metadata is not None:
                if not isinstance(run_metadata, dict):
                    raise ValueError("Resume run_metadata must be a dictionary")
                requested = _json_copy(run_metadata)
                for key, value in requested.items():
                    if key not in state["run_metadata"] or state["run_metadata"][key] != value:
                        raise ValueError(f"Collection configuration changed at {key!r}; restore the recorded configuration")
            session = cls(root, plan, state, lock)
            recovered = session._reconcile()
            state["pass"] += 1
            state["queue"] = [layout.task for layout in plan.layouts
                              if session._success_count(state["layouts"][layout.task]) < session.target_successes]
            state["position"] = 0
            session._persist("session_resumed", recovered_attempts=recovered)
            return session
        except BaseException:
            lock.close()
            raise

    @property
    def target_successes(self) -> int:
        return self._state["target_successes"]

    @staticmethod
    def validate_total_target(target, per_layout, layout_count):
        if target is not None and (type(target) is not int or not 1 <= target <= per_layout * layout_count):
            raise ValueError("Total success target must be a positive integer within the layout quotas; "
                             "increase --episodes-per-layout for a larger total")

    @property
    def target_total_successes(self) -> int | None:
        return self._state["run_metadata"].get("target_total_successes")

    @property
    def total_successes(self) -> int:
        return sum(self._success_count(row) for row in self._state["layouts"].values())

    @property
    def total_target_reached(self) -> bool:
        return self.target_total_successes is not None and self.total_successes >= self.target_total_successes

    @property
    def run_metadata(self) -> dict:
        return _json_copy(self._state["run_metadata"])

    @property
    def current(self) -> Layout | None:
        if self.finished:
            return None
        return self._layouts[self._state["queue"][self._state["position"]]]

    @property
    def attempts(self) -> int:
        # Keep the last layout's counters available when its success ends the session.
        queue, position = self._state["queue"], self._state["position"]
        return len(self._state["layouts"][queue[position]]["attempts"]) if position < len(queue) else 0

    @staticmethod
    def _success_count(row: dict) -> int:
        return sum(attempt["success"] for attempt in row["attempts"].values())

    @property
    def successes(self) -> int:
        queue, position = self._state["queue"], self._state["position"]
        return self._success_count(self._state["layouts"][queue[position]]) if position < len(queue) else 0

    @property
    def complete(self) -> bool:
        if self.target_total_successes is not None:
            return self.total_target_reached
        return all(self._success_count(row) >= self.target_successes for row in self._state["layouts"].values())

    @property
    def finished(self) -> bool:
        return self.total_target_reached or self._state["position"] >= len(self._state["queue"])

    def _require_open(self):
        if self._lock is None:
            raise ValueError("Collection session is closed")

    def _persist(self, event: str, **details):
        self._require_open()
        self._state["updated_at"] = _now()
        payload = json.dumps(self._state, indent=2, sort_keys=True, allow_nan=False) + "\n"
        descriptor, temporary = tempfile.mkstemp(prefix=".session.", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(descriptor, "w") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.root / "session.json")
            directory_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        entry = {"time": self._state["updated_at"], "event": event, "pass": self._state["pass"], **details}
        with (self.root / "collection_events.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _part_path(root: Path, task: str, part, *, must_exist: bool) -> Path:
        if not isinstance(part, (str, Path)):
            raise ValueError("Attempt part must be a collection part path")
        requested = Path(part)
        path = (requested if requested.is_absolute() else root / requested).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Collection part escapes its session directory") from exc
        if (len(relative.parts) != 2 or relative.parts[0] != task
                or re.fullmatch(r"part_\d{3,}", relative.parts[1]) is None
                or (must_exist and not path.is_dir())):
            raise ValueError(f"Invalid collection part for {task}: {part}")
        return path

    def start_part(self) -> Path:
        self._require_open()
        current = self.current
        if current is None:
            raise ValueError("The current collection pass is finished")
        directory = self.root / current.task
        if directory.is_symlink():
            raise ValueError("Layout recording directory cannot be a symbolic link")
        directory.mkdir(exist_ok=True)
        indices = [int(path.name.removeprefix("part_")) for path in directory.iterdir()
                   if re.fullmatch(r"part_\d+", path.name)]
        path = directory / f"part_{max(indices, default=-1) + 1:03d}"
        path.mkdir()
        relative = path.relative_to(self.root).as_posix()
        self._state["layouts"][current.task]["parts"].append(relative)
        self._persist("part_started", task=current.task, part=relative)
        return path

    @staticmethod
    def _attempt_key(part: str, episode_index: int) -> str:
        return f"{part}:demo_{episode_index}"

    def record_attempt(self, success: bool, reason: str, part: Path, episode_index: int):
        """Persist one finalized attempt; repeating the same identity is harmless."""
        self._require_open()
        current = self.current
        if current is None:
            raise ValueError("Cannot record an attempt after the pass has ended")
        if (type(success) is not bool or not isinstance(reason, str) or not reason
                or type(episode_index) is not int or episode_index < 0):
            raise ValueError("Attempt requires a boolean outcome, reason, and nonnegative episode index")
        path = self._part_path(self.root, current.task, part, must_exist=True)
        relative = path.relative_to(self.root).as_posix()
        row = self._state["layouts"][current.task]
        if relative not in row["parts"]:
            raise ValueError("Attempt part was not opened by this session")
        key = self._attempt_key(relative, episode_index)
        values = {"success": success, "reason": reason, "part": relative, "episode_index": episode_index}
        previous = row["attempts"].get(key)
        if previous is not None:
            if any(previous[name] != value for name, value in values.items()):
                raise ValueError(f"Conflicting outcome for already finalized attempt {key}")
            return
        row["attempts"][key] = {**values, "finalized_at": _now()}
        self._persist("attempt_finalized", task=current.task, **values)

    def advance(self, skipped=False):
        self._require_open()
        current = self.current
        if current is None:
            return
        if type(skipped) is not bool:
            raise ValueError("skipped must be a boolean")
        if not skipped and self.successes < self.target_successes:
            raise ValueError("Cannot complete a layout before its success quota is reached")
        self._state["layouts"][current.task]["skipped"] = skipped and self.successes < self.target_successes
        self._state["position"] += 1
        self._persist("layout_skipped" if skipped else "layout_completed", task=current.task,
                      next_task=self.current.task if self.current else None)

    def _reconcile(self) -> int:
        # Optional import keeps selection/check mode independent of HDF5 and CUDA.
        import h5py

        recovered = 0
        for layout in self.plan.layouts:
            directory = self.root / layout.task
            row = self._state["layouts"][layout.task]
            if not directory.exists():
                if row["attempts"]:
                    raise ValueError(f"Finalized recording directory is missing: {directory}")
                continue
            if directory.is_symlink():
                raise ValueError("Layout recording directory cannot be a symbolic link")
            finalized_keys = set()
            for candidate in sorted(directory.glob("part_*")):
                part = self._part_path(self.root, layout.task, candidate, must_exist=True)
                relative = part.relative_to(self.root).as_posix()
                if relative not in row["parts"]:
                    row["parts"].append(relative)
                files = sorted(part.glob("*.hdf5"))
                if len(files) > 1:
                    raise ValueError(f"A collection part must have at most one HDF5 recording: {part}")
                for filename in files:
                    if filename.is_symlink():
                        raise ValueError("Collection recording cannot be a symbolic link")
                    try:
                        with h5py.File(filename, "r") as stream:
                            data = stream.get("data")
                            if data is None:
                                continue
                            if not isinstance(data, h5py.Group):
                                raise ValueError(f"Invalid recording data group: {filename}")
                            for attributes in (stream.attrs, data.attrs):
                                task = attributes.get("collection_task")
                                if task is not None and self._string(task) != layout.task:
                                    raise ValueError(f"Recording task identity conflicts with its layout directory: {filename}")
                            for name, episode in data.items():
                                match = re.fullmatch(r"demo_(\d+)", name)
                                if not match or not isinstance(episode, h5py.Group):
                                    continue
                                finalized = episode.attrs.get("collection_finalized", True)
                                finalized = finalized.item() if hasattr(finalized, "item") else finalized
                                if not isinstance(finalized, (bool, int)) or finalized not in (0, 1):
                                    raise ValueError(f"Invalid finalized attempt marker: {filename}/{name}")
                                if not finalized or "success" not in episode.attrs:
                                    continue
                                success = episode.attrs["success"]
                                success = success.item() if hasattr(success, "item") else success
                                if not isinstance(success, (bool, int)) or success not in (0, 1):
                                    raise ValueError(f"Invalid finalized attempt outcome: {filename}/{name}")
                                actions = episode.get("actions")
                                if actions is None:
                                    continue
                                if not isinstance(actions, h5py.Dataset):
                                    raise ValueError(f"Invalid finalized actions dataset: {filename}/{name}")
                                if actions.ndim < 1 or actions.shape[0] == 0:
                                    continue
                                if int(episode.attrs.get("num_samples", -1)) != actions.shape[0]:
                                    raise ValueError(f"Incomplete finalized attempt: {filename}/{name}")
                                task = episode.attrs.get("collection_task")
                                if task is not None and self._string(task) != layout.task:
                                    raise ValueError(f"Attempt task identity conflicts with its layout: {filename}/{name}")
                                episode_index = int(match.group(1))
                                key = self._attempt_key(relative, episode_index)
                                finalized_keys.add(key)
                                previous = row["attempts"].get(key)
                                if previous is not None:
                                    if previous["success"] != bool(success):
                                        raise ValueError(f"Recorded and persisted attempt outcomes disagree: {filename}/{name}")
                                    continue
                                reason = self._string(episode.attrs.get("collection_reason", "recovered_success" if success else "recovered_failure"))
                                row["attempts"][key] = {
                                    "success": bool(success), "reason": reason, "part": relative,
                                    "episode_index": episode_index, "finalized_at": _now(), "recovered": True,
                                }
                                recovered += 1
                    except OSError as exc:
                        raise ValueError(f"Cannot read collection recording {filename}: {exc}") from exc
            missing = set(row["attempts"]) - finalized_keys
            if missing:
                raise ValueError(f"Persisted attempts have no finalized HDF5 recording: {sorted(missing)}")
        return recovered

    @staticmethod
    def _string(value) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    def summary(self) -> dict:
        rows = self._state["layouts"]
        completed = sum(self._success_count(row) >= self.target_successes for row in rows.values())
        return {
            "object_name": self.plan.object_name,
            "catalog_index": self.plan.catalog_index,
            "total_layouts": len(self.plan.layouts),
            "completed": completed,
            "skipped": sum(row["skipped"] and self._success_count(row) < self.target_successes for row in rows.values()),
            "pending": len(self.plan.layouts) - completed,
            "attempts": sum(len(row["attempts"]) for row in rows.values()),
            "successes": sum(self._success_count(row) for row in rows.values()),
            "target_successes": self.target_successes,
            "target_total_successes": self.target_total_successes,
            "remaining_successes": (max(0, self.target_total_successes - self.total_successes)
                                    if self.target_total_successes is not None else
                                    sum(max(0, self.target_successes - self._success_count(row))
                                        for row in rows.values())),
            "current_task": self.current.task if self.current else None,
            "current_position": next((index + 1 for index, layout in enumerate(self.plan.layouts)
                                      if self.current and layout.task == self.current.task), None),
            "finished": self.finished,
            "complete": self.complete,
        }

    def close(self):
        if self._lock is not None:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()
