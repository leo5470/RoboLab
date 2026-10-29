"""One read-only CPU snapshot per audited OMY task-evaluation boundary.

Physics, actions, recorded states and observations retain their original devices.
Unsupported consumers keep the native query path. No diagnostic monkey patches
or global predicate replacements are installed.
"""

import inspect
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace

import torch

from .boundary import boundary_queries


class _CpuQueries:
    def get_pose(self, name, is_relative=True, as_matrix=False, env_id=None):
        if as_matrix:
            raise ValueError("CPU task snapshot does not support matrix pose queries")
        state = self.poses[name]
        pos = state[:, :3] - self.origins if is_relative else state[:, :3]
        quat = state[:, 3:7]
        return (pos, quat) if env_id is None else (pos[env_id], quat[env_id])

    def get_velocity(self, name, env_id=None):
        value = self.poses[name][:, 7:13]
        return value if env_id is None else value[env_id]

    def get_joint_positions(self, name, env_id=None):
        if name != "robot":
            raise ValueError("CPU task snapshot supports only robot joints")
        return self.joints if env_id is None else self.joints[env_id]

    def get_contact_force(self, first, second, env_id=None):
        pair = tuple(sorted((first, second)))
        force, reversed_sensor = self.forces[pair]
        value = force.sum(dim=(1, 2))
        if reversed_sensor != ((first, second) != pair):
            value = -value
        return value if env_id is None else value[env_id]

    def in_contact(self, first, second, force_threshold=.1, env_id=None):
        results = []
        for left in self.resolve_contact_bodies(first):
            for right in self.resolve_contact_bodies(second):
                if left in self.batch_forces and right in self.batch_indices[left]:
                    index = self.batch_indices[left][right]
                    force = self.batch_forces[left][:, :, index:index + 1]
                elif right in self.batch_forces and left in self.batch_indices[right]:
                    index = self.batch_indices[right][left]
                    force = self.batch_forces[right][:, :, index:index + 1]
                else:
                    force = self.forces[tuple(sorted((left, right)))][0]
                results.append((force.abs() > force_threshold).any(dim=-1).any(dim=-1).any(dim=-1))
        value = torch.stack(results).any(dim=0)
        return value if env_id is None else bool(value[env_id])

    def get_objects_in_contact_with(self, body, candidates, force_threshold=.1, env_id=None):
        return [name for name in candidates
                if self.in_contact(body, name, force_threshold, env_id=0 if env_id is None else env_id)]


def _root_callable(func):
    while isinstance(func, partial):
        func = func.func
    return func


def eligibility_reasons(env, placement, grabbed, timeout):
    """Admit the tested primitive set, not arbitrary registered predicates."""
    reasons = []
    if env.num_envs != 1 or not str(env.device).startswith("cuda"):
        reasons.append("CPU task snapshots require one CUDA environment")
    if not getattr(env, "_omy_query_cache", False) or not getattr(env, "_omy_compact_step", False):
        reasons.append("requires audited boundary queries and compact observations")
    if not getattr(env.cfg, "omy_shared_placement_checks", False):
        reasons.append("requires shared placement checks")
    if getattr(env.cfg, "omy_recording_profile", "full") != "compact":
        reasons.append("requires compact state-only recording")
    if env.cfg.episode_length_s / env.step_dt >= 2**24:
        reasons.append("episode step count exceeds the exact packed float32 clock range")
    placements = []
    for name in env.termination_manager.active_terms:
        term = env.termination_manager.get_term_cfg(name)
        func = _root_callable(term.func)
        if func is placement:
            placements.append(term)
        elif func is not timeout:
            reasons.append(f"termination/{name} is not audited for CPU snapshots")
    if len(placements) != 1:
        reasons.append("requires exactly one resting-in-container termination")
    else:
        term = placements[0]
        try:
            bound = inspect.signature(term.func).bind(env, **term.params)
            bound.apply_defaults()
            if not all(isinstance(bound.arguments[name], str) for name in ("object", "container")):
                reasons.append("placement requires one named object and container")
        except TypeError:
            reasons.append("cannot resolve placement parameters")
    for index, subtask in enumerate(getattr(env.cfg, "subtasks", None) or []):
        conditions = getattr(subtask, "conditions", None)
        if not isinstance(conditions, dict):
            reasons.append(f"subtask/{index} structure is not audited")
            continue
        for sequence in conditions.values():
            for condition in sequence:
                func = condition[0] if isinstance(condition, tuple) else condition
                if _root_callable(func) not in (placement, grabbed):
                    reasons.append(f"subtask/{index} predicate is not audited for CPU snapshots")
    return reasons


class TaskSnapshot:
    def __init__(self, env, world):
        from robolab.core.sensors.contact_sensor_utils import get_batch_contact_sensor, get_contact_sensor_with_order
        from robolab.core.task.predicate_logic import _untransformed_hull
        from robolab.core.world.world_state import WorldState

        self.env = env
        self.active = self.current = False
        self.boundaries = self.packed_bytes = 0
        dependencies = env._omy_plan["dependencies"]
        self.objects = sorted(set(dependencies["poses"]) | set(dependencies["velocities"]))
        if any(name not in env.scene.rigid_objects for name in self.objects):
            raise ValueError("CPU snapshot pose/velocity dependencies must be rigid objects")
        if set(dependencies["joints"]) - {"robot"}:
            raise ValueError("CPU snapshot joint dependencies must belong to robot")
        pairs = {tuple(pair) for pair in dependencies["contacts"]}
        self.sensors = {pair: get_contact_sensor_with_order(env.scene, *pair) for pair in sorted(pairs)}
        self.batches = {}
        for name, path in (env.cfg.contact_gripper or {}).items():
            if not isinstance(path, str):
                continue
            sensor = get_batch_contact_sensor(env.scene, name)
            if sensor is None:
                continue
            labels = [n for n in env.cfg.contact_object_list if n != "table" and hasattr(env.cfg.scene, n)]
            paths = list(sensor.cfg.filter_prim_paths_expr)
            expected = [getattr(env.cfg.scene, n).prim_path for n in labels]
            force = sensor.data.force_matrix_w
            if (len(paths) == len(expected) == len(set(paths)) and set(paths) == set(expected)
                    and force is not None and force.ndim == 4 and force.shape[2] == len(paths)):
                self.batches[name] = (sensor, {n: paths.index(p) for n, p in zip(labels, expected)})
        for sensor, _ in self.sensors.values():
            force = sensor.data.force_matrix_w
            if force is None or force.ndim != 4 or force.shape[0] != 1 or force.shape[-1] != 3:
                raise ValueError("CPU snapshot requires initialized four-dimensional contact matrices")

        class CpuWorld(_CpuQueries, WorldState):
            pass

        host = self.world = CpuWorld.__new__(CpuWorld)
        host.env = SimpleNamespace(device=torch.device("cpu"), num_envs=1, step_dt=env.step_dt,
                                   cfg=env.cfg, scene=env.scene, episode_length_buf=torch.zeros(1, dtype=torch.long))
        host.cfg, host.device, host.num_envs, host.step_dt = env.cfg, host.env.device, 1, env.step_dt
        host.episode_length_buf = host.env.episode_length_buf
        host._predicate_state, host._local_geometry_cache = {}, {}
        host._boundary_query_cache = None
        host._untransformed_hull_cache = {
            name: type(hull)(*(value.cpu() for value in hull))
            for name in self.objects for hull in [_untransformed_hull(world, name)]
        }
        host.batch_indices = {name: indices for name, (_, indices) in self.batches.items()}

    def capture(self):
        items, layout = [], []

        def add(kind, key, value):
            layout.append((kind, key, value.shape, value.numel()))
            items.append(value.reshape(-1))

        for name in self.objects:
            data = self.env.scene[name].data
            add("pose", name, torch.cat((data.root_pos_w, data.root_quat_w, data.root_vel_w), dim=-1))
        add("joints", None, self.env.scene["robot"].data.joint_pos)
        add("origins", None, self.env.scene.env_origins)
        add("step", None, self.env.episode_length_buf.to(torch.float32))
        for pair, (sensor, _) in self.sensors.items():
            add("force", pair, sensor.data.force_matrix_w)
        for name, (sensor, _) in self.batches.items():
            add("batch", name, sensor.data.force_matrix_w)
        packed = torch.cat(items).cpu()
        self.boundaries += 1
        self.packed_bytes += packed.numel() * packed.element_size()
        host, offset = self.world, 0
        host.poses, host.forces, host.batch_forces = {}, {}, {}
        for kind, key, shape, count in layout:
            value = packed[offset:offset + count].reshape(shape)
            offset += count
            if kind == "pose":
                host.poses[key] = value
            elif kind == "joints":
                host.joints = value
            elif kind == "origins":
                host.origins = value
            elif kind == "step":
                host.episode_length_buf = host.env.episode_length_buf = value.to(torch.long)
            elif kind == "force":
                host.forces[key] = (value, self.sensors[key][1])
            elif kind == "batch":
                host.batch_forces[key] = value
        self.current = True

    @contextmanager
    def boundary(self):
        self.capture()
        self.active = True
        try:
            with boundary_queries(self.world):
                yield
        finally:
            self.active = False

    def invalidate(self):
        self.current = False
        self.world._predicate_state.clear()
        self.world._boundary_query_cache = None


def configure_task_snapshot(env):
    """Install instance-owned state only after the complete admission audit."""
    cfg = env.cfg
    cfg.omy_task_checks_effective = "native"
    cfg.omy_task_checks_fallback_reasons = []
    if getattr(cfg, "omy_task_checks", "native") != "cpu":
        return None
    import isaaclab.envs.mdp as mdp
    from robolab.core.events.subtask_recorder import SubtaskCompletionRecorderTerm
    from robolab.core.task import conditionals
    from robolab.core.world.world_state import get_world
    from .requirements import default_recorder_types

    reasons = eligibility_reasons(env, conditionals.object_resting_in_container,
                                  conditionals.object_grabbed, mdp.time_out)
    allowed, _ = default_recorder_types()
    for name, term in env.recorder_manager._terms.items():
        if type(term) not in allowed or type(term).__name__ in (
                "PostStepBBoxRecorder", "PostStepEndEffectorPoseRecorder", "PostStepRobotRootPoseRecorder"):
            reasons.append(f"recorder/{name} is not audited for CPU snapshots")
    snapshot = None
    if not reasons:
        try:
            snapshot = TaskSnapshot(env, get_world(env))
        except (ValueError, KeyError, TypeError) as error:
            reasons.append(str(error))
    if reasons:
        cfg.omy_task_checks_fallback_reasons = list(dict.fromkeys(reasons))
        return None
    for term in env.recorder_manager._terms.values():
        if type(term) is SubtaskCompletionRecorderTerm and term._event_tracker is not None:
            term._event_tracker.to("cpu")
    cfg.omy_task_checks_effective = "cpu"
    return snapshot
