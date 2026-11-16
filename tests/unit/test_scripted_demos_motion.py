"""The executor's state machine on a fake arm (segment pacing, gripper timing, abort reasons)."""

import numpy as np

from robolab.core.scripted_demos.grasps import frames
from robolab.core.scripted_demos import motion
from robolab.core.scripted_demos.select import Plan


class FakeIK:
    """Tracks the last reference perfectly (or lags by a fixed offset), like an ideal arm."""

    def __init__(self, start, offset=0.0):
        self.T = start.copy()
        self.offset = offset

    def joint_target(self, T_target):
        self.T = T_target.copy()
        self.T[0, 3] += self.offset
        return np.zeros(7)

    def ee_pose(self):
        return self.T.copy()

    def pose_error(self, T_target):
        return float(np.linalg.norm(self.T[:3, 3] - T_target[:3, 3])), frames.rotation_angle(self.T[:3, :3], T_target[:3, :3])


def make_plan():
    down = frames.rot(1, np.pi / 2)
    wp = {}
    for name, p in {"pregrasp": [0.5, 0.0, 0.30], "grasp": [0.5, 0.0, 0.22], "lift": [0.5, 0.0, 0.40],
                    "preplace": [0.55, 0.2, 0.40], "seat": [0.55, 0.2, 0.25]}.items():
        wp[name] = frames.homogeneous(down, p)
    return Plan(grasp_id=0, waypoints=wp, joints={}, backoff=0.08, lift_extra=0.0, preplace_yaw=0.0, costs={},
                held_object_bottom_offset=0.1)


def run(executor, ik, object_pose_fn, closedness_fn, max_steps=400):
    names = []
    for step in range(max_steps):
        a = executor.action()
        assert a.shape == (8,) and a.dtype == np.float32 and a[7] in (0.0, 1.0)
        names.append(executor.name)
        failure = executor.update(object_pose_fn(step, executor), closedness_fn(executor), True)
        if failure or executor.finished or (executor.name == "settle" and executor.k >= 3):
            return failure, names
    return "loop", names


def test_interpolate_endpoints_and_midpoint():
    A = frames.homogeneous(np.eye(3), [0, 0, 0])
    B = frames.homogeneous(frames.rot(2, np.pi / 2), [1, 2, 3])
    assert np.allclose(motion.interpolate(A, B, 0.0), A)
    assert np.allclose(motion.interpolate(A, B, 1.0), B)
    M = motion.interpolate(A, B, 0.5)
    assert np.allclose(M[:3, 3], [0.5, 1, 1.5])
    assert np.isclose(frames.rotation_angle(A[:3, :3], M[:3, :3]), np.pi / 4)


def test_segment_pacing_is_within_bounds_and_random():
    params = motion.MotionParams()
    totals = set()
    for seed in range(30):
        steps = motion.segment_steps(np.random.default_rng(seed), params)
        assert set(steps) == set(motion.NOMINAL)
        total = sum(steps.values())
        assert params.motion_steps[0] - 10 <= total <= params.motion_steps[1] + 10
        totals.add(total)
    assert len(totals) > 5


def test_full_sequence_reaches_settle_with_gripper_timing():
    plan = make_plan()
    ik = FakeIK(frames.homogeneous(frames.rot(1, np.pi / 2), [0.36, 0.0, 0.47]))
    ex = motion.Executor(plan, ik, np.random.default_rng(0))
    held = {}

    def object_pose(step, e):  # the object follows the gripper once grasped
        if e.name in motion.HOLDING:
            return frames.homogeneous(t=ik.ee_pose()[:3, 3] - [0, 0, 0.1])
        return frames.homogeneous(t=[0.5, 0.0, 0.1])

    failure, names = run(ex, ik, object_pose, lambda e: 0.5 if e.name in motion.HOLDING or e.name == "close" else 0.0)
    assert failure is None and ex.done_moving
    order = list(dict.fromkeys(names))
    assert order == list(motion.NOMINAL) + ["settle"]
    grip = {n: motion.GRIPPER[n] for n in order}
    assert grip["approach"] == 0.0 and grip["close"] == 1.0 and grip["seat"] == 1.0 and grip["open"] == 0.0
    planned = sum(ex.steps.values())
    assert sum(1 for n in names if n != "settle") == planned  # perfect tracking: no settle allowance used


def test_slip_when_fingers_close_on_nothing():
    plan = make_plan()
    ik = FakeIK(frames.homogeneous(frames.rot(1, np.pi / 2), [0.36, 0.0, 0.47]))
    ex = motion.Executor(plan, ik, np.random.default_rng(1))
    failure, names = run(ex, ik, lambda s, e: frames.homogeneous(t=[0.5, 0.0, 0.1]), lambda e: 1.0 if e.name != "pregrasp" and e.name != "approach" else 0.0)
    assert failure == "slip" and names[-1] == "close"


def test_pushing_the_object_before_closing_aborts():
    plan = make_plan()
    ik = FakeIK(frames.homogeneous(frames.rot(1, np.pi / 2), [0.36, 0.0, 0.47]))
    ex = motion.Executor(plan, ik, np.random.default_rng(4))
    failure, names = run(ex, ik, lambda s, e: frames.homogeneous(t=[0.5 + 0.002 * s, 0.0, 0.1]), lambda e: 0.0)
    assert failure == "collision_abort" and names[-1] in ("pregrasp", "approach")


def test_tracking_error_when_the_arm_cannot_reach():
    plan = make_plan()
    ik = FakeIK(frames.homogeneous(frames.rot(1, np.pi / 2), [0.36, 0.0, 0.47]), offset=0.05)
    ex = motion.Executor(plan, ik, np.random.default_rng(2))
    failure, names = run(ex, ik, lambda s, e: frames.homogeneous(t=[0.5, 0.0, 0.1]), lambda e: 0.0)
    assert failure == "tracking_error" and names[-1] == "pregrasp"
    assert ex.max_tracking["pos_m"] >= 0.049


def test_transfer_mode_lifts_and_holds():
    plan = make_plan()
    ik = FakeIK(frames.homogeneous(frames.rot(1, np.pi / 2), [0.36, 0.0, 0.47]))
    ex = motion.Executor(plan, ik, np.random.default_rng(3), mode="transfer")
    def object_pose(step, e):
        if e.name in motion.HOLDING:
            return frames.homogeneous(t=ik.ee_pose()[:3, 3] - [0, 0, 0.1])
        return frames.homogeneous(t=[0.5, 0.0, 0.1])

    failure, names = run(ex, ik, object_pose, lambda e: 0.4 if e.name not in ("pregrasp", "approach") else 0.0)
    assert failure is None and ex.finished
    assert list(dict.fromkeys(names)) == list(motion.TRANSFER)
