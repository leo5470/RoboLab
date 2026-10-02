"""OMY-only pose observations, sharing work within one observation evaluation.

Keep DROID's observation names, frames and units. Read the articulation instead
of updating a second rigid-body view through the frame-transform sensor. Never
cache across compute() calls: reset/teleport can change a pose without a step.
"""

from functools import wraps


def _compute_poses(env):
    from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

    robot = env.scene["robot"]
    data = robot.data
    body_idx = data.body_names.index("base_link")
    position, orientation = subtract_frame_transforms(
        data.root_pos_w, data.root_quat_w,
        data.body_pos_w[:, body_idx], data.body_quat_w[:, body_idx],
    )
    target = next(frame for frame in env.scene["frames"].cfg.target_frames
                  if frame.name == "eef_frame")
    if not target.prim_path.endswith("/base_link"):
        raise ValueError("OMY optimized observations require eef_frame on base_link")
    # Build offset tensors once per installed cache, not once per GPU step.
    offset_key = (tuple(target.offset.pos), tuple(target.offset.rot),
                  orientation.device, orientation.dtype, env.num_envs)
    owner = getattr(env, "_omy_observation_pass", None)
    if owner is not None and owner.offset_key == offset_key:
        offset_pos, offset_rot = owner.offset_tensors
    else:
        offset_pos = position.new_tensor(target.offset.pos).expand(env.num_envs, -1)
        offset_rot = orientation.new_tensor(target.offset.rot).expand(env.num_envs, -1)
        if owner is not None:
            owner.offset_key = offset_key
            owner.offset_tensors = offset_pos, offset_rot
    eef_position, eef_orientation = combine_frame_transforms(
        position, orientation, offset_pos, offset_rot,
    )
    return position, orientation, eef_position, eef_orientation


def _poses(env):
    owner = getattr(env, "_omy_observation_pass", None)
    if owner is None or owner.values is None:
        return _compute_poses(env)
    if "poses" not in owner.values:
        owner.values["poses"] = _compute_poses(env)
    return owner.values["poses"]


def ee_pos(env):
    return _poses(env)[0]


def ee_quat(env):
    return _poses(env)[1]


def eef_pos(env):
    return _poses(env)[2]


def eef_quat(env):
    return _poses(env)[3]


class ObservationPassCache:
    """Scoped to a runner, with cache lifetime restricted to one compute call."""

    def __init__(self, env):
        if hasattr(env, "_omy_observation_pass"):
            raise RuntimeError("OMY observation cache already installed")
        self.env = env
        self.values = None
        self.offset_key = self.offset_tensors = None
        self.manager = env.observation_manager
        self.original = self.manager.compute
        self.had_local = "compute" in vars(self.manager)
        self.local = vars(self.manager).get("compute")

        @wraps(self.original)
        def compute(*args, **kwargs):
            previous, self.values = self.values, {}
            try:
                return self.original(*args, **kwargs)
            finally:
                self.values = previous

        self.wrapper = compute
        env._omy_observation_pass = self
        self.manager.compute = compute

    def close(self):
        if self.manager.compute is self.wrapper:
            if self.had_local:
                self.manager.compute = self.local
            else:
                del self.manager.compute
        if getattr(self.env, "_omy_observation_pass", None) is self:
            del self.env._omy_observation_pass
        self.values = self.offset_key = self.offset_tensors = None
