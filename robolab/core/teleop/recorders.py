"""Aligned SO-101 input diagnostics; imported only after Isaac Sim starts."""

from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


class LeaderDiagnosticsRecorder(RecorderTerm):
    def record_pre_step(self):
        snapshot = getattr(self._env, "so101_diagnostics", None)
        if snapshot is None:
            return None, None
        return "teleop/so101", snapshot


@configclass
class LeaderDiagnosticsRecorderCfg(RecorderTermCfg):
    class_type: type = LeaderDiagnosticsRecorder
