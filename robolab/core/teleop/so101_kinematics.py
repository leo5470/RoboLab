"""Load the standalone reader's FK without relying on a scripts package."""

import importlib.util
from pathlib import Path


def leader_kinematics(urdf, signs=None, offsets=None):
    # The helper deliberately lives outside the installed robolab package so
    # its separate Python environment never needs to import RoboLab/Isaac.
    path = Path(__file__).resolve().parents[3] / "scripts" / "so101_leader_reader.py"
    spec = importlib.util.spec_from_file_location("_robolab_so101_reader_fk", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SO-101 kinematics from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UrdfKinematics(urdf, signs=signs, offsets=offsets)
