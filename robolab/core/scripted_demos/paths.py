"""Where scripted-demonstration data and external tools live.

Outputs default to RoboLab's gitignored ``output/``; each location can be moved with an environment variable
(see docs/scripted_demos.md, "Data locations").
"""

import os
import pathlib

ROBOLAB_DIR = pathlib.Path(__file__).resolve().parents[3]


def _location(variable: str, default: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(os.environ.get(variable) or default)


def grasp_data_root() -> pathlib.Path:
    """Meshes, GraspDataGen runs, grasp library, trial and screening logs."""
    return _location("GRASP_DATA_DIR", ROBOLAB_DIR / "output" / "grasp_data")


def scripted_demos_root() -> pathlib.Path:
    """One directory per object family: ``pps.hdf5``, videos, attempt logs, manifests."""
    return _location("SCRIPTED_DEMOS_DIR", ROBOLAB_DIR / "output" / "scripted_demos")


def kit_portable_root() -> pathlib.Path:
    """Kit's data/cache/logs root for the simulator scripts and GraspDataGen runs."""
    return _location("KIT_PORTABLE_ROOT", ROBOLAB_DIR / "output" / "kit_portable")


def demo_viewer_cache() -> pathlib.Path:
    """Encoded episode videos of ``scripts/scripted_demos/demo_viewer.py``."""
    return _location("DEMO_VIEWER_CACHE", ROBOLAB_DIR / "output" / "demo_viewer_cache")


def graspdatagen_dir() -> pathlib.Path:
    """The GraspDataGen checkout; only the grasp tools need it. Default: the ``third_party/GraspDataGen`` submodule,
    pinned at ``objects.GRASPDATAGEN_COMMIT`` (``git submodule update --init third_party/GraspDataGen``)."""
    path = _location("GRASPDATAGEN_DIR", ROBOLAB_DIR / "third_party" / "GraspDataGen")
    if not (path / "scripts" / "graspgen").is_dir():
        raise RuntimeError(f"no GraspDataGen checkout at {path}: run `git submodule update --init third_party/GraspDataGen` "
                           f"or set GRASPDATAGEN_DIR (see docs/scripted_demos.md)")
    return path
