"""The objects in the first scripted-demo batch and where their data lives."""

from __future__ import annotations

import dataclasses
import functools
import json
import pathlib

import numpy as np

from robolab.core.scripted_demos import paths
from robolab.core.scripted_demos.paths import grasp_data_root, scripted_demos_root  # noqa: F401  (re-exported)

ROBOLAB_DIR = paths.ROBOLAB_DIR
GRASPDATAGEN_COMMIT = "8e09005"  # the GraspDataGen commit the grasp libraries were built with

# Held-out evaluation layouts (00, 05, ..., 45 of each Simple-PnP family). Training layouts never use these.
EVAL_LAYOUTS = tuple(f"{i:02d}" for i in range(0, 50, 5))


@dataclasses.dataclass(frozen=True)
class BatchObject:
    index: int  # object catalog index
    name: str  # catalog name, also the scene prim / WorldState body name
    family: str  # task-class prefix shared by all layouts, e.g. SimplePnp303Pitcher
    usd_path: str  # relative to the RoboLab checkout
    control: bool = False  # box-like control object (top-down/side grasps are enough)

    @property
    def key(self) -> str:
        return f"{self.index}_{self.name}"

    def task(self, layout: str) -> str:
        return f"{self.family}{layout}Task"


BATCH: tuple[BatchObject, ...] = (
    BatchObject(303, "pitcher", "SimplePnp303Pitcher", "assets/objects/ycb/pitcher.usd"),
    BatchObject(301, "mug", "SimplePnp301Mug", "assets/objects/ycb/mug.usd"),
    BatchObject(283, "utilityjug_a01", "SimplePnp283UtilityjugA01", "assets/objects/vomp/utilityjug_a01/utilityjug_a01.usd"),
    BatchObject(284, "utilityjug_a02", "SimplePnp284UtilityjugA02", "assets/objects/vomp/utilityjug_a02/utilityjug_a02.usd"),
    BatchObject(230, "milkjug_a01", "SimplePnp230MilkjugA01", "assets/objects/vomp/milkjug_a01/milkjug_a01.usd"),
    BatchObject(231, "milkjug_a02", "SimplePnp231MilkjugA02", "assets/objects/vomp/milkjug_a02/milkjug_a02.usd"),
    BatchObject(294, "cheez_it", "SimplePnp294CheezIt", "assets/objects/ycb/cheez_it.usd", control=True),
)
BY_INDEX = {o.index: o for o in BATCH}
BY_NAME = {o.name: o for o in BATCH}


def get(key: int | str) -> BatchObject:
    """Look an object up by catalog index (``303``/``"303"``), name, or ``<index>_<name>``."""
    if isinstance(key, int) or str(key).isdigit():
        return BY_INDEX[int(key)]
    key = str(key)
    if key in BY_NAME:
        return BY_NAME[key]
    head = key.split("_", 1)[0]
    if head.isdigit() and int(head) in BY_INDEX:
        return BY_INDEX[int(head)]
    raise KeyError(f"{key!r} is not in the scripted-demo batch")


def available_layouts(obj: BatchObject) -> list[str]:
    """Layout ids that have a generated task file, in order."""
    manifest = json.loads((ROBOLAB_DIR / "robolab/tasks/single_pnp/manifest.json").read_text())
    prefix = f"pnp_{obj.index:03d}_{obj.name}_"
    return sorted(t["task"][len(prefix) : len(prefix) + 2] for t in manifest["tasks"] if t["task"].startswith(prefix))


def training_layouts(obj: BatchObject, count: int = 10) -> list[str]:
    """The first ``count`` layouts that are not evaluation layouts: 01 02 03 04 06 07 08 09 11 12 for most objects
    (``count=100`` lists them all, in order)."""
    return [layout for layout in available_layouts(obj) if layout not in EVAL_LAYOUTS][:count]


@functools.lru_cache(maxsize=None)
def load_obj_vertices(path: pathlib.Path) -> np.ndarray:
    """Vertices of an OBJ file (as written by scripts/scripted_demos/grasps/export_meshes.py)."""
    rows = [line.split()[1:4] for line in pathlib.Path(path).read_text().splitlines() if line.startswith("v ")]
    return np.asarray(rows, dtype=np.float64)


def mesh_vertices(obj: BatchObject) -> np.ndarray:
    """The object's exported mesh vertices, in the RoboLab object frame."""
    return load_obj_vertices(grasp_data_root() / "meshes" / f"{obj.key}.obj")


def object_points(obj: BatchObject, max_points: int = 4000) -> np.ndarray:
    """``mesh_vertices``, evenly subsampled."""
    v = mesh_vertices(obj)
    return v[:: max(1, len(v) // max_points)]


@functools.lru_cache(maxsize=None)
def simulated_mass(key: str, default: float = 0.3) -> float:
    """Mass PhysX gives the object (``masses.json`` from scripts/scripted_demos/probe_objects.py)."""
    path = grasp_data_root() / "masses.json"
    return float(json.loads(path.read_text()).get(key, default)) if path.exists() else default
