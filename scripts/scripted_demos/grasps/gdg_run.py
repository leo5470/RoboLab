"""Run a GraspDataGen script unchanged, with Kit's data/cache/logs redirected to our own directory.

GraspDataGen builds ``AppLauncher(headless=...)`` itself, so Kit would write into the venv (possibly another user's tree, R6).
This adds ``--portable-root`` to every AppLauncher's kit args, then runs the script as ``__main__``.

    python scripts/scripted_demos/grasps/gdg_run.py <GraspDataGen script.py> [script args...]
"""

import os
import pathlib
import runpy
import sys


def main() -> None:
    script = pathlib.Path(sys.argv[1]).resolve()
    root = pathlib.Path(os.environ["KIT_PORTABLE_ROOT"])
    root.mkdir(parents=True, exist_ok=True)

    from isaaclab.app import AppLauncher

    original = AppLauncher.__init__

    def init(self, launcher_args=None, **kwargs):
        kwargs["kit_args"] = f"{kwargs.get('kit_args', '')} --portable-root {root}".strip()
        original(self, launcher_args, **kwargs)

    AppLauncher.__init__ = init
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
