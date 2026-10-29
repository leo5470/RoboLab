#!/usr/bin/env python3
"""Filter collected and MimicGen demos into an exact-size session dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robolab.core.omy_teleop.final_dataset import main

if __name__ == "__main__":
    raise SystemExit(main())
