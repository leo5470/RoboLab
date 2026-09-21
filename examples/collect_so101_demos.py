#!/usr/bin/env python3
"""Collect DROID demonstrations using so101 control; see docs/so101_teleoperation.md."""


def main():
    import cv2  # noqa: F401  # Must precede imports which may load Isaac Lab.

    from robolab.core.teleop.collector import launch

    return launch("so101")


if __name__ == "__main__":
    raise SystemExit(main())
