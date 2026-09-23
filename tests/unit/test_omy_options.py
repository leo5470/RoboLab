"""OMY CLI checks; run with --confcutdir=tests/unit without starting Isaac Sim."""

import hashlib
import json

import pytest

from robolab.core.omy_teleop.options import ARM_JOINTS, build_parser, calibration_metadata, validate_args


def test_defaults_match_upstream():
    parser = build_parser()
    args = parser.parse_args([])
    validate_args(parser, args)
    assert args.source == "serial"
    assert args.baudrate == 4_000_000
    assert args.read_hz == 100
    assert args.stale_s == 0.5
    assert not args.auto_zero
    assert not args.record and not args.record_images
    assert args.rendering_mode == "performance"
    assert args.episode_length_s == 10_000
    assert ARM_JOINTS == tuple(f"panda_joint{i}" for i in range(1, 8))


@pytest.mark.parametrize("argv", [
    ["--stale-s", "0"], ["--read-hz", "nan"], ["--vel-scale", "inf"],
    ["--auto-zero-settle-s", "-1"], ["--steps", "-1"], ["--num-episodes", "0"],
    ["--video-every", "0"], ["--baudrate", "0"], ["--tcp-port", "65536"],
    ["--video-port", "0"], ["--episode-length-s", "0"], ["--record-images"], ["--save-failures"],
    ["--calib", "/nonexistent/omy_calibration.json"],
])
def test_invalid_options_rejected(argv):
    parser = build_parser()
    with pytest.raises(SystemExit):
        validate_args(parser, parser.parse_args(argv))


def test_calibration_provenance(tmp_path):
    path = tmp_path / "calibration.json"
    contents = {"omy_sign": {"joint_1": -1.0}, "wrist_mode": "direct"}
    path.write_text(json.dumps(contents))
    metadata = calibration_metadata(path)
    assert metadata["path"] == str(path.resolve())
    assert metadata["contents"] == contents
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert calibration_metadata("")["contents"] is None
