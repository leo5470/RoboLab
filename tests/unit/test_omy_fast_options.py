"""Opt-in OMY launch contracts without starting Kit."""

import pytest

from robolab.core.omy_teleop.collection import build_parser, validate_args


def parse(*argv):
    parser = build_parser()
    args = parser.parse_args(list(argv))
    validate_args(parser, args)
    return args


def test_reference_remains_default():
    args = parse()
    assert args.solver_position_iterations is None
    assert args.stepping_profile == "ordinary"
    assert args.recording_profile == "full"
    assert args.render_interval == 2


def test_candidate_configuration_is_explicit():
    args = parse("--device", "cuda:0", "--control-hz", "15", "--solver-position-iterations", "16",
                 "--stepping-profile", "held-target", "--recording-profile", "compact", "--render-interval", "4")
    assert args.control_hz == 15 and args.render_interval == 4
    assert args.stepping_profile == "held-target" and args.recording_profile == "compact"


@pytest.mark.parametrize("argv", [
    ("--control-hz", "60", "--render-interval", "4"),
    ("--control-hz", "30", "--stepping-profile", "held-target"),
    ("--render-interval", "6"), ("--solver-position-iterations", "12"),
    ("--episode-buffer-mb", "nan"), ("--episode-buffer-mb", "0"),
])
def test_invalid_profile_rejected(argv):
    with pytest.raises(SystemExit):
        parse(*argv)
