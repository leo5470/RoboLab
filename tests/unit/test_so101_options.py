import pytest

from robolab.core.teleop.collector import build_parser
from robolab.core.teleop.so101_options import validate_arguments


class Launcher:
    @staticmethod
    def add_app_launcher_args(parser):
        pass


def test_mock_cannot_record_or_generate_unlabelled_demo():
    parser = build_parser("so101", Launcher)
    with pytest.raises(SystemExit):
        validate_arguments(parser, parser.parse_args(["--leader-mock"]))
    with pytest.raises(SystemExit):
        validate_arguments(parser, parser.parse_args(["--smoke-steps", "10", "--no-record"]))


def test_mock_smoke_needs_no_device_or_hardware_dependency():
    parser = build_parser("so101", Launcher)
    args = parser.parse_args(["--leader-mock", "--no-record", "--smoke-steps", "5"])
    validate_arguments(parser, args)
    assert args.retarget_config.orientation_mode == "fixed"
    assert args.filename == "so101_demos.hdf5"


def test_default_control_is_server_local_and_requires_a_device():
    parser = build_parser("so101", Launcher)
    args = parser.parse_args(["--leader-mock", "--no-record"])
    assert args.control_source == "evdev"
    with pytest.raises(SystemExit):
        validate_arguments(parser, args)


def test_launcher_can_consume_its_namespace_without_mutating_collector_args(monkeypatch):
    import sys
    from types import SimpleNamespace

    from robolab.core.teleop import collector

    calls = []

    class MutatingLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            # Isaac's helper parses known options before adding its own flags.
            known, unknown = parser.parse_known_args(["--no-record"])
            assert known.record is False and not unknown
            parser.add_argument("--livestream", type=int, default=0)
            parser.add_argument("--headless", action="store_true")
            parser.add_argument("--device", default="cpu")
            parser.add_argument("--kit_args", default="")

        def __init__(self, args):
            vars(args).pop("livestream")
            self.app = SimpleNamespace(close=lambda: calls.append("closed"))

    monkeypatch.setitem(sys.modules, "isaaclab.app", SimpleNamespace(AppLauncher=MutatingLauncher))

    def collect(args, app):
        assert args.livestream == 0
        assert not args.record
        calls.append("collected")
        return 0

    monkeypatch.setattr(collector, "collect", collect)
    assert collector.launch("keyboard", ["--no-record"]) == 0
    assert calls == ["collected", "closed"]


def test_wrist_roll_mode_cli_override():
    parser = build_parser("so101", Launcher)
    args = parser.parse_args([
        "--leader-mock", "--no-record", "--smoke-steps", "5",
        "--orientation-mode", "pose", "--orientation-mode", "wrist-roll",
    ])
    validate_arguments(parser, args)
    assert args.retarget_config.orientation_mode == "wrist-roll"
