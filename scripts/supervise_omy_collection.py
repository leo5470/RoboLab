#!/usr/bin/env python3
"""Keep collection, cleanup, offline processing and console logging in order."""

import argparse
import os
import selectors
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robolab.core.omy_teleop.collection_handoff import confirm_collector_exit, new_run_id, replace_report


class Supervisor:
    def __init__(self, log=None):
        self.log = log
        self.child = None
        self.stage = "collection"
        self.interrupts = 0
        self.cancelled = False
        self.messages = []

    def write(self, data):
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
        if self.log is not None:
            self.log.write(data)
            self.log.flush()

    def say(self, message):
        self.write((message + "\n").encode())

    def interrupt(self, signum, frame):
        self.interrupts += 1
        if self.stage == "collection" and self.interrupts == 1 and signum == signal.SIGINT:
            self.messages.append("Stopping collection; waiting for recording and simulator cleanup before postprocessing.")
            forwarded = signal.SIGINT
        else:
            self.cancelled = True
            forwarded = (signal.SIGINT if self.stage == "postprocessing" and self.interrupts == 1
                         else signal.SIGTERM if self.interrupts < 3 else signal.SIGKILL)
            self.messages.append("Cancelling; automatic postprocessing will not continue.")
        if self.child is not None:
            try:
                os.killpg(self.child.pid, forwarded)
            except ProcessLookupError:
                pass
        else:
            # An interrupt between stages cancels the handoff rather than
            # becoming a delayed interrupt delivered to a new renderer.
            self.cancelled = True

    def flush_messages(self):
        while self.messages:
            self.say(self.messages.pop(0))

    def run(self, command):
        # Only the supervisor receives the terminal's SIGINT. It forwards each
        # interrupt once to the current child and its renderer descendants.
        if self.cancelled:
            return 130
        self.child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        try:
            if self.cancelled:
                os.killpg(self.child.pid, signal.SIGTERM)
            with self.child.stdout as stream, selectors.DefaultSelector() as ready:
                ready.register(stream, selectors.EVENT_READ)
                while True:
                    self.flush_messages()
                    if not ready.select(timeout=0.2):
                        continue
                    data = stream.read1(64 * 1024)
                    if not data:
                        break
                    self.write(data)
            status = self.child.wait()
            self.flush_messages()
            return status if status >= 0 else 128 - status
        finally:
            # A broken output stream or logging failure must not leave a
            # detached collector/renderer running after its supervisor exits.
            if self.child.poll() is None:
                try:
                    os.killpg(self.child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.child.pid, signal.SIGKILL)
                    self.child.wait()
            self.child = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--postprocess-device", default="cuda:1")
    parser.add_argument("--postprocess-skip-failed", action="store_true")
    parser.add_argument("--postprocess-preview-videos", action="store_true")
    parser.add_argument("collector", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.collector
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("A collector command is required")
    log = args.log.open("ab") if args.log else None
    supervisor = Supervisor(log)
    previous = {sig: signal.signal(sig, supervisor.interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if args.log:
            supervisor.say(f"Console log: {args.log}")
        with tempfile.TemporaryDirectory(prefix="omy_collect_handoff_") as temporary:
            handoff_path = Path(temporary) / "handoff.json"
            run_id = new_run_id()
            status = supervisor.run([*command, "--collection-handoff", str(handoff_path),
                                     "--collection-run-id", run_id])
            if supervisor.cancelled:
                return 130
            if status not in (0, 130):
                supervisor.say(f"Collection failed (status {status}); automatic postprocessing skipped.")
                return status
            try:
                handoff = confirm_collector_exit(handoff_path, run_id=run_id, status=status)
            except (OSError, ValueError, KeyError) as error:
                supervisor.say(f"Collection cleanup was not confirmed; postprocessing skipped: {error}")
                return status if args.no_postprocess else status or 2
            handoff["postprocessing"] = {"device": args.postprocess_device,
                "skip_failed": args.postprocess_skip_failed, "preview_videos": args.postprocess_preview_videos}
            replace_report(handoff["handoff"], handoff)
            if args.no_postprocess:
                supervisor.say("Automatic postprocessing disabled (--no-postprocess).")
                return status
            supervisor.stage = "postprocessing"
            supervisor.interrupts = 0
            if supervisor.cancelled:
                return 130
            script = Path(__file__).resolve().with_name("postprocess_omy_session.py")
            supervisor.say(f"Starting postprocessing on {args.postprocess_device}; "
                           "you can start another collection in a separate terminal.")
            processing_flags = (["--skip-failed"] if args.postprocess_skip_failed else [])
            processing_flags += (["--preview-videos"] if args.postprocess_preview_videos else [])
            processed = supervisor.run([sys.executable, "-u", str(script), "--handoff", handoff["handoff"],
                                        "--device", args.postprocess_device, *processing_flags])
            if processed:
                supervisor.say(f"Postprocessing did not complete (status {processed}).")
                supervisor.say(f"Postprocessing output directory: {handoff['output_dir']}")
                supervisor.say(f"Processing report (if created): {handoff['output_dir']}/summary.json")
            return 130 if supervisor.cancelled else processed or status
    except OSError as error:
        supervisor.say(f"Collection supervisor failed: {error}")
        return 2
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if log is not None:
            log.close()


if __name__ == "__main__":
    raise SystemExit(main())
