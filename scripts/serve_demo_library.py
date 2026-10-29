#!/usr/bin/env python3
"""Serve the demonstration website or run its selected-only video service."""
import argparse
from pathlib import Path
import sys
import signal
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collected-root", action="append", type=Path)
    p.add_argument("--generated-root", action="append", type=Path)
    p.add_argument("--state-dir", type=Path, default=root / "output/demo_library")
    p.add_argument("--refresh-seconds", type=float, default=30)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--once", action="store_true")
    p.add_argument("--run", type=Path)
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    args = p.parse_args()
    from robolab.core.omy_postprocess.library import DemoCatalog
    catalog = DemoCatalog(args.collected_root or [root / "output/omy_teleop"],
                          args.generated_root or [root / "output/omy_mimicgen"], args.state_dir, args.refresh_seconds)
    if args.refresh_seconds <= 0 or len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        p.error("Refresh interval must be positive; GPU IDs must be distinct nonnegative integers")
    if args.run:
        if not catalog.allowed(args.run):
            p.error("--run must be inside a configured dataset root")
        catalog.selection(args.run / "selection.json")
    if args.worker:
        from robolab.core.omy_postprocess.library_worker import VideoQueue, service_lock, drain
        stopping = threading.Event()
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: stopping.set())
        try:
            with service_lock(args.state_dir):
                queue = VideoQueue(catalog)
                queue.recover()
                # Reconcile during a long backfill too: future runs and changed
                # selections must not wait for the current queue to drain.
                def reconcile_loop():
                    while not stopping.wait(max(1, args.refresh_seconds)):
                        try:
                            queue.reconcile()
                            queue.report()
                        except Exception as error:
                            print(f"Catalog reconciliation will retry: {error}", flush=True)
                monitor = threading.Thread(target=reconcile_loop, daemon=True)
                monitor.start()
                while not stopping.is_set():
                    queue.reconcile()
                    drain(queue, args.gpus, args.run, stopping)
                    if args.once:
                        counts = queue.report()
                        wanted = [v for k,v in counts.items() if not args.run or Path(k) == args.run.resolve()]
                        return 1 if any(v["ready"] != v["selected"] for v in wanted) else 0
                    stopping.wait(max(1, args.refresh_seconds))
        except BlockingIOError:
            print("Video service already running; it will reconcile new selections automatically.", flush=True)
            return 0
    else:
        import uvicorn
        from dashboard.app import create_app
        app = create_app(demo_catalog=catalog)
        print(f"Demonstrations: http://{args.host}:{args.port}/demonstrations", flush=True)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
