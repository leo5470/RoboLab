"""Review images for scripted OMY sessions, for the demo viewer.

    SIM_GPU=0 scripts/scripted_demos/run.sh scripts/scripted_demos/render_omy_review.py <session dir> [...] \\
        [--camera-scale 0.5] [--overwrite]

OMY recordings hold states, not images (OMY collection renders cameras later). For every recording
``<session>/<layout task>/part_NNN/omy_demos.hdf5`` this renders the recorded states with leocheng's materializer
(``scripts/materialize_demo_images.py``: the recording's own env and policy cameras, frames from recorded states,
never re-stepped physics) into ``<session>/review/<layout task>/part_NNN/images.hdf5``, encodes one MP4 per episode
beside it (over-shoulder | wrist, ``demo_N.mp4``) for the demo viewer, then deletes the images (about 150 MB per
episode at half resolution) unless ``--keep-images``. Review only: the training export renders its own images.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

from robolab.core.scripted_demos import omy_review, paths


def parts(session: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in session.glob("*/part_*/omy_demos.hdf5") if "review" not in p.relative_to(session).parts)


def review_path(session: pathlib.Path, recording: pathlib.Path) -> pathlib.Path:
    return session / "review" / recording.parent.relative_to(session) / "images.hdf5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", nargs="+", type=pathlib.Path)
    parser.add_argument("--camera-scale", type=float, default=0.5, help="policy cameras at this fraction (720x1280)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-images", action="store_true", help="keep images.hdf5 after encoding the videos")
    args = parser.parse_args()
    kit = paths.kit_portable_root()
    kit.mkdir(parents=True, exist_ok=True)
    script = paths.ROBOLAB_DIR / "scripts" / "materialize_demo_images.py"
    failed = 0
    for session in args.sessions:
        session = session.resolve()
        listed = omy_review.entries(session)
        for recording in parts(session):
            out = review_path(session, recording)
            mine = [e for e in listed if e.recording == recording]
            if not args.overwrite and all(omy_review.review_video(session, e).exists() for e in mine):
                print(f"have the videos of {recording.parent}", flush=True)
                continue
            if out.exists() and not args.overwrite:
                encode_videos(session, mine, out, args.keep_images)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-u", str(script), "--input", str(recording), "--output", str(out), "--headless",
                   "--require-native15", "--camera-scale", str(args.camera_scale), "--hdf5-compression", "lzf",
                   *(["--overwrite"] if args.overwrite else []),
                   "--kit_args", f"--portable-root {kit} --/renderer/multiGpu/enabled=false --/renderer/activeGpu=0"]
            print(f"rendering {recording} -> {out}", flush=True)
            log = out.with_suffix(".log")
            with log.open("w") as stream:
                result = subprocess.run(cmd, cwd=paths.ROBOLAB_DIR, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                failed += 1
                print(f"FAILED ({result.returncode}): see {log}", flush=True)
                continue
            encode_videos(session, mine, out, args.keep_images)
    raise SystemExit(1 if failed else 0)


def encode_videos(session: pathlib.Path, entries: list, images: pathlib.Path, keep: bool) -> None:
    for entry in entries:
        omy_review.encode(omy_review.frames(session, entry), omy_review.review_video(session, entry))
    print(f"encoded {len(entries)} videos beside {images}", flush=True)
    if not keep:
        images.unlink()


if __name__ == "__main__":
    main()
