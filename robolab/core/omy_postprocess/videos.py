"""CPU-only verification clips from completed, native-15 policy camera caches."""

import json
import math
import re
from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np

from .cameras import FRAME_MAPPING_GROUP
from .io import publish_directory, sha256_file, write_json


def _probe(path, count):
    result = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,r_frame_rate,nb_read_frames,width,height,duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True)
    stream, = json.loads(result.stdout)["streams"]
    if (stream["codec_name"] != "h264" or stream["r_frame_rate"] != "15/1"
            or int(stream["nb_read_frames"]) != count
            or not math.isclose(float(stream["duration"]), count / 15, abs_tol=0.01)):
        raise ValueError(f"Video failed frame count/timing verification: {path}")
    return stream


def export_success_videos(dataset, output=None):
    from PIL import Image, ImageDraw, ImageFont

    dataset = Path(dataset).resolve()
    if dataset.is_dir():
        dataset = dataset / "dataset.json"
    root = dataset.parent
    contents = json.loads(dataset.read_text())
    selected = [e for e in contents["episodes"] if e["outcome"] == "success"]
    if not selected:
        print("No completed successful episodes for verification videos.", flush=True)
        return None
    output = Path(output).absolute() if output else root / "verification_videos"
    source_sha = sha256_file(dataset)
    if output.exists():
        index = json.loads((output / "index.json").read_text())
        if index.get("source_dataset_sha256") != source_sha:
            raise FileExistsError("Verification videos belong to another dataset version; choose a new output")
        for filename, expected in index["artifact_sha256"].items():
            if Path(filename).name != filename or sha256_file(output / filename) != expected:
                raise ValueError(f"Verification video artifact changed: {filename}")
        if index.get("video_retention") == "combined_only":
            raise FileExistsError("Existing preview contains only the combined video; choose a new output "
                                  "to regenerate separate clips from the camera cache")
        print(f"Verification videos: {output}", flush=True)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 15) if font_path.exists() else ImageFont.load_default()
    cameras = ("over_shoulder_left_camera", "wrist_cam")
    clips, finals = [], []
    for number, episode in enumerate(selected):
        name = f"success_{number + 1:03d}_" + re.sub(r"[^A-Za-z0-9_-]", "_", str(episode["attempt_id"]))[:120]
        if Path(name).name != name:
            raise ValueError("Invalid attempt identity for video output")
        video = temporary / (name + ".mp4")
        with h5py.File(root / episode["images"], "r") as file:
            diagnostic = bool(file["data"].attrs.get("diagnostic_only", False))
            group = file[episode["source_group"]]
            channels = [group[f"obs/image_obs/{camera}"] for camera in cameras]
            count, height, width, rgb = channels[0].shape
            timestamps = group[f"{FRAME_MAPPING_GROUP}/timestamp_s"][:]
            if (count != episode["sample_count"] or channels[1].shape != channels[0].shape or rgb != 3
                    or any(c.dtype != np.dtype("uint8") for c in channels)
                    or not np.allclose(timestamps, np.arange(count) / 15, rtol=0, atol=1e-8)):
                raise ValueError("Verification inputs do not match native-15 camera rows")
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s:v", f"{2 * width}x{height + 72}", "-framerate", "15", "-i", "pipe:0",
                "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-threads", "2",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
            samples = {}
            with (temporary / (name + ".log")).open("wb") as log:
                encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
                try:
                    for frame in range(count):
                        panel = np.empty((height + 72, width * 2, 3), dtype=np.uint8)
                        panel[72:, :width] = channels[0][frame]
                        panel[72:, width:] = channels[1][frame]
                        banner = Image.new("RGB", (width * 2, 72), (22, 26, 31))
                        draw = ImageDraw.Draw(banner)
                        label = "DIAGNOSTIC FIXTURE: assigned SUCCESS" if diagnostic else "recorded SUCCESS"
                        draw.text((10, 4), f"{episode['attempt_id']} | {label}", font=font, fill="white")
                        draw.text((10, 39), f"Shoulder / wrist | frame {frame + 1}/{count} | t={timestamps[frame]:.2f}s | 15 FPS",
                                  font=font, fill="white")
                        panel[:72] = np.asarray(banner)
                        encoder.stdin.write(panel.tobytes())
                        if frame in (0, count // 2, count - 1):
                            samples[frame] = Image.fromarray(panel[72:]).copy()
                    encoder.stdin.close()
                    if encoder.wait() != 0:
                        raise RuntimeError(f"Verification encoder failed: {log.name}")
                finally:
                    if encoder.poll() is None:
                        encoder.kill()
                        encoder.wait()
            verified = _probe(video, count)
            sheet = Image.new("RGB", (960, 410), (22, 26, 31))
            draw = ImageDraw.Draw(sheet)
            for column, (frame, image) in enumerate(sorted(samples.items())):
                draw.text((column * 320 + 5, 5), f"Frame {frame + 1}/{count}; t={timestamps[frame]:.2f}s", font=small, fill="white")
                for row in range(2):
                    camera = image.crop((row * width, 0, (row + 1) * width, height))
                    sheet.paste(camera.resize((320, 180)), (column * 320, 45 + row * 180))
            preview = name + "_frames.jpg"
            sheet.save(temporary / preview, quality=94)
            finals.append(samples[count - 1].resize((640, 180)))
        clips.append({"video": video.name, "frames": preview, "attempt_id": episode["attempt_id"],
                      "diagnostic_only": diagnostic,
                      "frame_count": count, "duration_s": count / 15, "verified_stream": verified})
        print(f"Verification clip: {video.name} ({count} frames)", flush=True)
    (temporary / "concat.txt").write_text("".join(f"file '{c['video']}'\n" for c in clips))
    reel = temporary / "all_successes.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-f", "concat", "-safe", "1",
                    "-i", str(temporary / "concat.txt"), "-c", "copy", "-movflags", "+faststart", str(reel)], check=True)
    reel_stream = _probe(reel, sum(c["frame_count"] for c in clips))
    overview = Image.new("RGB", (1920, 215 * math.ceil(len(finals) / 3)), (22, 26, 31))
    draw = ImageDraw.Draw(overview)
    for i, image in enumerate(finals):
        x, y = i % 3 * 640, i // 3 * 215
        draw.text((x + 5, y + 3), f"Success {i + 1}: final recorded frame", font=small, fill="white")
        overview.paste(image, (x, y + 30))
    overview.save(temporary / "final_frames.jpg", quality=94)
    (temporary / "README.md").write_text("# Verification videos\n\nOpen all_successes.mp4 or an individual success clip. "
        "Left: shoulder camera; right: wrist camera. One frame per recorded step at 15 FPS. "
        "Success labels come from collection metadata. Clips end at the final recorded frame. "
        "Individual videos are retained alongside the combined video. "
        "Episode frame ranges and start times in the combined video are recorded in index.json.\n")
    if sha256_file(dataset) != source_sha:
        raise RuntimeError("Dataset changed during verification encoding")
    start_frame = 0
    for clip in clips:
        clip.update(combined_video=reel.name, start_frame=start_frame, start_time_s=start_frame / 15,
                    end_frame_exclusive=start_frame + clip["frame_count"])
        start_frame += clip["frame_count"]
    (temporary / "concat.txt").unlink()
    write_json(temporary / "index.json", {"schema_version": 3, "source_dataset_sha256": source_sha,
        "video_retention": "combined_and_individual",
        "fps": 15, "clips": clips, "combined_stream": reel_stream,
        "artifact_sha256": {f.name: sha256_file(f) for f in temporary.iterdir() if f.is_file()}})
    publish_directory(temporary, output)
    print(f"Verification videos: {output}", flush=True)
    return output
