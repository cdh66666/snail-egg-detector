"""Record annotated MaixCAM snapshots and live status into an MP4."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np


def fetch(url: str, timeout: float = 2.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.10.108")
    parser.add_argument("--token", default="maixcam")
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = f"http://{args.host}:8000"
    snapshot_url = f"{base}/snapshot.jpg?token={args.token}"
    status_url = f"{base}/api/status?token={args.token}"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    statuses: list[dict] = []
    started = time.monotonic()
    next_frame = started
    failures = 0
    while time.monotonic() - started < args.seconds:
        try:
            status = json.loads(fetch(status_url).decode("utf-8"))
            encoded = np.frombuffer(fetch(snapshot_url), dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("invalid JPEG frame")
            elapsed = time.monotonic() - started
            label_1 = (
                f"REAL DEVICE  t={elapsed:05.1f}s  loop={status.get('fps', 0):.1f} FPS  "
                f"eggs={status.get('eggs', 0)}  lock={status.get('primary', 0)}"
            )
            label_2 = f"dot={status.get('dot', '?')}  relay={status.get('relay', '?')}"
            cv2.rectangle(frame, (0, frame.shape[0] - 38), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, label_1, (4, frame.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.31,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, label_2, (4, frame.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.31,
                        (255, 255, 255), 1, cv2.LINE_AA)
            frames.append(frame)
            statuses.append(status)
        except Exception as exc:
            failures += 1
            print(f"capture failure {failures}: {exc}")

        next_frame += 1.0 / args.fps
        delay = next_frame - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    if not frames:
        raise RuntimeError("no MaixCAM frames captured")
    capture_elapsed = time.monotonic() - started
    output_fps = len(frames) / max(0.001, capture_elapsed)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("unable to create MP4 writer")
    for frame in frames:
        writer.write(frame)
    writer.release()

    summary = {
        "frames": len(frames),
        "duration_s": round(capture_elapsed, 2),
        "recording_fps": round(output_fps, 2),
        "capture_failures": failures,
        "device_fps_mean": round(sum(float(s.get("fps", 0)) for s in statuses) / len(statuses), 2),
        "eggs_min": min(int(s.get("eggs", 0)) for s in statuses),
        "eggs_max": max(int(s.get("eggs", 0)) for s in statuses),
        "dot_fresh_ratio": round(sum(s.get("dot") == "fresh" for s in statuses) / len(statuses), 3),
        "relay_on_ratio": round(sum(s.get("relay") == "AIM ON" for s in statuses) / len(statuses), 3),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
