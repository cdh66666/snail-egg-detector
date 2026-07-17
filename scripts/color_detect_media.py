"""Broad pink-color baseline for comparison, not a snail-egg classifier.

Usage:
  python scripts/color_detect_media.py input.jpg --output runs/color.jpg
  python scripts/color_detect_media.py input.mp4 --output runs/color.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def boxes(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Broad pink/red range. Deliberately permissive for the backup experiment.
    mask = cv2.inRange(hsv, np.array([160, 35, 35]), np.array([179, 255, 255]))
    mask |= cv2.inRange(hsv, np.array([0, 35, 35]), np.array([12, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    result = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h >= 24 and w >= 4 and h >= 4:
            result.append((x, y, w, h))
    return sorted(result, key=lambda b: (b[1], b[0]))


def draw(frame: np.ndarray) -> np.ndarray:
    found = boxes(frame)
    for index, (x, y, w, h) in enumerate(found, 1):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, f"PINK {index}", (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"COLOR CANDIDATES {len(found)}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-stride", type=int, default=1)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(args.input))
    if image is not None:
        cv2.imwrite(str(args.output), draw(image))
        print(f"saved {args.output} candidates={len(boxes(image))}")
        return
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % max(1, args.video_stride) == 0:
            writer.write(draw(frame))
        index += 1
    cap.release()
    writer.release()
    print(f"saved {args.output} frames={index}")


if __name__ == "__main__":
    main()
