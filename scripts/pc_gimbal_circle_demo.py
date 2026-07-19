"""Render the exact current 50 Hz software trajectory following a circle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from test_gimbal_circle_tracking import simulate_circle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/gimbal_circle_50hz_current_controller.mp4"))
    parser.add_argument("--report", type=Path, default=Path("outputs/gimbal_circle_50hz_current_controller.json"))
    return parser.parse_args()


def point(center, scale, pan, tilt):
    return int(center[0] + pan * scale), int(center[1] + tilt * scale)


def main():
    args = parse_args()
    rows, metrics = simulate_circle()
    width, height = 1280, 720
    center = (480, 360)
    scale = 48.0
    fps = 50.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    trail = []

    for row in rows:
        frame = np.full((height, width, 3), (20, 24, 26), dtype=np.uint8)
        cv2.putText(frame, "CURRENT MAIXCAM GIMBAL SOFTWARE - 50 Hz CIRCLE TEST", (42, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.88, (240, 245, 246), 2)
        cv2.putText(frame, "OFFLINE SOFTWARE OUTPUT - NOT A HARDWARE RECORDING", (42, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 180, 255), 1)

        # Commanded 5 x 5 degree circle in gimbal angle space.
        cv2.circle(frame, center, int(5 * scale), (85, 94, 98), 2, cv2.LINE_AA)
        for degrees in range(0, 360, 30):
            import math
            marker = point(center, scale, 5 * math.cos(math.radians(degrees)), 5 * math.sin(math.radians(degrees)))
            cv2.circle(frame, marker, 3, (75, 82, 86), -1)

        target = point(center, scale, row["target_pan"], row["target_tilt"])
        applied = point(center, scale, row["pan"], row["tilt"])
        trail.append(applied)
        trail = trail[-400:]
        if len(trail) > 1:
            cv2.polylines(frame, [np.asarray(trail, dtype=np.int32)], False, (230, 180, 60), 3, cv2.LINE_AA)
        cv2.circle(frame, target, 10, (35, 220, 90), -1, cv2.LINE_AA)
        cv2.drawMarker(frame, applied, (255, 255, 255), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
        cv2.line(frame, target, applied, (90, 100, 105), 1, cv2.LINE_AA)

        panel_x = 850
        cv2.putText(frame, "GREEN  target angle", (panel_x, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (35, 220, 90), 2)
        cv2.putText(frame, "WHITE  50 Hz servo command", (panel_x, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2)
        cv2.putText(frame, "CYAN   applied trajectory", (panel_x, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 180, 60), 2)
        values = [
            f"time        {row['time']:5.2f} s",
            f"pan         {row['pan']:+5.2f} deg",
            f"tilt        {row['tilt']:+5.2f} deg",
            f"speed       {row['speed']:5.2f} deg/s",
            f"accel       {row['acceleration']:5.2f} deg/s2",
            f"path error  {row['error']:5.2f} deg",
        ]
        for index, value in enumerate(values):
            cv2.putText(frame, value, (panel_x, 300 + index * 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (220, 226, 228), 1, cv2.LINE_AA)
        cv2.putText(frame, "Software limits: 3 deg/s, 8 deg/s2, 40 deg/s3", (42, 690),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 170, 175), 1)
        writer.write(frame)

    writer.release()
    args.report.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
