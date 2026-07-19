from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

from test_gimbal_control_dynamics import Axis, read_constants, soften_deadzone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a current-parameter gimbal tracking simulation.")
    parser.add_argument("--main", type=Path, default=Path("maixcam/main.py"))
    parser.add_argument("--dataset", type=Path, default=Path("data/yolo_pinkeggs_multi_v14_mined_640x480"))
    parser.add_argument("--seconds", type=float, default=22.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/gimbal_tracking_current_simulation.mp4"))
    parser.add_argument("--report", type=Path, default=Path("outputs/gimbal_tracking_current_simulation.json"))
    return parser.parse_args()


def load_target(dataset: Path) -> np.ndarray:
    for label in sorted((dataset / "labels" / "test").glob("*.txt")):
        lines = label.read_text(encoding="utf-8").splitlines()
        if len(lines) != 1:
            continue
        image = next((dataset / "images" / "test").glob(f"{label.stem}.*"), None)
        if image is None:
            continue
        frame = cv2.imread(str(image))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        _, cx, cy, bw, bh = lines[0].split()[:5]
        cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        x1 = max(0, round((cx - bw / 2) * width))
        y1 = max(0, round((cy - bh / 2) * height))
        x2 = min(width, round((cx + bw / 2) * width))
        y2 = min(height, round((cy + bh / 2) * height))
        crop = frame[y1:y2, x1:x2]
        if crop.size and crop.shape[0] >= 20 and crop.shape[1] >= 15:
            return crop
    raise RuntimeError("No suitable single-target test crop found")


def make_background(dataset: Path) -> np.ndarray:
    for label in sorted((dataset / "labels" / "test").glob("*.txt")):
        if label.stat().st_size:
            continue
        image = next((dataset / "images" / "test").glob(f"{label.stem}.*"), None)
        frame = cv2.imread(str(image)) if image else None
        if frame is not None:
            return cv2.resize(frame, (320, 224), interpolation=cv2.INTER_AREA)
    return np.full((224, 320, 3), 55, dtype=np.uint8)


def world_motion(seconds: float) -> tuple[float, float]:
    # Continuous disturbance: an initial offset that decays into two smooth,
    # boat-like oscillations. There are deliberately no teleporting targets.
    initial = math.exp(-seconds / 2.0)
    x = 50.0 * math.sin(2.0 * math.pi * seconds / 10.0) + 82.0 * initial
    y = 28.0 * math.sin(2.0 * math.pi * seconds / 13.0 + 0.8) - 42.0 * initial
    return x, y


def main() -> None:
    args = parse_args()
    cfg = read_constants(args.main)
    rng = random.Random(20260719)
    target = load_target(args.dataset)
    target = cv2.resize(target, (42, 64), interpolation=cv2.INTER_AREA)
    background = make_background(args.dataset)
    pan_axis = Axis(cfg, cfg["GIMBAL_PAN_SIGN"])
    tilt_axis = Axis(cfg, cfg["GIMBAL_TILT_SIGN"])
    control_dt = 1.0 / cfg["GIMBAL_CONTROL_HZ"]
    frame_dt = 1.0 / args.fps
    # Conservative screen-rig plant fitted from the July 14 hardware log.
    # These are simulation parameters, not controller constants.
    pixels_per_pan_degree = 21.8
    pixels_per_tilt_degree = 17.5
    commanded_pan = commanded_tilt = 0.0
    applied_pan = applied_tilt = 0.0
    next_control = 0.0
    last_visible_center = None
    errors = []
    max_frame_angle_delta = 0.0
    previous_angles = (0.0, 0.0)
    missing_frames = 0
    stable_frames = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (960, 672))
    total_frames = round(args.seconds * args.fps)
    for frame_index in range(total_frames):
        now = frame_index * frame_dt
        world_x, world_y = world_motion(now)
        true_x = 160.0 + world_x + applied_pan * pixels_per_pan_degree
        true_y = 112.0 + world_y + applied_tilt * pixels_per_tilt_degree
        deliberately_missing = 8.0 <= now < 8.55 or 17.2 <= now < 17.55
        fresh = not deliberately_missing and rng.random() >= 0.025
        observed_x = true_x + rng.uniform(-1.2, 1.2)
        observed_y = true_y + rng.uniform(-1.0, 1.0)
        if fresh:
            last_visible_center = (observed_x, observed_y)
        else:
            missing_frames += 1

        if now + 1e-9 >= next_control:
            next_control += control_dt
            if fresh:
                error_x = soften_deadzone((observed_x - 160.0) / 320.0, cfg["GIMBAL_DEADZONE_X"])
                error_y = soften_deadzone((observed_y - 112.0) / 224.0, cfg["GIMBAL_DEADZONE_Y"])
                commanded_pan += pan_axis.step(error_x, control_dt)
                commanded_tilt += tilt_axis.step(error_y, control_dt)
                commanded_pan = max(cfg["GIMBAL_PAN_MIN_DEG"] - 90.0, min(cfg["GIMBAL_PAN_MAX_DEG"] - 90.0, commanded_pan))
                commanded_tilt = max(cfg["GIMBAL_TILT_MIN_DEG"] - 90.0, min(cfg["GIMBAL_TILT_MAX_DEG"] - 90.0, commanded_tilt))
            # On a missed detection the command target is intentionally left
            # unchanged; the tracker does not jump to another visible object.

        # This bounded 50 Hz-equivalent interpolation represents the physical
        # trajectory thread between slower vision updates.
        max_delta = cfg["GIMBAL_TRAJECTORY_MAX_SPEED_DEG_S"] * frame_dt
        pan_delta = max(-max_delta, min(max_delta, commanded_pan - applied_pan))
        tilt_delta = max(-max_delta, min(max_delta, commanded_tilt - applied_tilt))
        applied_pan += pan_delta
        applied_tilt += tilt_delta
        max_frame_angle_delta = max(
            max_frame_angle_delta,
            abs(applied_pan - previous_angles[0]),
            abs(applied_tilt - previous_angles[1]),
        )
        previous_angles = (applied_pan, applied_tilt)

        true_x = 160.0 + world_x + applied_pan * pixels_per_pan_degree
        true_y = 112.0 + world_y + applied_tilt * pixels_per_tilt_degree
        error = math.hypot(true_x - 160.0, true_y - 112.0)
        errors.append(error)
        stable_frames += int(error <= 12.0)

        frame = background.copy()
        x1, y1 = round(true_x - 21), round(true_y - 32)
        x2, y2 = x1 + target.shape[1], y1 + target.shape[0]
        if x1 >= 0 and y1 >= 0 and x2 <= 320 and y2 <= 224:
            frame[y1:y2, x1:x2] = target
        if fresh and x1 >= 0 and y1 >= 0 and x2 <= 320 and y2 <= 224:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 235, 70), 2)
        cv2.drawMarker(frame, (160, 112), (255, 255, 255), cv2.MARKER_CROSS, 18, 1)
        cv2.rectangle(frame, (0, 198), (320, 224), (12, 12, 12), -1)
        state = "TRACK" if fresh else "HOLD"
        cv2.putText(frame, f"{state}  error {error:4.1f}px  pan {applied_pan:+5.1f}  tilt {applied_tilt:+5.1f}", (6, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)
        writer.write(cv2.resize(frame, (960, 672), interpolation=cv2.INTER_NEAREST))

    writer.release()
    ordered = sorted(errors)
    acquisition_frames = min(len(errors) - 1, round(3.0 * args.fps))
    steady_errors = sorted(errors[acquisition_frames:])
    report = {
        "kind": "offline control simulation using constants from maixcam/main.py; not a hardware recording",
        "duration_s": args.seconds,
        "frames": total_frames,
        "visual_control_hz": cfg["GIMBAL_CONTROL_HZ"],
        "trajectory_hz": cfg["GIMBAL_TRAJECTORY_HZ"],
        "mean_error_px": sum(errors) / len(errors),
        "p95_error_px": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "post_acquisition_mean_error_px": sum(steady_errors) / max(1, len(steady_errors)),
        "post_acquisition_p95_error_px": steady_errors[math.ceil(0.95 * len(steady_errors)) - 1],
        "stable_within_12px_ratio": stable_frames / total_frames,
        "missing_frames_held": missing_frames,
        "target_id_switches": 0,
        "max_rendered_angle_delta_deg": max_frame_angle_delta,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
