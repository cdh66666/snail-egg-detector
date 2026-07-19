"""Continuous circular-command regression for the deployed gimbal trajectory."""

from __future__ import annotations

import json
import math
from pathlib import Path

from test_gimbal_control_dynamics import read_constants
from test_gimbal_trajectory import load_trajectory


MAIN = Path("maixcam/main.py")


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def simulate_circle(seconds=32.0, period=16.0, pan_radius=5.0, tilt_radius=5.0):
    cfg = read_constants(MAIN)
    trajectory = load_trajectory()
    duration_for = trajectory["quintic_trajectory_duration"]
    coefficients_for = trajectory["quintic_trajectory_coefficients"]
    sample = trajectory["quintic_trajectory_sample"]
    dt = 1.0 / cfg["GIMBAL_TRAJECTORY_HZ"]

    applied = [0.0, 0.0]
    velocity = [0.0, 0.0]
    acceleration = [0.0, 0.0]
    trajectory_target = [0.0, 0.0]
    coefficients = None
    segment_start = 0.0
    segment_duration = 0.0
    rows = []
    replans = 0

    for index in range(round(seconds / dt)):
        now = index * dt
        phase = 2.0 * math.pi * now / period
        pending = [pan_radius * math.cos(phase), tilt_radius * math.sin(phase)]
        target_changed = max(
            abs(pending[axis] - trajectory_target[axis]) for axis in range(2)
        ) >= cfg["GIMBAL_TARGET_REPLAN_EPS_DEG"]
        if coefficients is None:
            target_changed = max(
                abs(pending[axis] - applied[axis]) for axis in range(2)
            ) > cfg["GIMBAL_TRAJECTORY_POSITION_EPS_DEG"]
        if target_changed:
            durations = [
                duration_for(
                    pending[axis] - applied[axis],
                    velocity[axis],
                    acceleration[axis],
                )
                for axis in range(2)
            ]
            segment_duration = max(durations)
            segment_start = now
            trajectory_target = pending[:]
            coefficients = [
                coefficients_for(
                    applied[axis], velocity[axis], acceleration[axis],
                    pending[axis], segment_duration,
                )
                for axis in range(2)
            ]
            replans += 1

        old_acceleration = acceleration[:]
        if coefficients is not None:
            elapsed = now - segment_start
            for axis in range(2):
                applied[axis], velocity[axis], acceleration[axis] = sample(
                    coefficients[axis], elapsed, segment_duration,
                    trajectory_target[axis],
                )
        error = math.hypot(pending[0] - applied[0], pending[1] - applied[1])
        speed = math.hypot(velocity[0], velocity[1])
        accel = math.hypot(acceleration[0], acceleration[1])
        jerk = math.hypot(
            (acceleration[0] - old_acceleration[0]) / dt,
            (acceleration[1] - old_acceleration[1]) / dt,
        )
        rows.append({
            "time": now,
            "target_pan": pending[0],
            "target_tilt": pending[1],
            "pan": applied[0],
            "tilt": applied[1],
            "speed": speed,
            "acceleration": accel,
            "jerk": jerk,
            "error": error,
        })

    steady = [row for row in rows if row["time"] >= period]
    metrics = {
        "kind": "exact 50 Hz software trajectory simulation; not a hardware recording",
        "trajectory_hz": cfg["GIMBAL_TRAJECTORY_HZ"],
        "period_s": period,
        "pan_radius_deg": pan_radius,
        "tilt_radius_deg": tilt_radius,
        "replans_per_s": replans / seconds,
        "steady_mean_error_deg": sum(row["error"] for row in steady) / len(steady),
        "steady_p95_error_deg": percentile([row["error"] for row in steady], 0.95),
        "max_speed_deg_s": max(row["speed"] for row in steady),
        "max_acceleration_deg_s2": max(row["acceleration"] for row in steady),
        "max_jerk_deg_s3": max(row["jerk"] for row in steady),
    }
    return rows, metrics


def main():
    _rows, metrics = simulate_circle()
    assert metrics["steady_mean_error_deg"] <= 2.25
    assert metrics["steady_p95_error_deg"] <= 3.0
    assert metrics["max_speed_deg_s"] <= 3.01
    assert metrics["max_acceleration_deg_s2"] <= 8.05
    assert metrics["max_jerk_deg_s3"] <= 40.5
    print(json.dumps({"passed": True, **metrics}, indent=2))


if __name__ == "__main__":
    main()
