"""Regression checks for the 50 Hz minimum-jerk servo trajectory."""

import ast
import json
from pathlib import Path


MAIN = Path("maixcam/main.py")


def load_trajectory():
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    namespace = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.startswith("GIMBAL_TRAJECTORY_")
        ):
            exec(compile(ast.Module([node], type_ignores=[]), str(MAIN), "exec"), namespace)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("quintic_trajectory_"):
            exec(compile(ast.Module([node], type_ignores=[]), str(MAIN), "exec"), namespace)
    return namespace


def main():
    ns = load_trajectory()
    duration_for = ns["quintic_trajectory_duration"]
    coefficients_for = ns["quintic_trajectory_coefficients"]
    sample = ns["quintic_trajectory_sample"]
    dt = 1.0 / ns["GIMBAL_TRAJECTORY_HZ"]
    cases = []
    passed = True
    for target in (1.0, 5.0, 12.0, -12.0, 30.0):
        duration = duration_for(target)
        coefficients = coefficients_for(0.0, 0.0, 0.0, target, duration)
        rows = [
            sample(coefficients, min(index * dt, duration), duration, target)
            for index in range(int(duration / dt) + 2)
        ]
        positions = [row[0] for row in rows]
        velocities = [row[1] for row in rows]
        accelerations = [row[2] for row in rows]
        overshoot = max(0.0, max(position - target for position in positions)) if target > 0 else max(
            0.0, max(target - position for position in positions)
        )
        max_jerk = max(
            abs(accelerations[index] - accelerations[index - 1]) / dt
            for index in range(1, len(accelerations))
        )
        case = {
            "target_deg": target,
            "duration_s": duration,
            "max_speed_deg_s": max(map(abs, velocities)),
            "max_accel_deg_s2": max(map(abs, accelerations)),
            "max_jerk_deg_s3": max_jerk,
            "overshoot_deg": overshoot,
            "end": rows[-1],
        }
        cases.append(case)
        passed = passed and (
            case["max_speed_deg_s"] <= ns["GIMBAL_TRAJECTORY_MAX_SPEED_DEG_S"] + 0.01
            and case["max_accel_deg_s2"] <= ns["GIMBAL_TRAJECTORY_MAX_ACCEL_DEG_S2"] + 0.05
            and case["max_jerk_deg_s3"] <= ns["GIMBAL_TRAJECTORY_MAX_JERK_DEG_S3"] + 1.0
            and overshoot <= 1e-8
            and rows[-1] == (target, 0.0, 0.0)
        )
    pan_target = 8.0
    tilt_target = 4.0
    sync_duration = max(duration_for(pan_target), duration_for(tilt_target))
    pan_coefficients = coefficients_for(0.0, 0.0, 0.0, pan_target, sync_duration)
    tilt_coefficients = coefficients_for(0.0, 0.0, 0.0, tilt_target, sync_duration)
    sync_error = 0.0
    for index in range(int(sync_duration / dt) + 2):
        elapsed = min(index * dt, sync_duration)
        pan_position = sample(pan_coefficients, elapsed, sync_duration, pan_target)[0]
        tilt_position = sample(tilt_coefficients, elapsed, sync_duration, tilt_target)[0]
        sync_error = max(
            sync_error,
            abs(pan_position / pan_target - tilt_position / tilt_target),
        )
    passed = passed and sync_error <= 1e-9
    print(json.dumps({
        "passed": passed,
        "cases": cases,
        "synchronized_line": {
            "pan_target_deg": pan_target,
            "tilt_target_deg": tilt_target,
            "max_normalized_progress_error": sync_error,
        },
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
