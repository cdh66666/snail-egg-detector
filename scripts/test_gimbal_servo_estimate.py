"""Host-side check for the open-loop servo position estimator."""

from __future__ import annotations

import ast
import math
from pathlib import Path


def constants():
    tree = ast.parse((Path(__file__).resolve().parents[1] / "maixcam" / "main.py").read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def main():
    cfg = constants()
    tau = float(cfg["GIMBAL_SERVO_ESTIMATE_TAU_S"])
    max_speed = float(cfg["GIMBAL_SERVO_ESTIMATE_MAX_SPEED_DEG_S"])
    dt = 1.0 / float(cfg["GIMBAL_TRAJECTORY_HZ"])
    estimate = 0.0
    values = []
    for _ in range(50):
        alpha = 1.0 - math.exp(-dt / tau)
        estimate += max(-max_speed * dt, min(max_speed * dt, (10.0 - estimate) * alpha))
        values.append(estimate)
    assert all(a <= b for a, b in zip(values, values[1:]))
    assert all(0.0 <= value <= 10.0 for value in values)
    previous = estimate
    for _ in range(10):
        alpha = 1.0 - math.exp(-dt / tau)
        estimate += max(-max_speed * dt, min(max_speed * dt, (-5.0 - estimate) * alpha))
    assert estimate < previous
    assert abs(estimate - previous) <= max_speed * dt * 10.0 + 1e-6
    print({"bounded_lag_estimate": "passed", "tau_s": tau, "max_speed_deg_s": max_speed})


if __name__ == "__main__":
    main()
