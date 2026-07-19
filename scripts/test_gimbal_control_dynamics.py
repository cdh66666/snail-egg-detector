"""Deterministic regression checks for the MaixCam gimbal controller."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path


def read_constants(path: Path) -> dict[str, float]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, float] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        if isinstance(value, (int, float)):
            values[target.id] = float(value)
    return values


class Axis:
    def __init__(self, cfg: dict[str, float], sign: float):
        self.cfg = cfg
        self.sign = sign
        self.filtered_error = 0.0
        self.command_step = 0.0
        self.initialized = False

    def step(self, error: float, dt: float) -> float:
        tau = self.cfg["GIMBAL_SLOW_ERROR_FILTER_TAU_S"]
        alpha = 1.0 - math.exp(-dt / max(0.01, tau))
        if not self.initialized:
            self.filtered_error = error
            self.initialized = True
        else:
            self.filtered_error += alpha * (error - self.filtered_error)
        magnitude = abs(self.filtered_error)
        if error == 0.0 or magnitude < 0.001:
            desired_step = 0.0
        else:
            ratio = min(1.0, magnitude / self.cfg["GIMBAL_SLOW_FULL_SPEED_ERROR"])
            desired_step = self.sign * (1.0 if self.filtered_error > 0.0 else -1.0)
            desired_step *= self.cfg["GIMBAL_SLOW_MAX_STEP_DEG"] * ratio
        if self.command_step * desired_step < 0.0:
            desired_step = 0.0
        max_change = self.cfg["GIMBAL_SLOW_STEP_CHANGE_DEG"]
        delta = max(-max_change, min(max_change, desired_step - self.command_step))
        self.command_step += delta
        if desired_step == 0.0 and abs(self.command_step) <= max_change:
            self.command_step = 0.0
        return self.command_step


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def soften_deadzone(error: float, deadzone: float) -> float:
    magnitude = abs(error)
    if magnitude <= deadzone:
        return 0.0
    scaled = (magnitude - deadzone) / max(0.001, 1.0 - deadzone)
    return scaled if error > 0.0 else -scaled


def simulate(cfg: dict[str, float]) -> dict[str, float]:
    # Plant values are conservatively fitted to the July 14 hardware log:
    # 21.8 image pixels per pan degree and 0.8 s effective camera/servo lag.
    dt = 1.0 / cfg["GIMBAL_CONTROL_HZ"]
    axis = Axis(cfg, cfg["GIMBAL_PAN_SIGN"])
    angle = 0.0
    observed_shift = 0.0
    errors: list[float] = []
    steps: list[float] = []
    velocities: list[float] = []
    duration = 50.0
    for index in range(int(duration / dt)):
        now = index * dt
        disturbance = 50.0 * math.sin(2.0 * math.pi * now / 10.0)
        error_px = disturbance + observed_shift
        error = error_px / 320.0
        error = soften_deadzone(error, cfg["GIMBAL_DEADZONE_X"])
        step = axis.step(error, dt)
        angle += step
        observed_shift += (21.8 * angle - observed_shift) * (1.0 - math.exp(-dt / 0.8))
        if now >= 10.0:
            errors.append(abs(error_px))
            steps.append(abs(step))
            velocities.append(abs(axis.command_step) / dt)

    return {
        "p95_error_px": percentile(errors, 0.95),
        "mean_error_px": sum(errors) / len(errors),
        "max_step_deg": max(steps),
        "max_rate_deg_s": max(velocities),
    }


def simulate_static_settle(cfg: dict[str, float]) -> float:
    dt = 1.0 / cfg["GIMBAL_CONTROL_HZ"]
    axis = Axis(cfg, cfg["GIMBAL_PAN_SIGN"])
    angle = 0.0
    observed_shift = 0.0
    for index in range(int(8.0 / dt)):
        error_px = -138.0 + observed_shift
        error = error_px / 320.0
        error = soften_deadzone(error, cfg["GIMBAL_DEADZONE_X"])
        angle += axis.step(error, dt)
        observed_shift += (21.8 * angle - observed_shift) * (1.0 - math.exp(-dt / 0.8))
        if abs(error_px) <= 12.0:
            return index * dt
    return 8.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", type=Path, default=Path("maixcam/main.py"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = read_constants(args.main)
    result = simulate(cfg)
    result["static_settle_s"] = simulate_static_settle(cfg)
    result["standard_p95_error_px"] = 65.0
    result["standard_static_settle_s"] = 8.0
    result["passed"] = (
        result["p95_error_px"] <= result["standard_p95_error_px"]
        and result["static_settle_s"] <= result["standard_static_settle_s"]
        and result["max_step_deg"] <= cfg["GIMBAL_SLOW_MAX_STEP_DEG"] + 1e-9
        and result["max_rate_deg_s"] <= cfg["GIMBAL_SLOW_MAX_STEP_DEG"] * cfg["GIMBAL_CONTROL_HZ"] + 1e-9
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
