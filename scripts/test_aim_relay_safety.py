"""Regression tests for automatic gating, manual override, and emergency stop."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "maixcam" / "main.py"


def load_gate():
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & {
                "AIM_RELAY_ON_STABLE_FRAMES",
                "AIM_RELAY_OFF_MISSING_S",
                "AIM_RELAY_NO_TARGET_TEST_FLAG",
            }:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {"aim_relay_decision", "web_aim_relay_decision"}:
            selected.append(node)
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(MAIN), "exec"), namespace)
    return namespace


def main():
    ns = load_gate()
    decide = ns["aim_relay_decision"]
    web_decide = ns["web_aim_relay_decision"]
    assert ns["AIM_RELAY_OFF_MISSING_S"] == 2.0
    assert ns["AIM_RELAY_NO_TARGET_TEST_FLAG"].endswith("test_no_target_safety")

    # Startup and unstable whole-view detections cannot enable the light.
    assert decide(False, 0, 1, 0.5) is None
    assert decide(True, 1, 0) is None
    assert decide(True, 2, 0) is None
    assert decide(True, 3, 0) is True

    # The relay decision is independent of which target is selected as primary:
    # any fresh valid egg in the view keeps the light eligible to stay on.
    assert decide(True, 3, 0) is True

    # Brief gaps, including Kalman-only predictions, are tolerated; two
    # seconds without a fresh whole-view detection fails closed.
    assert decide(False, 0, 1, 1.99) is None
    assert decide(False, 0, 50, 2.0) is False

    # Manual ON is an explicit calibration override for the low-power aiming
    # light. It does not require a target; emergency stop always wins.
    assert web_decide(False, True, True, False, 0, 50, 999.0) is True
    assert web_decide(True, True, True, True, 3, 0, 0.0) is False
    assert web_decide(False, False, True, True, 3, 0, 0.0) is False
    assert web_decide(False, None, True, False, 0, 50, 2.0) is False
    assert web_decide(False, None, False, False, 0, 0, 0.0) is None
    print({"stable_enable_frames": 3, "missing_seconds_to_off": 2.0, "manual_override": "passed", "emergency_priority": "passed"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
