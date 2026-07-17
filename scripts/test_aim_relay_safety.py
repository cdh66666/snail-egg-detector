"""Regression test for the fail-closed red aiming-light safety gate."""

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
                "AIM_RELAY_OFF_MISSING_FRAMES",
                "AIM_RELAY_NO_TARGET_TEST_FLAG",
            }:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "aim_relay_decision":
            selected.append(node)
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(MAIN), "exec"), namespace)
    return namespace


def main():
    ns = load_gate()
    decide = ns["aim_relay_decision"]
    assert ns["AIM_RELAY_OFF_MISSING_FRAMES"] == 1
    assert ns["AIM_RELAY_NO_TARGET_TEST_FLAG"].endswith("test_no_target_safety")

    # Startup and unstable whole-view detections cannot enable the light.
    assert decide(False, 0, 1) is False
    assert decide(True, 1, 0) is None
    assert decide(True, 2, 0) is None
    assert decide(True, 3, 0) is True

    # The relay decision is independent of which target is selected as primary:
    # any fresh valid egg in the view keeps the light eligible to stay on.
    assert decide(True, 3, 0) is True

    # One missing frame, including a Kalman-only prediction, fails closed.
    assert decide(False, 0, 1) is False
    assert decide(False, 0, 50) is False
    print({"stable_enable_frames": 3, "missing_frames_to_off": 1, "fail_closed": "passed"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
