"""Regression checks for the phone-selected target confidence gate."""

import ast
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "maixcam" / "main.py"


def load_gate():
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    selected = []
    names = {
        "SELECTED_LOCK_CONFIRM_DETECTIONS",
        "SELECTED_LOCK_TIMEOUT_DETECTIONS",
        "SELECTED_LOCK_MAX_CENTER_SHIFT_BOXES",
        "SELECTED_LOCK_MAX_SIZE_RATIO",
    }
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if targets & names:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "obj_center":
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "SelectedLockGate":
            selected.append(node)
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(MAIN), "exec"), namespace)
    return namespace


def target(cx, cy, width=30, height=40, predicted=False):
    return SimpleNamespace(
        x=cx - width * 0.5,
        y=cy - height * 0.5,
        w=width,
        h=height,
        predicted=predicted,
    )


def main():
    ns = load_gate()
    Gate = ns["SelectedLockGate"]

    stable = Gate()
    stable.reset(target(100, 100))
    assert stable.state() == "confirming"
    for observation in (target(102, 99), target(101, 101), target(103, 100)):
        stable.observe(observation)
    assert stable.state() == "stable"
    assert stable.confirmed and not stable.failed
    # A stable lock remains stable through later missing observations; the
    # separate selected-loss policy decides when to stop and hand off.
    for _ in range(20):
        assert stable.observe(None) == "stable"

    missing = Gate()
    missing.reset(target(200, 120))
    for _ in range(ns["SELECTED_LOCK_TIMEOUT_DETECTIONS"]):
        missing.observe(None)
    assert missing.state() == "approach_hold"
    assert missing.failed and not missing.confirmed

    jitter = Gate()
    jitter.reset(target(80, 80))
    for index in range(ns["SELECTED_LOCK_TIMEOUT_DETECTIONS"]):
        jitter.observe(target(150 + index * 10, 150, width=12, height=70))
    assert jitter.state() == "approach_hold"

    source = MAIN.read_text(encoding="utf-8")
    assert "web_control.confirm_selection(selected_track_id)" in source
    assert "control_target = primary_obj if" in source
    assert "STOP_CENTER" not in source
    print({
        "stable_confirmations": ns["SELECTED_LOCK_CONFIRM_DETECTIONS"],
        "unstable_timeout": ns["SELECTED_LOCK_TIMEOUT_DETECTIONS"],
        "stable_lock": "passed",
        "conservative_handoff": "passed",
        "shutdown_no_sweep": "passed",
    })


if __name__ == "__main__":
    main()
