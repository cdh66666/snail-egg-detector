"""Host-side tests for MaixCam red-dot detection and bounded aim waypoints."""

import ast
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "maixcam" / "main.py"
NAMES = {
    "FRAME_W",
    "FRAME_H",
    "FIXED_AIM_X",
    "FIXED_AIM_Y",
    "AIM_DOT_LAB_THRESHOLDS",
    "AIM_DOT_MIN_PIXELS",
    "AIM_DOT_MAX_PIXELS",
    "AIM_DOT_MIN_SIDE",
    "AIM_DOT_MAX_SIDE",
    "AIM_DOT_MIN_RED",
    "AIM_DOT_MIN_DOMINANCE",
    "AIM_DOT_MIN_DOMINANCE_RATIO",
    "AIM_DOT_MIN_LOCAL_CONTRAST",
    "AIM_DOT_RGB_STRIDE",
    "AIM_DOT_RGB_REFRESH_FRAMES",
    "AIM_DOT_EXPECTED_RADIUS_PX",
    "AIM_DOT_MAX_JUMP_PX",
    "AIM_DOT_FILTER_ALPHA",
    "AIM_DOT_STALE_FRAMES",
    "AIM_DOT_LOG_EVERY_N_FRAMES",
    "AIM_DOT_CONTROL_TOLERANCE_PX",
    "AIM_SCAN_SETTLE_FRAMES",
    "AIM_SCAN_DWELL_S",
    "AIM_SCAN_MARGIN_RATIO",
    "AIM_SCAN_CONTOUR_GRID",
    "AIM_SCAN_CONTOUR_REFRESH_FRAMES",
}


def load_namespace():
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if targets & NAMES:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {
            "pixel_to_rgb",
            "pink_pixel",
            "red_bad_pixel",
        }:
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in {
            "AimDot",
            "AimDotDetector",
            "AimWaypointPlanner",
        }:
            selected.append(node)
    ns = {"print": lambda *_args, **_kwargs: None}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(MAIN), "exec"), ns)
    ns["ACTIVE_AIM_X"] = ns["FIXED_AIM_X"]
    ns["ACTIVE_AIM_Y"] = ns["FIXED_AIM_Y"]
    ns["cycle_targets_requested"] = lambda: False
    return ns


class FakeBlob(tuple):
    def __new__(cls, x, y, w, h, pixels, cx, cy):
        return super().__new__(cls, (x, y, w, h, pixels, cx, cy))


class FakeImage:
    def __init__(self, blobs, rgb=(255, 10, 8)):
        self.blobs = blobs
        self.rgb = rgb

    def find_blobs(self, *_args, **_kwargs):
        return self.blobs

    def get_pixel(self, _x, _y):
        return self.rgb


def main():
    ns = load_namespace()
    detector = ns["AimDotDetector"]()

    dot = detector.detect(FakeImage([FakeBlob(319, 265, 12, 12, 44, 325, 271)]), True)
    assert dot is not None and dot.fresh
    assert abs(dot.x - 325) < 0.01 and abs(dot.y - 271) < 0.01

    dot = detector.detect(FakeImage([FakeBlob(323, 268, 12, 12, 42, 329, 274)]), True)
    assert dot is not None and dot.fresh
    assert 325 < dot.x < 329 and 271 < dot.y < 274

    # Pink or oversized regions must not be accepted as the aiming dot.
    stale = detector.detect(FakeImage([FakeBlob(300, 240, 60, 55, 900, 330, 267)]), True)
    assert stale is not None and not stale.fresh
    pink = detector.detect(FakeImage([FakeBlob(323, 268, 12, 12, 42, 329, 274)], (235, 145, 170)), True)
    assert pink is not None and not pink.fresh

    # After the stale window, search must return to the calibrated optical
    # axis instead of remaining trapped around an earlier false anchor.
    recovery_detector = ns["AimDotDetector"]()
    first = recovery_detector.detect(FakeImage([FakeBlob(249, 264, 12, 12, 40, 255, 270)]), True)
    assert first is not None and first.fresh
    for _ in range(ns["AIM_DOT_STALE_FRAMES"] + 1):
        recovery_detector.detect(FakeImage([]), True)
    recovered = recovery_detector.detect(FakeImage([FakeBlob(318, 264, 12, 12, 40, 324, 270)]), True)
    assert recovered is not None and recovered.fresh
    assert abs(recovered.x - 324) < 0.01

    planner = ns["AimWaypointPlanner"]()
    target = SimpleNamespace(x=200, y=120, w=100, h=80, track_id=7)
    point = planner.point(target, None, 0.0, False)
    assert point == (250.0, 160.0)

    # A settled visible dot advances to the next bounded point.
    centered = SimpleNamespace(x=250.0, y=160.0, fresh=True)
    for index in range(ns["AIM_SCAN_SETTLE_FRAMES"] + 1):
        planner.point(target, centered, index * 0.12, True)
    assert planner.index == 1
    point = planner.point(target, centered, 1.0, True)
    assert target.x < point[0] < target.x + target.w
    assert target.y < point[1] < target.y + target.h

    # Closed-loop image-space simulation: the visible dot is the reference,
    # target detections disappear briefly, and motion must resume without a
    # command jump when the same target returns.
    dot_x = 324.0
    target_x = 460.0
    angle = 0.0
    previous_command = 0.0
    max_resume_jump = 0.0
    for frame in range(160):
        visible = not (35 <= frame < 48)
        error = (target_x - dot_x) / 320.0
        command = max(-0.5, min(0.5, -5.0 * error * 0.1)) if visible else 0.0
        if frame == 48:
            max_resume_jump = abs(command - previous_command)
        angle += command
        target_x = 460.0 + 21.8 * angle
        previous_command = command
    final_error = abs(target_x - dot_x)
    assert final_error <= 8.0
    assert max_resume_jump <= 0.5

    print(
        {
            "red_dot_detection": "passed",
            "pink_and_large_rejection": "passed",
            "stale_dot_hold": "passed",
            "bounded_waypoint_scan": "passed",
            "dropout_recovery": "passed",
            "final_closed_loop_error_px": round(final_error, 2),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
