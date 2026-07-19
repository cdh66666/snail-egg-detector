"""Host-side regression test for multi-target lock persistence."""

import ast
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "maixcam" / "main.py"
ASSIGNMENTS = {
    "PRIMARY_TARGET_POLICY",
    "PRIMARY_COLUMN_TOLERANCE_PX",
    "LOCK_TARGET_ID",
    "LOCK_REACQUIRE_RADIUS_PX",
    "LOCK_REACQUIRE_MAX_COST",
    "LOCK_RELEASE_MISSING_FRAMES",
    "LOCK_SWITCH_MISSING_FRAMES",
    "GIMBAL_PREDICT_MAX_MISSES",
    "FRAME_W",
    "FRAME_H",
    "_locked_track_id",
    "_locked_last_center",
    "_locked_last_size",
    "_locked_missing_frames",
    "_primary_center",
    "_completed_track_ids",
}


def load_namespace():
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & ASSIGNMENTS:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {"obj_center", "select_primary_target"}:
            selected.append(node)
    namespace = {"USE_FAST_MODEL": False, "FAST_MODEL_IS_YOLO11": False}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(MAIN), "exec"), namespace)
    namespace["cycle_targets_requested"] = lambda: False
    return namespace


def target(track_id, cx, cy, w=24, h=26, predicted=False, misses=0):
    return SimpleNamespace(
        track_id=track_id,
        x=cx - w * 0.5,
        y=cy - h * 0.5,
        w=w,
        h=h,
        predicted=predicted,
        misses=misses,
    )


def main():
    ns = load_namespace()
    assert 20 <= ns["LOCK_RELEASE_MISSING_FRAMES"] <= 30
    ns["PRIMARY_TARGET_POLICY"] = "leftmost"
    choose = ns["select_primary_target"]

    original = target(3, 110, 120, w=22, h=26)
    distractor = target(4, 220, 120)
    assert choose([original, distractor]) is original

    # A tracker ID change and a large box-shape fluctuation must reacquire the
    # nearest physical target instead of switching to a similarly colored egg.
    same_target = target(19, 119, 124, w=20, h=50)
    closer_left_distractor = target(20, 65, 120)
    old_prediction = target(3, 110, 120, w=22, h=26, predicted=True, misses=1)
    assert choose([old_prediction, same_target, closer_left_distractor]) is same_target
    assert ns["_locked_track_id"] == 19

    # A nearby round target is not the same object as the locked tall mass.
    round_neighbor = target(22, 160, 125, w=28, h=26)
    assert choose([round_neighbor]) is None
    assert ns["_locked_track_id"] == 19

    # A fresh unrelated detection should take over after the short switch
    # window instead of leaving the gimbal parked on an empty lock.
    far_only = target(21, 400, 300)
    remaining_hold = ns["LOCK_SWITCH_MISSING_FRAMES"] - ns["_locked_missing_frames"]
    for _ in range(max(0, remaining_hold - 1)):
        assert choose([far_only]) is None
        assert ns["_locked_track_id"] == 19

    released = choose([far_only])
    assert released is far_only
    assert ns["_locked_track_id"] == 21

    manual_ns = load_namespace()
    manual_ns["PRIMARY_TARGET_POLICY"] = "leftmost"
    manual_choose = manual_ns["select_primary_target"]
    selected = target(41, 100, 100)
    other = target(42, 240, 120)
    assert manual_choose([selected, other]) is selected
    # Even a look-alike immediately beside the last selected position must not
    # steal a phone-selected lock when the original detector box disappears.
    nearby_other = target(43, 108, 103, w=24, h=26)
    assert manual_choose([nearby_other], manual_lock=True) is None
    assert manual_ns["_locked_track_id"] == 41
    selected_prediction = target(41, 102, 101, predicted=True, misses=2)
    assert manual_choose([selected_prediction, nearby_other], manual_lock=True) is selected_prediction
    assert manual_ns["_locked_track_id"] == 41
    for _ in range(manual_ns["LOCK_RELEASE_MISSING_FRAMES"] + 4):
        assert manual_choose([other], manual_lock=True) is None
        assert manual_ns["_locked_track_id"] == 41

    robust_ns = load_namespace()
    robust_choose = robust_ns["select_primary_target"]
    tiny_edge = target(31, 18, 30, w=14, h=15)
    large_center = target(32, 350, 235, w=55, h=70)
    large_center.score = 0.8
    tiny_edge.score = 0.9
    assert robust_choose([tiny_edge, large_center]) is large_center
    print(
        {
            "nearest_reacquire": "passed",
            "fast_visible_failover": "passed",
            "visible_switch_frames": ns["LOCK_SWITCH_MISSING_FRAMES"],
            "empty_view_hold_frames": ns["LOCK_RELEASE_MISSING_FRAMES"],
            "manual_selection_no_switch": "passed",
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
