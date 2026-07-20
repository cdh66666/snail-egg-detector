"""Host regression checks for track-specific weak detections and gimbal compensation."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "maixcam" / "main.py"


def load_namespace():
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    wanted = {
        "ScalarKalman",
        "obj_center",
        "box_iou",
        "track_box",
        "track_match_cost",
        "gimbal_image_shift",
        "measurement_noise_for_score",
        "fresh_detection_for_safety",
    }
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    ns = {
        "TRACK_MAX_VELOCITY_PX": 27.0,
        "TRACK_MIN_DT_FRAMES": 0.35,
        "TRACK_MAX_DT_FRAMES": 2.50,
        "TRACK_KF_PROCESS_NOISE": 2.0,
        "TRACK_KF_MEASUREMENT_NOISE": 16.0,
        "TRACK_MATCH_MIN_DISTANCE_PX": 45.0,
        "TRACK_MATCH_DISTANCE_SCALE": 1.35,
        "TRACK_ASSOC_MAX_SIZE_RATIO": 1.9,
        "TRACK_ASSOC_MAX_COST": 1.45,
        "TRACK_LOCKED_LOW_ASSOC_MAX_COST": 0.92,
        "LOCKED_ASSOC_MIN_IOU": 0.05,
        "LOCKED_ASSOC_MAX_CENTER_RATIO": 0.90,
        "LOCKED_ASSOC_MAX_SIZE_RATIO": 1.55,
        "DISCOVERY_MODEL_CONF": 0.35,
        "_manual_lock_active": True,
        "_locked_track_id": 7,
        "FRAME_W": 320,
        "FRAME_H": 224,
        "CAMERA_H_FOV_DEG": 55.7,
        "CAMERA_V_FOV_DEG": 36.5,
        "GIMBAL_PAN_SIGN": -1.0,
        "GIMBAL_TILT_SIGN": -1.0,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(MAIN), "exec"), ns)
    return ns


def make_track(ns, track_id=7, cx=120, cy=90, w=30, h=40):
    return {
        "id": track_id,
        "cx": ns["ScalarKalman"](cx),
        "cy": ns["ScalarKalman"](cy),
        "w": ns["ScalarKalman"](w),
        "h": ns["ScalarKalman"](h),
    }


def detection(cx, cy, w=30, h=40, score=0.15):
    return SimpleNamespace(x=cx - w / 2, y=cy - h / 2, w=w, h=h, score=score)


def main():
    ns = load_namespace()
    cost = ns["track_match_cost"]
    selected = make_track(ns)

    # A weak observation can update the selected trajectory.
    assert cost(detection(123, 92), selected) is not None
    # The same weak observation cannot update or create an unrelated track.
    assert cost(detection(123, 92), make_track(ns, track_id=8)) is None
    # Proximity alone is insufficient when shape changes too much.
    assert cost(detection(122, 91, w=8, h=70), selected) is None
    # A distant weak observation is rejected even for the selected ID.
    assert cost(detection(250, 180), selected) is None

    # A nearby same-size small target must not steal a phone-selected ID.
    small_selected = make_track(ns, track_id=7, cx=120, cy=90, w=16, h=16)
    assert cost(detection(123, 91, w=16, h=16, score=0.42), small_selected) is not None
    assert cost(detection(136, 90, w=16, h=16, score=0.42), small_selected) is None

    shift = ns["gimbal_image_shift"]
    pixels_per_pan = ns["FRAME_W"] / ns["CAMERA_H_FOV_DEG"]
    # With the calibrated negative pan sign, a -5 degree servo move shifts the
    # same world target left by about five degrees worth of image pixels.
    dx, dy = shift(0.0, 0.0, -5.0, 0.0)
    assert abs(dx + 5.0 * pixels_per_pan) < 1e-6
    assert abs(dy) < 1e-6

    # Variable loop intervals must scale prediction distance instead of
    # treating a slow iteration as exactly one camera frame.
    kalman = ns["ScalarKalman"](0)
    kalman.vel = 10.0
    kalman.predict(2.0)
    assert abs(kalman.pos - 20.0) < 1e-6

    # A weak but accepted measurement observes the trajectory without pulling
    # it as hard as a high-confidence detection.
    strong = ns["ScalarKalman"](0)
    weak = ns["ScalarKalman"](0)
    strong.update(20.0, ns["measurement_noise_for_score"](0.35))
    weak.update(20.0, ns["measurement_noise_for_score"](0.10))
    assert ns["measurement_noise_for_score"](0.10) > ns["measurement_noise_for_score"](0.35)
    assert abs(weak.pos) < abs(strong.pos)

    safety = ns["fresh_detection_for_safety"]
    assert safety([SimpleNamespace(score=0.50, associated_track_id=0)])
    assert safety([SimpleNamespace(score=0.15, associated_track_id=7)])
    assert not safety([SimpleNamespace(score=0.15, associated_track_id=8)])

    source = MAIN.read_text(encoding="utf-8")
    assert "obj.score >= birth_threshold" in source
    filter_body = source[source.index("def filter_candidates"):source.index("class ScalarKalman")]
    assert "track_match_cost(" not in filter_body
    assert "elif score_ok and _manual_lock_active:" in filter_body
    assert "color_ok = not red_reject" in filter_body
    loop_body = source[source.index("while not app.need_exit()") :]
    assert loop_body.index("shift_measurements_to_current_frame(") < loop_body.index("stable = update_tracks(")
    print({"weak_selected_update": "passed", "weak_track_birth_blocked": "passed", "gimbal_compensation": "passed", "variable_dt_prediction": "passed", "weak_measurement_damping": "passed", "aligned_weak_gate": "passed", "safety_gate": "passed"})


if __name__ == "__main__":
    main()
