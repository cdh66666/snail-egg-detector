"""Sequence-level regression for a selected target during gimbal motion."""

from __future__ import annotations

from types import SimpleNamespace

from test_track_association import detection, load_namespace, make_track


def shifted_detection(ns, cx, cy, from_pan, to_pan, score=0.14, w=30, h=40):
    dx, dy = ns["gimbal_image_shift"](from_pan, 0.0, to_pan, 0.0)
    return detection(cx + dx, cy + dy, w=w, h=h, score=score), dx, dy


def main():
    ns = load_namespace()
    selected = make_track(ns, track_id=7, cx=120, cy=90, w=30, h=40)
    neighbor = make_track(ns, track_id=8, cx=190, cy=94, w=28, h=38)
    lower = make_track(ns, track_id=9, cx=82, cy=164, w=34, h=36)
    tracks = [selected, neighbor, lower]
    pan_steps = (0.0, -1.0, -2.2, -3.5, -4.7, -5.5)

    for old_pan, new_pan in zip(pan_steps, pan_steps[1:]):
        weak_selected, dx, dy = shifted_detection(ns, 120, 90, old_pan, new_pan)
        for track in tracks:
            track["cx"].shift(dx)
            track["cy"].shift(dy)

        selected_cost = ns["track_match_cost"](weak_selected, selected)
        assert selected_cost is not None
        assert ns["track_match_cost"](weak_selected, neighbor) is None
        assert ns["track_match_cost"](weak_selected, lower) is None

        noise = ns["measurement_noise_for_score"](weak_selected.score)
        selected["cx"].update(weak_selected.x + weak_selected.w * 0.5, noise)
        selected["cy"].update(weak_selected.y + weak_selected.h * 0.5, noise)

    # Once the selected mass disappears, a strong neighboring detection may
    # remain visible but must not count as a weak continuation of track 7.
    visible_neighbor = SimpleNamespace(
        x=neighbor["cx"].pos - 14,
        y=neighbor["cy"].pos - 19,
        w=28,
        h=38,
        score=0.20,
        associated_track_id=8,
    )
    assert ns["track_match_cost"](visible_neighbor, selected) is None
    assert not ns["fresh_detection_for_safety"]([visible_neighbor])

    print({
        "frames": len(pan_steps) - 1,
        "selected_track": 7,
        "neighbor_switch": "blocked",
        "dual_buffer_gimbal_alignment": "passed",
        "weak_selected_continuity": "passed",
    })


if __name__ == "__main__":
    main()
