"""Analyze a captured MaixCam stdout log instead of judging a few frames."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STAT_RE = re.compile(r"STAT,(\d+),FPS,([0-9.]+),RAW,(\d+),CAND,(\d+),EGGS,(\d+),TILE,(.*)")
DOT_RE = re.compile(r"DOT,(-?\d+),(-?\d+),SCORE,([0-9.]+),FRESH,(\d),MISSES,(\d+)")
AIM_RE = re.compile(r"AIM,(TRACK|PREDICT),\d+,EX,(-?[0-9.]+),EY,(-?[0-9.]+),PAN,(-?[0-9.]+),TILT,(-?[0-9.]+)")
GIMBAL_RE = re.compile(r"GIMBAL,TRACK,PAN_OFFSET,(-?[0-9.]+),TILT_OFFSET,(-?[0-9.]+),PAN_ANGLE,(-?[0-9.]+),TILT_ANGLE,(-?[0-9.]+)")
EGG_RE = re.compile(r"^EGG,\d+,")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.log.read_bytes()
    # PowerShell Tee-Object commonly writes UTF-16LE; SSH logs are usually UTF-8.
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    lines = raw.decode(encoding, errors="replace").splitlines()
    stats = []
    dots = []
    aims = []
    gimbals = []
    egg_lines = 0
    relay_on = relay_off = relay_transitions = 0
    target_complete = 0
    for line in lines:
        match = STAT_RE.search(line)
        if match:
            stats.append({"frame": int(match[1]), "fps": float(match[2]), "raw": int(match[3]),
                          "cand": int(match[4]), "eggs": int(match[5]), "tile": match[6]})
        match = DOT_RE.search(line)
        if match:
            dots.append({"x": int(match[1]), "y": int(match[2]), "score": float(match[3]),
                         "fresh": bool(int(match[4])), "misses": int(match[5])})
        match = AIM_RE.search(line)
        if match:
            aims.append({"kind": match[1], "ex": float(match[2]), "ey": float(match[3]),
                         "pan": float(match[4]), "tilt": float(match[5])})
        match = GIMBAL_RE.search(line)
        if match:
            gimbals.append({"pan_offset": float(match[1]), "tilt_offset": float(match[2]),
                            "pan_angle": float(match[3]), "tilt_angle": float(match[4])})
        if EGG_RE.search(line):
            egg_lines += 1
        if "AIM_RELAY_STATE,ON" in line:
            relay_on += 1
        if "AIM_RELAY_STATE,OFF" in line:
            relay_off += 1
        if "AIM_RELAY_TRANSITION" in line:
            relay_transitions += 1
        if "TARGET,COMPLETE" in line:
            target_complete += 1

    fresh = sum(1 for item in dots if item["fresh"])
    track = sum(1 for item in aims if item["kind"] == "TRACK")
    predict = sum(1 for item in aims if item["kind"] == "PREDICT")
    report = {
        "stat_samples": len(stats),
        "fps_mean": round(sum(x["fps"] for x in stats) / len(stats), 3) if stats else 0.0,
        "fps_min": min((x["fps"] for x in stats), default=0.0),
        "fps_max": max((x["fps"] for x in stats), default=0.0),
        "raw_mean": round(sum(x["raw"] for x in stats) / len(stats), 3) if stats else 0.0,
        "candidate_mean": round(sum(x["cand"] for x in stats) / len(stats), 3) if stats else 0.0,
        "egg_mean": round(sum(x["eggs"] for x in stats) / len(stats), 3) if stats else 0.0,
        "egg_min": min((x["eggs"] for x in stats), default=0),
        "egg_max": max((x["eggs"] for x in stats), default=0),
        "egg_lines": egg_lines,
        "dot_samples": len(dots),
        "dot_fresh_ratio": round(fresh / len(dots), 3) if dots else 0.0,
        "aim_samples": len(aims),
        "aim_track_ratio": round(track / len(aims), 3) if aims else 0.0,
        "aim_predict_ratio": round(predict / len(aims), 3) if aims else 0.0,
        "aim_abs_error_mean": round(sum((abs(x["ex"]) + abs(x["ey"])) / 2 for x in aims) / len(aims), 4) if aims else 0.0,
        "aim_abs_error_max": round(max(((abs(x["ex"]) + abs(x["ey"])) / 2 for x in aims), default=0.0), 4),
        "pan_angle_min": min((x["pan_angle"] for x in gimbals), default=None),
        "pan_angle_max": max((x["pan_angle"] for x in gimbals), default=None),
        "tilt_angle_min": min((x["tilt_angle"] for x in gimbals), default=None),
        "tilt_angle_max": max((x["tilt_angle"] for x in gimbals), default=None),
        "relay_on_events": relay_on,
        "relay_off_events": relay_off,
        "relay_transitions": relay_transitions,
        "target_complete_events": target_complete,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
