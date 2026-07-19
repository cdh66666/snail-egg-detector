"""Analyze a captured MaixCam stdout log instead of judging a few frames."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STAT_RE = re.compile(
    r"STAT,(\d+),(?:FPS|LOOP_FPS),([0-9.]+),"
    r"(?:DETECT_HZ,([0-9.]+),STREAM_FPS,([0-9.]+),)?"
    r"(?:YOLO_FRAME,\d+,)?RAW,(\d+),CAND,(\d+),EGGS,(\d+),TILE,(.*)"
)
DOT_RE = re.compile(r"DOT,(-?\d+),(-?\d+),SCORE,([0-9.]+),FRESH,(\d),MISSES,(\d+)")
AIM_RE = re.compile(r"AIM,(TRACK|PREDICT),\d+,EX,(-?[0-9.]+),EY,(-?[0-9.]+),PAN,(-?[0-9.]+),TILT,(-?[0-9.]+)")
GIMBAL_RE = re.compile(r"GIMBAL,TRACK,PAN_OFFSET,(-?[0-9.]+),TILT_OFFSET,(-?[0-9.]+),PAN_ANGLE,(-?[0-9.]+),TILT_ANGLE,(-?[0-9.]+)")
EGG_RE = re.compile(r"^EGG,\d+,")
PROF_RE = re.compile(
    r"PROF,(\d+),FRAMES,(\d+),DETECTS,(\d+),READ_MS,([0-9.]+),DOT_MS,([0-9.]+),"
    r"DETECT_MS,([0-9.]+),CONTROL_DRAW_MS,([0-9.]+),"
    r"(?:ENCODE_MS,([0-9.]+),ENQUEUE_MS,([0-9.]+),)?DISPLAY_MS,([0-9.]+)"
)
TRACK_RE = re.compile(
    r"TRACK,(\d+),DETECT,(\d+),RAW,(\d+),CAND,(\d+),EGGS,(\d+),PRIMARY,(\d+),"
    r"(?:RAWID,(\d+),)?"
    r"PCX,(-?\d+),PCY,(-?\d+),PRED,(\d+),MISSES,(-?\d+),FRESH,(\d+)"
)
AIMSTAT_RE = re.compile(
    r"AIMSTAT,(\d+),VALID,(\d+),EX,(-?[0-9.]+),EY,(-?[0-9.]+),DOT,(\d+),PRIMARY,(\d+)"
)


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
    profiles = []
    tracks = []
    aimstats = []
    egg_lines = 0
    relay_on = relay_off = relay_transitions = 0
    target_complete = 0
    for line in lines:
        match = STAT_RE.search(line)
        if match:
            stats.append({
                "frame": int(match[1]), "fps": float(match[2]),
                "detect_hz": float(match[3] or 0.0), "stream_fps": float(match[4] or 0.0),
                "raw": int(match[5]), "cand": int(match[6]), "eggs": int(match[7]), "tile": match[8],
            })
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
        match = PROF_RE.search(line)
        if match:
            profiles.append({
                "frame": int(match[1]), "frames": int(match[2]), "detects": int(match[3]),
                "read_ms": float(match[4]), "dot_ms": float(match[5]),
                "detect_ms": float(match[6]), "control_draw_ms": float(match[7]),
                "encode_ms": float(match[8] or 0.0), "enqueue_ms": float(match[9] or 0.0),
                "display_ms": float(match[10]),
            })
        match = TRACK_RE.search(line)
        if match:
            tracks.append({
                "frame": int(match[1]), "detect": bool(int(match[2])), "raw": int(match[3]),
                "cand": int(match[4]), "eggs": int(match[5]), "primary": int(match[6]),
                "raw_id": int(match[7] or 0), "cx": int(match[8]), "cy": int(match[9]),
                "predicted": bool(int(match[10])), "misses": int(match[11]),
                "fresh": bool(int(match[12])),
            })
        match = AIMSTAT_RE.search(line)
        if match:
            aimstats.append({
                "frame": int(match[1]), "valid": bool(int(match[2])), "ex": float(match[3]),
                "ey": float(match[4]), "dot_fresh": bool(int(match[5])), "primary": int(match[6]),
            })
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
        "detect_hz_mean": round(sum(x["detect_hz"] for x in stats) / len(stats), 3) if stats else 0.0,
        "stream_publish_hz_mean": round(sum(x["stream_fps"] for x in stats) / len(stats), 3) if stats else 0.0,
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
        "profile_samples": len(profiles),
        "profile_frames": sum(x["frames"] for x in profiles),
        "profile_detects": sum(x["detects"] for x in profiles),
        "track_samples": len(tracks),
        "track_predicted_ratio": round(sum(x["predicted"] for x in tracks) / len(tracks), 3) if tracks else 0.0,
        "track_fresh_ratio": round(sum(x["fresh"] for x in tracks) / len(tracks), 3) if tracks else 0.0,
        "track_no_primary_samples": sum(1 for x in tracks if x["primary"] == 0),
        "track_primary_switches": sum(
            1 for before, after in zip([x for x in tracks if x["primary"]], [x for x in tracks if x["primary"]][1:])
            if before["primary"] != after["primary"]
        ) if tracks else 0,
        "track_raw_id_switches": sum(
            1 for before, after in zip([x for x in tracks if x["raw_id"]], [x for x in tracks if x["raw_id"]][1:])
            if before["raw_id"] != after["raw_id"]
        ) if tracks else 0,
        "aimstat_samples": len(aimstats),
        "aimstat_valid_ratio": round(sum(x["valid"] for x in aimstats) / len(aimstats), 3) if aimstats else 0.0,
        "aimstat_error_mean": round(
            sum((abs(x["ex"]) + abs(x["ey"])) / 2 for x in aimstats if x["valid"])
            / max(1, sum(1 for x in aimstats if x["valid"])), 4
        ) if aimstats else 0.0,
    }
    report["detect_hz_estimate"] = round(
        report["fps_mean"] * sum(x["detects"] for x in profiles) / max(1, sum(x["frames"] for x in profiles)),
        3,
    ) if profiles and stats else 0.0

    def percentile(values, fraction):
        ordered = sorted(values)
        if not ordered:
            return 0.0
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
        return ordered[index]

    for key in ("read_ms", "dot_ms", "detect_ms", "control_draw_ms", "encode_ms", "enqueue_ms", "display_ms"):
        report[f"{key}_mean"] = round(sum(x[key] for x in profiles) / len(profiles), 3) if profiles else 0.0
        report[f"{key}_p95"] = round(percentile([x[key] for x in profiles], 0.95), 3) if profiles else 0.0
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
