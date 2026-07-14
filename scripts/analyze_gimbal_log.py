"""Summarize MaixCam gimbal tracking logs without third-party dependencies."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize(path: Path) -> dict:
    error_x: list[float] = []
    error_y: list[float] = []
    fps: list[float] = []
    track_ids: set[int] = set()
    hold_reasons: dict[str, int] = {}

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if "AIM,TRACK," in line:
            line = line[line.index("AIM,TRACK,") :]
            parts = line.split(",")
            try:
                track_ids.add(int(parts[2]))
                error_x.append(abs(float(parts[4])))
                error_y.append(abs(float(parts[6])))
            except (ValueError, IndexError):
                continue
        elif "AIM,HOLD," in line:
            line = line[line.index("AIM,HOLD,") :]
            reason = line.split(",", 2)[-1]
            hold_reasons[reason] = hold_reasons.get(reason, 0) + 1
        elif "STAT," in line and ",FPS," in line:
            line = line[line.index("STAT,") :]
            parts = line.split(",")
            try:
                fps.append(float(parts[3]))
            except (ValueError, IndexError):
                continue

    combined = [math.hypot(x, y) for x, y in zip(error_x, error_y)]
    return {
        "log": str(path),
        "track_updates": len(combined),
        "track_ids": sorted(track_ids),
        "id_switch_count_upper_bound": max(0, len(track_ids) - 1),
        "mean_abs_error_x": statistics.fmean(error_x) if error_x else None,
        "mean_abs_error_y": statistics.fmean(error_y) if error_y else None,
        "mean_radial_error": statistics.fmean(combined) if combined else None,
        "p95_radial_error": percentile(combined, 0.95),
        "max_radial_error": max(combined) if combined else None,
        "mean_fps": statistics.fmean(fps) if fps else None,
        "min_fps": min(fps) if fps else None,
        "hold_reasons": hold_reasons,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize AIM and STAT lines from a MaixCam log")
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.log.is_file():
        parser.error("log does not exist: %s" % args.log)
    result = summarize(args.log)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

