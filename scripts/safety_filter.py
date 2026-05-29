from __future__ import annotations

import cv2
import numpy as np


def color_ratios_bgr(bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple[float, float]:
    h, w = bgr.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    if crop.shape[0] > 80 or crop.shape[1] > 80:
        crop = cv2.resize(crop, (80, 80), interpolation=cv2.INTER_AREA)

    b = crop[:, :, 0].astype(np.int16)
    g = crop[:, :, 1].astype(np.int16)
    r = crop[:, :, 2].astype(np.int16)

    red_only_reject = ((r >= 125) & (b < 50) & ((r - g) >= 32)) | ((r >= 150) & ((r - b) > 135))
    orange_red = (r >= 150) & (g >= 70) & (b < 62) & ((r - b) >= 110)
    # Real apple-snail egg masses are pink/salmon: they keep red and blue
    # separation from green. This rejects gray concrete and white highlights,
    # which otherwise look "pale pink" under broad RGB thresholds.
    pink = (
        (r >= 80)
        & (b >= 50)
        & (g >= 40)
        & (((r - g) >= 10) | ((b - g) >= 12))
        & ((r - b) <= 120)
        & ((r - b) >= -45)
        & ~red_only_reject
    )
    red_bad = red_only_reject | orange_red
    return float(pink.mean()), float(red_bad.mean())


def pink_ratio_bgr(bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    pink, _ = color_ratios_bgr(bgr, x1, y1, x2, y2)
    return pink


def pass_laser_safe_filter(
    bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    confidence: float,
    *,
    min_conf: float = 0.62,
    min_area_ratio: float = 0.00004,
    max_area_ratio: float = 0.12,
    min_aspect: float = 0.18,
    max_aspect: float = 5.5,
    min_pink_ratio: float = 0.03,
    max_red_bad_ratio: float = 0.55,
    red_bad_dominance: float = 2.6,
    strong_conf: float = 0.50,
) -> tuple[bool, float]:
    if confidence < min_conf:
        return False, 0.0
    h, w = bgr.shape[:2]
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    if bw <= 0 or bh <= 0:
        return False, 0.0
    area_ratio = (bw * bh) / float(w * h)
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return False, 0.0
    aspect = bw / float(bh)
    if aspect < min_aspect or aspect > max_aspect:
        return False, 0.0
    pink_ratio, red_ratio = color_ratios_bgr(bgr, x1, y1, x2, y2)
    if red_ratio > max_red_bad_ratio and red_ratio > pink_ratio * red_bad_dominance:
        return False, pink_ratio
    if confidence >= strong_conf:
        return True, pink_ratio
    return pink_ratio >= min_pink_ratio, pink_ratio
