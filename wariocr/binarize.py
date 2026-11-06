"""Black-and-white conversion: Sauvola threshold and speck removal."""
from __future__ import annotations

import cv2
import numpy as np


def sauvola(gray: np.ndarray, window: int, k: float = 0.18, R: float = 128.0) -> np.ndarray:
    """Sauvola local threshold: 255 = paper, 0 = ink."""
    g = gray.astype(np.float64)
    mean = cv2.boxFilter(g, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(g * g, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    thr = mean * (1 + k * (std / R - 1))
    return np.where(g > thr, 255, 0).astype(np.uint8)


def despeckle(bw: np.ndarray, dpi: int) -> np.ndarray:
    """Drop ink specks smaller than ~5 px at 300 dpi (scaled with dpi)."""
    min_area = max(3, int(5 * (dpi / 300.0) ** 2))
    inv = 255 - bw
    _, lab, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    small = np.where(stats[:, cv2.CC_STAT_AREA] < min_area)[0]
    small = small[small != 0]
    if small.size:
        inv[np.isin(lab, small)] = 0
    return 255 - inv
