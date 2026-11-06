"""Coarse page orientation (0/90/180/270) via Tesseract OSD."""
from __future__ import annotations

import shutil
import subprocess

import cv2
import numpy as np


def osd_rotation(gray: np.ndarray, min_conf: float = 8.0,
                 min_script_conf: float = 1.0) -> int:
    """Clockwise degrees (0/90/180/270) that put the page upright; 0 without tesseract.

    Thresholds are strict: on handwriting OSD answers 180 with confidence ~5,
    and a wrongly flipped page is far worse than a missed rotation.
    """
    if shutil.which("tesseract") is None:
        return 0
    ok, buf = cv2.imencode(".png", gray)
    if not ok:
        return 0
    try:
        r = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "0"],
                           input=buf.tobytes(), capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        return 0
    rot, conf, sconf = 0, 0.0, 0.0
    for line in r.stdout.decode(errors="ignore").splitlines():
        if line.startswith("Rotate:"):
            rot = int(line.split(":")[1])
        elif line.startswith("Orientation confidence:"):
            conf = float(line.split(":")[1])
        elif line.startswith("Script confidence:"):
            sconf = float(line.split(":")[1])
    return rot if conf >= min_conf and sconf >= min_script_conf else 0


def apply_rotation(img: np.ndarray, deg: int) -> np.ndarray:
    """Rotate clockwise by 90/180/270 degrees; other values are a no-op."""
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
             270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    return cv2.rotate(img, codes[deg]) if deg in codes else img
