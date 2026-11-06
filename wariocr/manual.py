"""Hand-drawn sheet outlines (pure blue strokes over a `--debug` image)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .geometry import hull_to_quad, line_intersection, order_corners
from .io import EXTS


def find_annotation(corners_dir: str | Path | None, stem: str) -> Path | None:
    """The `<stem>_detect.*` image in `corners_dir`, if any."""
    if not corners_dir:
        return None
    for f in Path(corners_dir).glob(f"{stem}_detect.*"):
        if f.suffix.lower() in EXTS:
            return f
    return None


def manual_quad(annot: Path, shape) -> np.ndarray | None:
    """Sheet outline drawn in pure blue over a `--debug` image, in photo pixels.

    The sheet is the region enclosed by the blue strokes and the image border
    (a side may be left undrawn where the sheet runs out of the frame).
    Only saturated, bright blue counts, so ballpoint ink is ignored.
    """
    im = cv2.imread(str(annot))
    if im is None:
        return None
    ah, aw = im.shape[:2]
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (105, 150, 150), (130, 255, 255))
    # keep only long strokes
    n, lab, stats, _ = cv2.connectedComponentsWithStats(blue, connectivity=8)
    keep = [i for i in range(1, n)
            if max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) > 0.1 * min(ah, aw)]
    if not keep:
        return None
    blue = np.isin(lab, keep).astype(np.uint8) * 255
    walls = cv2.dilate(blue, np.ones((5, 5), np.uint8))
    region = np.where(walls > 0, 0, 255).astype(np.uint8)
    ff = np.zeros((ah + 2, aw + 2), np.uint8)
    cv2.floodFill(region, ff, (aw // 2, ah // 2), 128)
    inside = (region == 128).astype(np.uint8) * 255
    if not 0.1 < inside.mean() / 255 < 0.97:
        return None
    inside = cv2.dilate(inside, np.ones((5, 5), np.uint8))  # back to the stroke centre
    cnts, _ = cv2.findContours(inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    q = hull_to_quad(max(cnts, key=cv2.contourArea))
    if q is None:
        return None
    q = order_corners(q).astype(np.float64)
    # refine each drawn side on the stroke pixels (undrawn sides stay on the border)
    ys, xs = np.nonzero(blue)
    P = np.stack([xs, ys], 1).astype(np.float64)
    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        ln = np.linalg.norm(b - a)
        u = (b - a) / ln
        nrm = np.array([-u[1], u[0]])
        t = (P - a) @ u
        sel = P[(np.abs((P - a) @ nrm) < 6) & (t > 0.1 * ln) & (t < 0.9 * ln)]
        if len(sel) > 0.3 * ln:
            vx, vy, x0, y0 = cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER,
                                         0, 0.01, 0.01).ravel()
            lines.append((np.array([x0, y0]), np.array([vx, vy])))
        else:
            lines.append((a, u))
    out = []
    for i in range(4):
        (p1, u1), (p2, u2) = lines[i - 1], lines[i]
        x = line_intersection(p1, p1 + u1, p2, p2 + u2)
        out.append(q[i] if x is None else x)
    h, w = shape[:2]
    return (np.array(out) * [w / aw, h / ah]).astype(np.float32)
