"""Quad helpers: corner ordering, validity, line intersection, hull fitting."""
from __future__ import annotations

import cv2
import numpy as np

#: Paper formats, portrait (width, height) in millimetres.
PAPER_MM = {
    "a4": (210.0, 297.0),
    "a5": (148.0, 210.0),
    "letter": (215.9, 279.4),
    "legal": (215.9, 355.6),
}


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left (float32)."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def quad_is_valid(q: np.ndarray, shape) -> bool:
    """Plausible sheet outline: convex, large enough, near the frame, no extreme angles."""
    h, w = shape[:2]
    q = order_corners(q)
    if len({tuple(p) for p in q.round(1)}) < 4:
        return False
    if not cv2.isContourConvex(q.reshape(-1, 1, 2).astype(np.int32)):
        return False
    area = cv2.contourArea(q)
    if not (0.15 * w * h <= area <= 1.3 * w * h):
        return False
    # corners may fall outside the frame (sheet cut off), but not far
    if (q[:, 0] < -0.25 * w).any() or (q[:, 0] > 1.25 * w).any() \
            or (q[:, 1] < -0.25 * h).any() or (q[:, 1] > 1.25 * h).any():
        return False
    for i in range(4):
        a, b, c = q[i - 1], q[i], q[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        if not (55 <= ang <= 125):
            return False
    return True


def line_intersection(p1, p2, p3, p4):
    """Intersection of lines p1p2 and p3p4, None if (nearly) parallel."""
    d1, d2 = p2 - p1, p4 - p3
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den
    return p1 + t * d1


def hull_to_quad(cnt: np.ndarray) -> np.ndarray | None:
    """Minimum-area enclosing quad of the convex hull (greedy edge removal).

    Unlike approxPolyDP it keeps the long sides exact and extends them to meet,
    so rounded, dog-eared or out-of-frame corners land where the sheet's corner is.
    """
    hull = cv2.convexHull(cnt).reshape(-1, 2).astype(np.float64)
    peri = cv2.arcLength(hull.astype(np.float32), True)
    pts = cv2.approxPolyDP(hull.astype(np.float32), 0.004 * peri, True)
    pts = pts.reshape(-1, 2).astype(np.float64)
    if len(pts) < 4:
        return None
    while len(pts) > 4:
        n = len(pts)
        best, best_i, best_x = np.inf, -1, None
        for i in range(n):
            a, b = pts[i - 1], pts[i]
            c, d = pts[(i + 1) % n], pts[(i + 2) % n]
            x = line_intersection(a, b, d, c)
            if x is None:
                continue
            # x must lie beyond b and c (edges converge outward)
            if np.dot(x - b, b - a) < 0 or np.dot(x - c, c - d) < 0:
                continue
            added = abs((c[0] - b[0]) * (x[1] - b[1]) - (x[0] - b[0]) * (c[1] - b[1])) / 2
            if added < best:
                best, best_i, best_x = added, i, x
        if best_i < 0:
            return None
        i = best_i
        nxt = (i + 1) % n
        pts[i] = best_x
        pts = np.delete(pts, nxt, axis=0)
    return pts.astype(np.float32)


def side_on_frame(a: np.ndarray, b: np.ndarray, shape) -> bool:
    """True if a quad side is the photo border (or beyond it), not a sheet edge."""
    h, w = shape[:2]
    tol = 0.015 * max(h, w)
    for idx, hi in ((0, w - 1), (1, h - 1)):
        for edge in (0, hi):
            if abs(a[idx] - edge) < tol and abs(b[idx] - edge) < tol:
                return True
    m = (a + b) / 2
    return bool(m[0] < -tol or m[0] > w - 1 + tol or m[1] < -tol or m[1] > h - 1 + tol)
