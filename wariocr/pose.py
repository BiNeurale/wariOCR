"""Sheet pose from vanishing points and metric rectification at true scale."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import PAPER_MM, order_corners, side_on_frame


def _vanishing_point(segs: list, c: np.ndarray, s: float) -> np.ndarray | None:
    """Least-squares vanishing point of weighted segments (p, q, weight),
    homogeneous, in centred pixel coords scaled by 1/s. One outlier pass."""
    if len(segs) < 2:
        return None
    L, W, D, M = [], [], [], []
    for p, q, wt in segs:
        a = np.array([(p[0] - c[0]) / s, (p[1] - c[1]) / s, 1.0])
        b = np.array([(q[0] - c[0]) / s, (q[1] - c[1]) / s, 1.0])
        ln = np.cross(a, b)
        L.append(ln / (np.linalg.norm(ln[:2]) + 1e-12))
        W.append(wt)
        D.append((b - a)[:2] / (np.linalg.norm((b - a)[:2]) + 1e-12))
        M.append((a + b)[:2] / 2)
    L, W, D, M = map(np.array, (L, W, D, M))
    keep = np.ones(len(L), bool)
    v = None
    for _ in range(2):
        if keep.sum() < 2:
            return v
        A = (L[keep] * W[keep, None]).T @ L[keep]
        v = np.linalg.eigh(A)[1][:, 0]
        # angular residual: segment direction vs direction towards the VP
        to_v = v[:2][None, :] - M * v[2]
        to_v /= np.linalg.norm(to_v, axis=1, keepdims=True) + 1e-12
        keep = np.abs(D[:, 0] * to_v[:, 1] - D[:, 1] * to_v[:, 0]) < np.sin(np.radians(1.5))
    return v


def plane_pose(img: np.ndarray, quad: np.ndarray, real: list[bool]):
    """Camera focal + sheet plane axes from vanishing points.

    Evidence for each paper direction: the real (non-frame) quad sides plus
    long straight segments inside the sheet roughly parallel to them (ruled
    lines, table rules, text baselines). Focal comes from the orthogonality of
    the two vanishing points when measurable, else a typical phone value.
    Returns (K, e1, e2) with e1/e2 the sheet's horizontal/vertical 3D axes.
    """
    h, w = img.shape[:2]
    c = np.array([w / 2.0, h / 2.0])
    s = float(max(h, w))
    q = quad.astype(np.float64)
    sides = [(q[0], q[1]), (q[3], q[2]), (q[0], q[3]), (q[1], q[2])]  # top, bot, left, right
    real_tb, real_lr = [real[0], real[2]], [real[3], real[1]]
    fam = [[], []]
    for k, ((p, r), ok) in enumerate(zip(sides, real_tb + real_lr, strict=True)):
        if ok:
            fam[k // 2].append((p, r, 3.0 * np.linalg.norm(r - p)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    segs = cv2.createLineSegmentDetector().detect(gray)[0]
    if segs is not None:
        mask = np.zeros((h, w), np.uint8)
        cen = q.mean(axis=0)
        cv2.fillPoly(mask, [(cen + 0.94 * (q - cen)).round().astype(np.int32)], 1)
        ref = [[(sides[0][1] - sides[0][0]), (sides[1][1] - sides[1][0])],
               [(sides[2][1] - sides[2][0]), (sides[3][1] - sides[3][0])]]
        for x1, y1, x2, y2 in segs.reshape(-1, 4):
            p, r = np.array([x1, y1], float), np.array([x2, y2], float)
            ln = np.linalg.norm(r - p)
            if ln < 0.04 * min(h, w):
                continue
            mx, my = ((p + r) / 2).astype(int)
            if not (0 <= mx < w and 0 <= my < h and mask[my, mx]):
                continue
            d = (r - p) / ln
            for k in range(2):
                for rv in ref[k]:
                    rv = rv / (np.linalg.norm(rv) + 1e-12)
                    if abs(d[0] * rv[1] - d[1] * rv[0]) < np.sin(np.radians(8)):
                        fam[k].append((p, r, ln))
                        break
                else:
                    continue
                break

    v = [_vanishing_point(fam[0], c, s), _vanishing_point(fam[1], c, s)]
    # last resort: the quad's own sides (frame ones included)
    for k in range(2):
        if v[k] is None and not any(real_tb if k == 0 else real_lr):
            pair = sides[0:2] if k == 0 else sides[2:4]
            v[k] = _vanishing_point([(p, r, 1.0) for p, r in pair], c, s)

    f = 0.75 * s  # ~26 mm-equivalent phone lens
    if v[0] is not None and v[1] is not None and abs(v[0][2]) > 1e-6 and abs(v[1][2]) > 1e-6:
        p1, p2 = v[0][:2] / v[0][2] * s, v[1][:2] / v[1][2] * s
        f2 = -float(np.dot(p1, p2))
        if (0.45 * s) ** 2 < f2 < (2.5 * s) ** 2:
            f = float(np.sqrt(f2))
    K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1.0]])
    Ki = np.linalg.inv(K)

    def direction(vk):
        return np.array([vk[0] * s / f, vk[1] * s / f, vk[2]])

    d = [None if vk is None else direction(vk) for vk in v]
    # one family with a single real edge: it lies in that edge's viewing plane
    # and is orthogonal to the other direction
    for k in range(2):
        if d[k] is None and d[1 - k] is not None:
            fam_real = real_tb if k == 0 else real_lr
            for (p, r), ok in zip(sides[2 * k:2 * k + 2], fam_real, strict=True):
                if ok:
                    ln = np.cross([*p, 1.0], [*r, 1.0])
                    d[k] = np.cross(K.T @ ln, d[1 - k])
                    break
    if d[0] is None or d[1] is None:
        return None
    d = [x / np.linalg.norm(x) for x in d]
    # orient: e1 left->right along the top side, e2 top->bottom
    x0 = Ki @ np.array([*q.mean(axis=0), 1.0])
    for k, ref in ((0, sides[0][1] - sides[0][0]), (1, sides[2][1] - sides[2][0])):
        p = K @ (x0 + 0.01 * d[k])
        p0 = K @ x0
        if np.dot(p[:2] / p[2] - p0[:2] / p0[2], ref) < 0:
            d[k] = -d[k]
    # symmetric orthogonalisation
    m = d[0] + d[1]
    m /= np.linalg.norm(m)
    n = d[0] - d[1]
    n /= np.linalg.norm(n)
    e1, e2 = (m + n) / np.sqrt(2), (m - n) / np.sqrt(2)
    return K, e1, e2


def rectify(img: np.ndarray, quad: np.ndarray | None, paper: str, dpi: int):
    """Metric rectification of the sheet, completed beyond the photo if needed.

    The plane pose gives true proportions even for tilted, partial sheets.
    Sheet extent: each real edge fixes one side; a missing side is rebuilt
    from the paper format (A4: 210 x 297 mm) when the other dimension is
    known, otherwise the visible part is kept. Returns
    (page at ~photo resolution, seen mask, final size at dpi, rectified-by-edges).
    """
    h, w = img.shape[:2]
    if quad is None:
        quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    q = order_corners(quad).astype(np.float64)
    real = [not side_on_frame(q[i], q[(i + 1) % 4], img.shape) for i in range(4)]
    pose = plane_pose(img, q, real)
    if pose is None:
        dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64)
        H = cv2.getPerspectiveTransform(dst.astype(np.float32), q.astype(np.float32))
        H = H.astype(np.float64)
    else:
        K, e1, e2 = pose
        t = np.linalg.inv(K) @ np.array([*q.mean(axis=0), 1.0])
        H = K @ np.column_stack([e1, e2, t])
    Hi = np.linalg.inv(H)

    def to_plane(pts):
        p = np.column_stack([pts, np.ones(len(pts))]) @ Hi.T
        return p[:, :2] / p[:, 2:3]

    qp = to_plane(q)
    rect = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    _, vis = cv2.intersectConvexConvex(q.astype(np.float32), rect)
    vp = to_plane(vis.reshape(-1, 2).astype(np.float64)) if vis is not None else qp
    ext = [vp[:, 1].min(), vp[:, 1].max(), vp[:, 0].min(), vp[:, 0].max()]  # top bot left right
    known = [None] * 4
    if real[0]:
        known[0] = (qp[0, 1] + qp[1, 1]) / 2
    if real[2]:
        known[1] = (qp[2, 1] + qp[3, 1]) / 2
    if real[3]:
        known[2] = (qp[3, 0] + qp[0, 0]) / 2
    if real[1]:
        known[3] = (qp[1, 0] + qp[2, 0]) / 2
    top, bot, left, right = [k if k is not None else e for k, e in zip(known, ext, strict=True)]

    W = right - left if known[2] is not None and known[3] is not None else None
    Hh = bot - top if known[0] is not None and known[1] is not None else None
    mm = None  # millimetres per plane unit
    if paper != "none":
        pw, ph = PAPER_MM[paper]
        if W and Hh:
            # whole sheet seen: keep every measured edge (snapping the extent
            # would crop content); only the mm scale comes from the format
            a, b = (pw, ph) if Hh >= W else (ph, pw)
            mm = float(np.sqrt(a * b / (W * Hh)))
        elif W or Hh:
            span = W or Hh
            other_vis = (bot - top) if W else (right - left)
            # orientation needing the least rebuilt paper that still holds what is seen
            for a, b in sorted(((pw, ph), (ph, pw)), key=lambda ab: ab[1] / ab[0]):
                need = span * b / a
                if other_vis <= need * 1.03:
                    break
            else:
                a, need = (pw if W else ph), other_vis
            mm = a / span
            lo, hi = (0, 1) if W else (2, 3)
            vals = [top, bot, left, right]
            if known[lo] is not None:
                vals[hi] = max(vals[hi], vals[lo] + need)
            elif known[hi] is not None:
                vals[lo] = min(vals[lo], vals[hi] - need)
            top, bot, left, right = vals
        else:
            mm = pw / (right - left) if (bot - top) >= (right - left) else ph / (right - left)

    # density: photo pixels per plane unit at the sheet centre
    cp = np.array([(left + right) / 2, (top + bot) / 2])
    pts = np.array([cp, cp + [1e-3, 0], cp + [0, 1e-3]])
    ip = np.column_stack([pts, np.ones(3)]) @ H.T
    ip = ip[:, :2] / ip[:, 2:3]
    src_den = np.sqrt(np.linalg.norm(ip[1] - ip[0]) * np.linalg.norm(ip[2] - ip[0])) / 1e-3
    out_den = mm / 25.4 * dpi if mm else src_den
    out_size = (int(round((right - left) * out_den)), int(round((bot - top) * out_den)))
    k = min(src_den, out_den)
    nat = (max(8, int(round((right - left) * k))), max(8, int(round((bot - top) * k))))
    A = np.array([[1 / k, 0, left], [0, 1 / k, top], [0, 0, 1.0]])
    M = H @ A
    flags = cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP
    page = cv2.warpPerspective(img, M, nat, flags=flags, borderMode=cv2.BORDER_REPLICATE)
    inq = np.zeros((h, w), np.uint8)
    cv2.fillPoly(inq, [q.round().astype(np.int32)], 255)
    inq = cv2.dilate(inq, np.ones((3, 3), np.uint8))
    seen = cv2.warpPerspective(inq, M, nat, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return page, seen, out_size, sum(real) >= 2
