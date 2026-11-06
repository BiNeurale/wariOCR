"""Automatic sheet detection.

Candidate outlines come from contours, paper masks and pairs of straight
lines; each side is scored as a paper/background edge or as a seam between
stacked sheets (band probes, robust to text), plus overlap with a paper mask.
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import hull_to_quad, line_intersection, order_corners, quad_is_valid


def whiteness(img: np.ndarray) -> np.ndarray:
    """Paper likelihood: bright and unsaturated (Lab L minus chroma)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[:, :, 1] - 128, lab[:, :, 2] - 128)
    return np.clip(lab[:, :, 0] - 2.0 * chroma, 0, 255).astype(np.uint8)


#: Detector weights, tuned against hand-drawn outlines (see --corners).
DETECT = {
    "frame": 0.1,       # support of a side lying on the photo border (weak evidence)
    "seam": 0.8,        # support of a paper-on-paper edge (line + small brightness step)
    "seam_dl": 2.0,     # min |ΔL| across a seam
    "w_paper": 0.5,     # weight of the region term (quad vs paper mask IoU)
    "w_area": 0.0,      # bonus for larger quads
    "min_score": 0.4,
}


class Evidence:
    """Per-photo maps shared by all candidate quads."""

    def __init__(self, small: np.ndarray):
        h, w = small.shape[:2]
        self.shape = small.shape
        self.gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        self.white = cv2.GaussianBlur(whiteness(small), (5, 5), 0)
        self.lab = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0).astype(np.float32) / 255.0,
                                cv2.COLOR_BGR2LAB)
        self.d = max(4.0, 0.006 * max(h, w))
        segs = cv2.createLineSegmentDetector().detect(self.gray)[0]
        self.segs = np.zeros((0, 4), np.float32) if segs is None else segs.reshape(-1, 4)
        # straight-edge maps by orientation: a seam between two sheets shows up
        # as a line even when both sides are paper
        self.lines_h = np.zeros((h, w), np.uint8)
        self.lines_v = np.zeros((h, w), np.uint8)
        for x1, y1, x2, y2 in self.segs:
            if np.hypot(x2 - x1, y2 - y1) < 0.03 * min(h, w):
                continue
            m = self.lines_h if abs(x2 - x1) >= abs(y2 - y1) else self.lines_v
            cv2.line(m, (int(x1), int(y1)), (int(x2), int(y2)), 1, 5)

    def _band(self, base: np.ndarray, nv: np.ndarray) -> np.ndarray:
        """Median Lab over a band of probes along `nv` (text strokes, which
        cover less than half of the band, drop out of the median)."""
        hh, ww = self.lab.shape[:2]
        vals = []
        for k in (0.5, 1.0, 1.5, 2.0, 2.5):
            pts = np.clip(base + k * self.d * nv, 0, [ww - 1, hh - 1]).astype(int)
            vals.append(self.lab[pts[:, 1], pts[:, 0]])
        return np.median(np.stack(vals), axis=0)

    def step(self, pin: np.ndarray, pout: np.ndarray, horizontal: bool) -> np.ndarray:
        """Edge evidence per sample (pin/pout: probes inside/outside the side)."""
        mid = (pin + pout) / 2
        nv = (pin - pout) / (np.linalg.norm(pin - pout, axis=1, keepdims=True) + 1e-9)
        ci, co = self._band(mid, nv), self._band(mid, -nv)
        # paper → background: colour step, paper side the more paper-like one
        wi = ci[:, 0] - 2 * np.hypot(ci[:, 1], ci[:, 2])
        wo = co[:, 0] - 2 * np.hypot(co[:, 1], co[:, 2])
        bg = (np.linalg.norm(ci - co, axis=1) > 12) & (wi - wo > 6)
        # paper → paper (stacked sheets): a straight line on the side plus a
        # step in paper tone between the two sheets
        mi = mid.astype(int)
        lm = self.lines_h if horizontal else self.lines_v
        seam = (lm[mi[:, 1], mi[:, 0]] > 0) & (np.abs(ci[:, 0] - co[:, 0]) > DETECT["seam_dl"])
        return np.where(bg, 1.0, np.where(seam, DETECT["seam"], 0.0))


def quad_support(q: np.ndarray, ev: Evidence) -> np.ndarray:
    """Per-side fraction of samples that sit on a sheet edge.

    Samples whose outer probe falls off the photo count DETECT["frame"]; thin
    ruled/table lines get nothing because both probes land on the same paper.
    """
    h, w = ev.shape[:2]
    q = order_corners(q).astype(np.float64)
    center = q.mean(axis=0)
    d = ev.d
    t = np.linspace(0.06, 0.94, 40)[:, None]
    fracs = np.empty(4)
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        p = a + t * (b - a)
        nrm = np.array([-(b - a)[1], (b - a)[0]])
        nrm /= np.linalg.norm(nrm) + 1e-9
        if np.dot(center - (a + b) / 2, nrm) < 0:
            nrm = -nrm
        pin, pout = p + d * nrm, p - d * nrm
        off = (pout[:, 0] < 2) | (pout[:, 0] > w - 3) | (pout[:, 1] < 2) | (pout[:, 1] > h - 3)
        inside = (pin[:, 0] >= 0) & (pin[:, 0] < w) & (pin[:, 1] >= 0) & (pin[:, 1] < h)
        m = ~off & inside
        score = np.where(off, DETECT["frame"], 0.0)
        if m.any():
            horiz = abs(b[0] - a[0]) >= abs(b[1] - a[1])
            score[m] = ev.step(pin[m], pout[m], horiz)
        fracs[i] = score.mean()
    return fracs


def long_lines(ev: Evidence, n_edge: int = 8, n_seam: int = 5) -> tuple[list, list]:
    """Straight lines that look like sheet edges, split into ~horizontal and
    ~vertical, plus the four photo borders.

    Collinear LSD pieces are merged first (a seam between stacked sheets is
    often broken into short bits). Paper/background edges and paper/paper
    seams are ranked separately, so neither can crowd the other out.
    """
    h, w = ev.shape[:2]
    d = ev.d
    c0 = np.array([w / 2.0, h / 2.0])
    segs = [s_ for s_ in ev.segs if np.hypot(s_[2] - s_[0], s_[3] - s_[1]) >= 0.02 * min(h, w)]
    segs.sort(key=lambda s_: -np.hypot(s_[2] - s_[0], s_[3] - s_[1]))
    clusters = []  # [theta, rho, p, u, [segments]]
    for x1, y1, x2, y2 in segs:
        p, r = np.array([x1, y1], float), np.array([x2, y2], float)
        u = (r - p) / np.linalg.norm(r - p)
        nrm = np.array([-u[1], u[0]])
        th = np.degrees(np.arctan2(u[1], u[0])) % 180
        rho = float(np.dot(nrm, (p + r) / 2 - c0))
        for cl in clusters:
            dth = abs(th - cl[0])
            dth = min(dth, 180 - dth)
            n0 = np.array([-cl[3][1], cl[3][0]])
            if dth < 2.0 and abs(float(np.dot(n0, (p + r) / 2 - cl[2]))) < 0.006 * max(h, w):
                cl[4].append((p, r))
                break
        else:
            clusters.append([th, rho, (p + r) / 2, u, [(p, r)]])

    edges, seams = [], []
    for _th, _, p, u, members in clusters:
        total = sum(np.linalg.norm(r_ - p_) for p_, r_ in members)
        if total < 0.08 * min(h, w):
            continue
        nrm = np.array([-u[1], u[0]])
        pts = np.vstack([p_ + np.linspace(0, 1, max(4, int(np.linalg.norm(r_ - p_) / 8)))[:, None]
                         * (r_ - p_) for p_, r_ in members])
        # project onto the cluster line so probes are symmetric about it
        pts = p + ((pts - p) @ u)[:, None] * u
        pa = np.clip(pts + d * nrm, 0, [w - 1, h - 1])
        pb = np.clip(pts - d * nrm, 0, [w - 1, h - 1])
        horiz = abs(u[0]) >= abs(u[1])
        sa_, sb_ = ev.step(pa, pb, horiz), ev.step(pb, pa, horiz)
        st = sa_ if sa_.sum() >= sb_.sum() else sb_
        bg, any_ = (st == 1).mean(), (st > 0).mean()
        if bg >= 0.3:
            edges.append((total * bg, p, u))
        elif any_ >= 0.4:
            seams.append((total * any_, p, u))

    hor, ver = [], []
    for pool, cap in ((edges, n_edge), (seams, n_seam)):
        taken = {"h": 0, "v": 0}
        for _, p, u in sorted(pool, key=lambda c_: -c_[0]):
            ang = np.degrees(np.arctan2(u[1], u[0])) % 180
            key = "h" if (ang < 45 or ang > 135) else "v"
            if taken[key] < cap:
                (hor if key == "h" else ver).append((p, u))
                taken[key] += 1
    hor += [(np.array([0.0, 0.0]), np.array([1.0, 0.0])),
            (np.array([0.0, h - 1.0]), np.array([1.0, 0.0]))]
    ver += [(np.array([0.0, 0.0]), np.array([0.0, 1.0])),
            (np.array([w - 1.0, 0.0]), np.array([0.0, 1.0]))]
    return hor, ver


def candidate_quads(ev: Evidence) -> list[np.ndarray]:
    """Sheet outline hypotheses from contours, paper masks and line pairs."""
    sh = ev.shape
    sa = sh[0] * sh[1]
    gray, white = ev.gray, ev.white
    k = max(3, int(max(sh) / 80))

    quads: list[np.ndarray] = []

    # Method A: edges (auto Canny on luminance and on whiteness)
    contours: list[np.ndarray] = []
    for ch in (gray, white):
        blur = cv2.GaussianBlur(ch, (5, 5), 0)
        med = float(np.median(blur))
        edges = cv2.Canny(blur, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours += sorted(cnts, key=cv2.contourArea, reverse=True)[:8]

    # Method B: paper segmentation at a few thresholds around Otsu
    otsu, _ = cv2.threshold(white, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    for thr in (otsu - 25, otsu - 10, otsu, otsu + 15, otsu + 30):
        mask = (white > thr).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=2)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours += sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
    for c in contours:
        if cv2.contourArea(c) >= 0.12 * sa:
            q = hull_to_quad(c)
            if q is not None:
                quads.append(q)

    # Method C: every pair of ~horizontal x pair of ~vertical edge-like lines
    hor, ver = long_lines(ev)
    for i, (p1, u1) in enumerate(hor):
        for (p2, u2) in hor[i + 1:]:
            for j, (p3, u3) in enumerate(ver):
                for (p4, u4) in ver[j + 1:]:
                    pts = [line_intersection(pa, pa + ua, pb, pb + ub)
                           for (pa, ua), (pb, ub) in
                           (((p1, u1), (p3, u3)), ((p1, u1), (p4, u4)),
                            ((p2, u2), (p4, u4)), ((p2, u2), (p3, u3)))]
                    if all(x is not None for x in pts):
                        quads.append(np.array(pts, dtype=np.float32))

    return [q for q in quads if quad_is_valid(q, sh)]


def paper_mask(ev: Evidence) -> np.ndarray:
    """Coarse paper mask (bright, unsaturated, not inside a colourful pattern)."""
    sh = ev.shape
    otsu, _ = cv2.threshold(ev.white, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # neighbourhood colourfulness: white patches of a patterned cloth are not paper
    local_c = cv2.blur(np.hypot(ev.lab[:, :, 1], ev.lab[:, :, 2]), (max(sh) // 40,) * 2)
    f = 240.0 / max(sh[:2])
    return cv2.resize(((ev.white > otsu) & (local_c < 12)).astype(np.uint8), None, fx=f, fy=f,
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def score_quad(q: np.ndarray, ev: Evidence, pm: np.ndarray) -> float:
    """Blend of per-side edge support and quad/paper-mask IoU; higher is better."""
    fr = quad_support(q, ev)
    f = pm.shape[1] / ev.shape[1]
    qm = np.zeros(pm.shape, np.uint8)
    cv2.fillPoly(qm, [(order_corners(q) * f).round().astype(np.int32)], 1)
    qm = qm.astype(bool)
    iou = (qm & pm).sum() / max(1, (qm | pm).sum())
    wp = DETECT["w_paper"]
    sa = ev.shape[0] * ev.shape[1]
    return float((1 - wp) * (0.6 * fr.mean() + 0.4 * fr.min()) + wp * iou
                 + DETECT["w_area"] * cv2.contourArea(order_corners(q)) / sa)


def detect_document(img: np.ndarray) -> np.ndarray | None:
    """Find the sheet outline in a BGR photo.

    Returns the 4 corners (TL, TR, BR, BL, float32, photo pixels; they may lie
    outside the frame when the sheet is cut off), or None if nothing scores
    above `DETECT["min_score"]`.
    """
    h, w = img.shape[:2]
    scale = min(1.0, 1200.0 / max(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1.0 else img
    ev = Evidence(small)
    pm = paper_mask(ev)
    best, best_score = None, -1.0
    for q in candidate_quads(ev):
        s = score_quad(q, ev, pm)
        if s > best_score:
            best, best_score = q, s
    if best is None or best_score < DETECT["min_score"]:
        return None
    return order_corners(best) / scale
