"""Denoising, illumination flattening and colour in Lab, final resample, deskew."""
from __future__ import annotations

import cv2
import numpy as np

from ._util import odd, smoothstep


def denoise(img: np.ndarray, strength: float) -> np.ndarray:
    """Remove JPEG blocking/ringing; chroma (4:2:0, coarser) gets more than luma."""
    if strength <= 0:
        return img
    return cv2.fastNlMeansDenoisingColored(img, None, h=3.0 * strength,
                                           hColor=6.0 * strength,
                                           templateWindowSize=7, searchWindowSize=21)


def background(ch: np.ndarray) -> np.ndarray:
    """Paper (background) estimate of one channel: closing + median at 1/4 scale."""
    h, w = ch.shape
    f = 0.25
    s = cv2.resize(ch, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    k = odd(max(s.shape) / 60)
    s = cv2.dilate(s, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    s = cv2.medianBlur(s, min(odd(max(s.shape) / 25), 255))
    s = cv2.GaussianBlur(s, (0, 0), 2)
    return cv2.resize(s, (w, h), interpolation=cv2.INTER_CUBIC)

def enhance(img: np.ndarray, saturation: float, whiten: float = 0.5) -> np.ndarray:
    """Flat, white, neutral paper with faithful ink colours.

    Returns float32 Lab (L 0..100, a/b signed). Luminance is divided by the
    paper estimate (shadows, vignetting), the paper's own colour cast is
    subtracted from a/b (white balance that leaves inks alone), then levels
    push paper to pure white and chroma below the JPEG noise floor is dropped.
    """
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)

    bg = background(np.clip(L * 2.55, 0, 255).astype(np.uint8)).astype(np.float32) / 2.55
    Ln = L / np.maximum(bg, 5.0)

    # paper cast: one global estimate (median a/b of clean paper). A local one
    # would learn large tinted areas (map fills, highlighter) as "paper" and erase them.
    chroma = np.hypot(a, b)
    paper = (Ln > 0.88) & (chroma < 12)
    if paper.sum() > 1000:
        a = a - float(np.median(a[paper]))
        b = b - float(np.median(b[paper]))

    # levels: white point = paper level minus a few noise sigmas. `whiten`
    # (0..1) sets how many: low keeps faint pencil and light fills visible.
    pv = Ln[paper]
    if pv.size > 1000:
        med = float(np.median(pv))
        sig = 1.4826 * float(np.median(np.abs(pv - med)))
        white = med - (1.0 + 3.0 * whiten) * sig
    else:
        white = 0.95
    white = float(np.clip(white, 0.85, 0.99))
    black = float(min(np.percentile(Ln, 0.3), 0.35)) * whiten
    x = np.clip((Ln - black) / (white - black), 0, 1)
    L_ink = 100.0 * x ** (1.0 + 0.3 * whiten)  # gamma > 1: denser black/grey strokes
    # coloured inks (red pencil, stamps) would turn near-black under that
    # curve: they only get the white point, keeping their real lightness
    L_col = 100.0 * np.clip(Ln / white, 0, 1)

    # chroma: smooth JPEG colour blocks, gate the noise floor, mild boost
    a = cv2.GaussianBlur(a, (0, 0), 1.0)
    b = cv2.GaussianBlur(b, (0, 0), 1.0)
    C = np.hypot(a, b)
    wcol = smoothstep(8.0, 22.0, C)
    Lo = L_ink * (1 - wcol) + L_col * wcol
    gate = smoothstep(3.0, 8.0, C) * saturation
    # near-white pixels keep only strong colours (highlighter yes, paper tint no)
    gate *= np.where(Lo > 97, smoothstep(4.0, 9.0, C), 1.0)
    return cv2.merge([Lo, a * gate, b * gate])


def lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    """Float Lab (L 0..100) to BGR uint8."""
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return np.clip(bgr * 255.0 + 0.5, 0, 255).astype(np.uint8)


def unsharp(img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    """Unsharp mask: img + amount * (img - blur(img, sigma))."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def resample(lab: np.ndarray, size: tuple[int, int], angle: float) -> np.ndarray:
    """Single final resample: deskew rotation + scale to the output DPI."""
    h, w = lab.shape[:2]
    out_w, out_h = size
    sx, sy = out_w / w, out_h / h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    M = (S @ np.vstack([M, [0, 0, 1]]))[:2]
    interp = cv2.INTER_LANCZOS4 if sx >= 1 else cv2.INTER_AREA
    return cv2.warpAffine(lab, M, (out_w, out_h), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(100.0, 0.0, 0.0))


def deskew_angle(gray: np.ndarray) -> float:
    """Residual skew in degrees (+-10) maximising the row-projection variance of ink."""
    s = 1000.0 / max(gray.shape)
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    _, ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if ink.mean() < 1.0:  # nearly blank page
        return 0.0
    h, w = ink.shape
    c = (w / 2, h / 2)

    def score(a: float) -> float:
        M = cv2.getRotationMatrix2D(c, a, 1.0)
        r = cv2.warpAffine(ink, M, (w, h), flags=cv2.INTER_NEAREST)
        return float(np.var(r.sum(axis=1, dtype=np.float64)))

    angles = np.arange(-10, 10.01, 0.5)
    best = max(angles, key=score)
    fine = np.arange(best - 0.5, best + 0.51, 0.05)
    best = max(fine, key=score)
    return float(best) if abs(best) > 0.1 else 0.0
