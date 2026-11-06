"""Synthetic document photos with known geometry."""
from __future__ import annotations

import cv2
import numpy as np

PAGE_W, PAGE_H = 840, 1188  # A4 proportions
PHOTO_W, PHOTO_H = 900, 1200


def page(tone: int = 246, seed: int = 0) -> np.ndarray:
    """A white sheet with lines of black text and a red stamp-like mark."""
    rng = np.random.default_rng(seed)
    img = np.full((PAGE_H, PAGE_W, 3), tone, np.uint8)
    words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"]
    for y in range(120, PAGE_H - 120, 46):
        x = 80
        while x < PAGE_W - 160:
            w = str(rng.choice(words))
            cv2.putText(img, w, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2,
                        cv2.LINE_AA)
            x += 22 * len(w) + 18
    cv2.circle(img, (PAGE_W - 180, PAGE_H - 200), 60, (40, 40, 200), 4, cv2.LINE_AA)
    return img


def background(seed: int = 1) -> np.ndarray:
    """Wood-like texture: brown base, grain stripes and blotchy noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:PHOTO_H, 0:PHOTO_W].astype(np.float32)
    grain = np.sin(xx / 9.0 + 3 * np.sin(yy / 70.0)) * 18
    blotch = cv2.GaussianBlur(rng.normal(0, 40, (PHOTO_H, PHOTO_W)).astype(np.float32),
                              (0, 0), 12)
    v = 110 + grain + blotch
    bg = np.stack([v * 0.55, v * 0.8, v * 1.1], axis=2)
    return np.clip(bg, 0, 255).astype(np.uint8)


def place(photo: np.ndarray, sheet: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Composite `sheet` onto `photo` so its corners land on `quad` (TL, TR, BR, BL)."""
    h, w = sheet.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    H = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    size = (photo.shape[1], photo.shape[0])
    warped = cv2.warpPerspective(sheet, H, size, flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, size, flags=cv2.INTER_LINEAR)
    a = (mask.astype(np.float32) / 255.0)[:, :, None]
    return (warped * a + photo * (1 - a)).astype(np.uint8)


def shading(photo: np.ndarray, seed: int = 2) -> np.ndarray:
    """Soft illumination gradient, slight blur and sensor noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:PHOTO_H, 0:PHOTO_W].astype(np.float32)
    light = 0.85 + 0.15 * (xx / PHOTO_W) * (1 - 0.5 * yy / PHOTO_H)
    out = cv2.GaussianBlur(photo, (0, 0), 0.8).astype(np.float32) * light[:, :, None]
    out += rng.normal(0, 2.0, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def scene(quad, under=None) -> np.ndarray:
    """Photo of a page at `quad`, optionally on top of a duller sheet at `under`."""
    photo = background()
    if under is not None:
        photo = place(photo, page(tone=222, seed=5), np.asarray(under, np.float32))
    return shading(place(photo, page(), np.asarray(quad, np.float32)))


def iou(q1: np.ndarray, q2: np.ndarray, shape=(PHOTO_H, PHOTO_W)) -> float:
    """IoU of two quads restricted to the photo frame."""
    def mask(q):
        m = np.zeros(shape[:2], np.uint8)
        cv2.fillPoly(m, [np.asarray(q).round().astype(np.int32)], 1)
        return m.astype(bool)
    a, b = mask(q1), mask(q2)
    return float((a & b).sum() / (a | b).sum())
