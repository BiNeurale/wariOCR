#!/usr/bin/env python3
"""Score automatic sheet detection against hand-drawn outlines.

For every `<stem>_detect.*` annotation (pure blue outline, see `wariocr.manual_quad`)
the photo `<stem>.<ext>` is looked up in PHOTO_DIR, detected, and compared:
IoU of the two quads and the worst side-midpoint distance (% of the long side).

    python tools/eval_manual_corners.py PHOTO_DIR ANNOT_DIR [--jobs N]
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import wariocr
from wariocr.io import EXTS

SUFFIX = "_detect"


def find_photo(photo_dir: Path, stem: str) -> Path | None:
    """The photo named `stem` with any supported extension."""
    for f in photo_dir.glob(f"{stem}.*"):
        if f.suffix.lower() in EXTS:
            return f
    return None


def quad_mask(q: np.ndarray, shape) -> np.ndarray:
    """Boolean mask of a quad clipped to the image."""
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [wariocr.order_corners(q).round().astype(np.int32)], 1)
    return m.astype(bool)


def evaluate(pair: tuple[Path, Path]) -> tuple[str, float, float | None] | None:
    """(photo name, IoU, side error %) for one photo/annotation pair; None if unannotated."""
    src, annot = pair
    img = wariocr.load_image(src)
    gt = wariocr.manual_quad(annot, img.shape)
    if gt is None:
        return None
    q = wariocr.detect_document(img)
    if q is None:
        return src.name, 0.0, None
    a, b = quad_mask(gt, img.shape), quad_mask(q, img.shape)
    iou = float((a & b).sum() / (a | b).sum())
    g, d = wariocr.order_corners(gt), wariocr.order_corners(q)
    mid = [np.linalg.norm((g[i] + g[(i + 1) % 4]) / 2 - (d[i] + d[(i + 1) % 4]) / 2)
           for i in range(4)]
    return src.name, iou, float(max(mid) / max(img.shape[:2]) * 100)


def main() -> None:
    """Print per-photo scores and the summary line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photo_dir", type=Path)
    ap.add_argument("annot_dir", type=Path)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()

    pairs = []
    for annot in sorted(a.annot_dir.iterdir()):
        if annot.suffix.lower() in EXTS and annot.stem.endswith(SUFFIX):
            src = find_photo(a.photo_dir, annot.stem[:-len(SUFFIX)])
            if src is None:
                print(f"! no photo for {annot.name}", file=sys.stderr)
            else:
                pairs.append((src, annot))
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = [r for r in ex.map(evaluate, pairs) if r]
    if not res:
        sys.exit("No usable annotations.")
    for name, iou, err in res:
        side = "-" if err is None else f"{err:.1f}%"
        print(f"{name:40s} IoU {iou:.3f}  side-err {side}")
    ious = [r[1] for r in res]
    print(f"MEAN IoU {np.mean(ious):.3f}  min {np.min(ious):.3f}  "
          f"ok(>=0.97) {sum(i >= 0.97 for i in ious)}/{len(ious)}")


if __name__ == "__main__":
    main()
