"""One photo in, one cleaned page image out."""
from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ._util import odd
from .binarize import despeckle, sauvola
from .detect import detect_document
from .enhance import denoise, deskew_angle, enhance, lab_to_bgr, resample, unsharp
from .io import load_image
from .manual import find_annotation, manual_quad
from .orient import apply_rotation, osd_rotation
from .pose import rectify

#: Subdirectory of the output dir that receives the page images.
PAGES_DIR = "pages"

#: Lightness of rebuilt, never-photographed paper: light grey, not white.
MISSING_L = 96.0


@dataclass
class Result:
    """Outcome of `process` for one photo."""

    src: str
    out: str  # page image path, "" on error
    detected: bool  # a sheet outline was found (auto or manual)
    angle: float  # applied deskew, degrees
    error: str = ""
    manual: bool = False  # outline came from a hand-drawn annotation
    missing: float = 0.0  # fraction of the page rebuilt (outside the photo)


def process(path: str, outdir: str, mode: str = "color", dpi: int = 300, paper: str = "a4",
            jpeg_q: int = 92, debug: bool = False, denoise_k: float = 1.0,
            saturation: float = 1.25, whiten: float = 0.5,
            corners: str | None = None) -> Result:
    """Turn the photo at `path` into a page image in `outdir/pages/`.

    mode: "color" (JPEG), "gray" (PNG) or "bw" (1-bit G4 TIFF). `paper` fixes
    the true scale ("none": keep the photo's own scale). `corners` is a
    directory of hand-drawn outlines (see `manual_quad`). With `debug` the
    detected outline is saved to `outdir/debug/<stem>_detect.jpg`.
    Never raises: errors are reported in `Result.error`.
    """
    p = Path(path)
    try:
        img = load_image(p)
        annot = find_annotation(corners, p.stem)
        quad = manual_quad(annot, img.shape) if annot else None
        manual = quad is not None
        if not manual:
            quad = detect_document(img)

        if debug:
            # full resolution: these images are meant to be corrected by hand
            dbg = img.copy()
            if quad is not None:
                cv2.polylines(dbg, [quad.round().astype(np.int32)], True, (0, 0, 255),
                              max(2, img.shape[1] // 400))
            dpath = Path(outdir) / "debug" / f"{p.stem}_detect.jpg"
            dpath.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dpath), dbg, [cv2.IMWRITE_JPEG_QUALITY, 92])

        page, seen, size, by_edges = rectify(img, quad, paper, dpi)
        miss = seen < 128
        if miss.any() and (~miss).any():
            # neutral paper stand-in so the shading estimate isn't fooled
            page[miss] = np.median(page[~miss].reshape(-1, 3), axis=0)
        page = denoise(page, denoise_k)
        lab = enhance(page, saturation if mode == "color" else 0.0, whiten)
        ang = deskew_angle(np.clip(lab[:, :, 0] * 2.55, 0, 255).astype(np.uint8))
        # edges already fix the geometry: only small residual skew of the print
        if by_edges and abs(ang) > 1.5:
            ang = 0.0
        lab = resample(lab, size, ang)
        seen_f = cv2.resize(seen, size, interpolation=cv2.INTER_LINEAR) if ang == 0 else \
            resample(cv2.merge([seen.astype(np.float32)] * 3), size, ang)[:, :, 0]

        # luminance-only sharpening: crisp strokes without colour fringes
        up = size[0] / page.shape[1]
        L = unsharp(lab[:, :, 0], sigma=max(0.8, 0.6 * up), amount=0.8)
        lab[:, :, 0] = np.clip(L, 0, 100)
        wm = np.clip(seen_f.astype(np.float32) / 255.0, 0, 1)
        wm = cv2.GaussianBlur(wm, (0, 0), max(1.0, dpi / 150))[:, :, None]
        lab = lab * wm + np.array([MISSING_L, 0, 0], np.float32) * (1 - wm)
        # 2 mm white margin: removes background leftovers along the edges
        m = max(1, int(round(dpi / 25.4 * 2.0)))
        for sl in (np.s_[:m], np.s_[-m:], np.s_[:, :m], np.s_[:, -m:]):
            lab[sl] = (100.0, 0.0, 0.0)
        lab = lab.astype(np.float32)

        gray = np.clip(lab[:, :, 0] * 2.55 + 0.5, 0, 255).astype(np.uint8)
        _, probe = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        rot = osd_rotation(probe)

        pages_dir = Path(outdir) / PAGES_DIR
        pages_dir.mkdir(parents=True, exist_ok=True)

        if mode == "bw":
            bw = sauvola(gray, odd(dpi / 300.0 * 41))
            bw = apply_rotation(despeckle(bw, dpi), rot)
            out = pages_dir / f"{p.stem}.tif"
            Image.fromarray(bw).convert("1").save(
                out, compression="group4", dpi=(dpi, dpi))
        elif mode == "gray":
            out = pages_dir / f"{p.stem}.png"
            Image.fromarray(apply_rotation(gray, rot)).save(out, dpi=(dpi, dpi), optimize=True)
        else:
            c = apply_rotation(lab_to_bgr(lab), rot)
            out = pages_dir / f"{p.stem}.jpg"
            Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)).save(
                out, quality=jpeg_q, subsampling=0, dpi=(dpi, dpi))

        return Result(path, str(out), quad is not None, ang, manual=manual,
                      missing=float(1 - wm.mean()))
    except Exception as e:
        return Result(path, "", False, 0.0, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
