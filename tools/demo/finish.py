"""Give a clean render the look of a phone JPEG: sensor noise, vignetting, compression.

    python tools/demo/finish.py RENDER.png OUT.jpg [SEED]
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image


def finish(src: str, dst: str, seed: int = 0) -> None:
    """Write `src` as a noisy, slightly vignetted JPEG (quality 85, like a messaging app)."""
    rng = np.random.default_rng(seed)
    im = np.asarray(Image.open(src).convert("RGB"), np.float32)
    h, w = im.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    im *= (1 - 0.12 * r2 / 2)[..., None]
    # shot noise grows with brightness, plus a little chroma noise
    im += rng.normal(0, 1, im.shape[:2])[..., None] * (1.5 + 0.012 * im)
    im += rng.normal(0, 1.2, im.shape)
    Image.fromarray(np.clip(im, 0, 255).astype(np.uint8)).save(dst, quality=85)


if __name__ == "__main__":
    finish(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
