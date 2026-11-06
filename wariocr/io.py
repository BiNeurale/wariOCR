"""Image input: supported formats, EXIF-aware loading, file ordering."""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_OK = True
except ImportError:
    HEIF_OK = False

#: Lower-case suffixes accepted as input photos.
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
if HEIF_OK:
    EXTS |= {".heic", ".heif"}


def load_image(path: Path | str) -> np.ndarray:
    """Load an image as BGR uint8, applying its EXIF orientation."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        arr = np.asarray(im)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def natural_key(p: Path) -> list:
    """Sort key that orders embedded numbers numerically ("p2" < "p10")."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]
