"""PDF assembly at true scale (img2pdf, no recompression) and OCR (ocrmypdf)."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import img2pdf

from .geometry import PAPER_MM


def make_pdf(images: list[str], dst: Path, paper: str) -> None:
    """Write one page per image to `dst`.

    With a paper format every page is that sheet (orientation following the
    image), the scan centred at true scale and shrunk only if larger, so it
    prints 1:1. With "none" the page is the image at its own DPI.
    """
    layout = None
    if paper != "none":
        pw, ph = PAPER_MM[paper]
        layout = img2pdf.get_layout_fun((img2pdf.mm_to_pt(pw), img2pdf.mm_to_pt(ph)),
                                        fit=img2pdf.FitMode.shrink, auto_orient=True)
    with open(dst, "wb") as f:
        f.write(img2pdf.convert(images, layout_fun=layout))


def ocrmypdf_command() -> list[str] | None:
    """How to invoke ocrmypdf: this interpreter's module first, else the one on PATH."""
    if importlib.util.find_spec("ocrmypdf") is not None:
        return [sys.executable, "-m", "ocrmypdf"]
    exe = shutil.which("ocrmypdf")
    return [exe] if exe else None


def run_ocr(src: Path, dst: Path, lang: str, pdfa: bool, jobs: int) -> bool:
    """Add a text layer to `src` into `dst`; False (with a message) on failure."""
    base = ocrmypdf_command()
    if base is None:
        print(f"  ! OCR failed on {src.name}: ocrmypdf not found")
        return False
    # --optimize 0: keep our JPEGs byte-identical (no recompression);
    # no --rotate-pages: orientation is already handled, and OSD flips handwriting.
    cmd = [
        *base, "-l", lang, "--optimize", "0",
        "--output-type", "pdfa" if pdfa else "pdf", "--jobs", str(jobs),
        "--quiet", str(src), str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ! OCR failed on {src.name}: {r.stderr.strip()[:300]}")
        return False
    return True
