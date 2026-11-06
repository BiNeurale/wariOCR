"""Command line: a folder of photos to one PDF per page (plus an optional merged PDF).

Fixing a wrong outline: run with --debug, open debug/NAME_detect.jpg in an
image editor, draw the sheet outline in pure blue (0,0,255) and save it into
the --corners directory. A side may be left undrawn where the sheet runs out
of the photo.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .geometry import PAPER_MM
from .io import EXTS, HEIF_OK, natural_key
from .pdf import make_pdf, ocrmypdf_command, run_ocr
from .pipeline import PAGES_DIR, Result, process

#: Default corner-annotation directories inside the input dir, in priority order.
CORNER_DIRS = ("corners", "a-mano")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the `wariocr` command (Italian legacy flags hidden)."""
    ap = argparse.ArgumentParser(prog="wariocr",
                                 description="Document photos -> true-scale PDF with OCR")
    ap.add_argument("input", type=Path, help="folder with the photos")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output folder (default: INPUT/scans)")
    ap.add_argument("--mode", choices=["color", "gray", "bw"], default="color",
                    help="color = faithful colours (default), gray = greyscale, "
                         "bw = 1-bit text only, smallest files")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--paper", choices=[*PAPER_MM, "none"], default="a4")
    ap.add_argument("--merge", metavar="NAME.pdf",
                    help="also merge all pages into one PDF")
    ap.add_argument("--order", type=Path, default=None,
                    help="text file, one photo name per line (extension optional): "
                         "page order of the merged PDF; unlisted photos go last. "
                         "'#' starts a comment")
    ap.add_argument("--ordine", dest="order", type=Path, default=argparse.SUPPRESS,
                    help=argparse.SUPPRESS)
    ap.add_argument("--lang", default="ita+eng", help="Tesseract languages")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--pdfa", action="store_true", help="PDF/A output (archiving)")
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--denoise", type=float, default=1.0,
                    help="JPEG noise removal strength (0 = off, 1 = messaging-app "
                         "photos, 0.5 = original camera photos)")
    ap.add_argument("--saturation", type=float, default=1.25,
                    help="saturation of coloured inks (1 = unchanged)")
    ap.add_argument("--whiten", type=float, default=0.5,
                    help="0..1: how hard paper is pushed to white (low keeps more "
                         "faint detail)")
    ap.add_argument("--sbiancamento", dest="whiten", type=float, default=argparse.SUPPRESS,
                    help=argparse.SUPPRESS)
    ap.add_argument("--corners", type=Path, default=None,
                    help="folder of hand-corrected outlines: --debug images "
                         "(NAME_detect.jpg) with the sheet outline in pure blue; "
                         "default INPUT/corners (or INPUT/a-mano) if present")
    ap.add_argument("--bordi", dest="corners", type=Path, default=argparse.SUPPRESS,
                    help=argparse.SUPPRESS)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--debug", action="store_true",
                    help="save the photos with the detected outline drawn")
    return ap


def _order_rank(order_file: Path) -> dict[str, int]:
    wanted = [ln.split("#")[0].strip() for ln in order_file.read_text().splitlines()]
    return {w: i for i, w in enumerate(x for x in wanted if x)}


def main(argv: list[str] | None = None) -> None:
    """Entry point of the `wariocr` command."""
    a = build_parser().parse_args(argv)

    if not a.input.is_dir():
        sys.exit(f"Folder not found: {a.input}")
    outdir = a.output or (a.input / "scans")
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted((f for f in a.input.iterdir()
                    if f.is_file() and f.suffix.lower() in EXTS), key=natural_key)
    if not files:
        sys.exit("No images found." +
                 ("" if HEIF_OK else " (HEIC unsupported: pip install 'wariocr[heif]')"))

    do_ocr = not a.no_ocr and ocrmypdf_command() is not None
    if not a.no_ocr and not do_ocr:
        print("! ocrmypdf not found: writing PDFs without OCR.")

    print(f"{len(files)} photos | mode={a.mode} dpi={a.dpi} paper={a.paper} "
          f"OCR={'yes (' + a.lang + ')' if do_ocr else 'no'} | {a.jobs} processes")

    corners = a.corners
    if corners is None:
        corners = next((a.input / d for d in CORNER_DIRS if (a.input / d).is_dir()), None)
    elif not corners.is_dir():
        sys.exit(f"Corners folder not found: {corners}")

    results: dict[str, Result] = {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(process, str(f), str(outdir), a.mode, a.dpi, a.paper,
                          a.jpeg_quality, a.debug, a.denoise, a.saturation,
                          a.whiten, str(corners) if corners else None): f
                for f in files}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results[r.src] = r
            flag = "MANUAL " if r.manual else ("OK " if r.detected else "EDGE? ")
            if r.error:
                flag = "ERROR "
            print(f"[{i:>4}/{len(files)}] {flag}{Path(r.src).name}"
                  f"{'  ' + r.error if r.error else f'  deskew {r.angle:+.2f}°'}"
                  f"{f'  rebuilt {r.missing:.0%}' if r.missing > 0.01 else ''}")

    ordered = [results[str(f)] for f in files if results[str(f)].out]
    if a.order:
        rank = _order_rank(a.order)

        def pos(r: Result) -> int:
            p_ = Path(r.src)
            return rank.get(p_.name, rank.get(p_.stem, len(rank)))

        ordered.sort(key=pos)  # stable: unlisted keep the natural order
    tmp = outdir / ".tmp"
    tmp.mkdir(exist_ok=True)

    def finalize(images: list[str], name: str) -> None:
        raw = tmp / name
        make_pdf(images, raw, a.paper)
        final = outdir / name
        if do_ocr and run_ocr(raw, final, a.lang, a.pdfa, a.jobs):
            raw.unlink()
        else:
            shutil.move(raw, final)

    for r in ordered:
        finalize([r.out], Path(r.src).stem + ".pdf")
    print(f"\n-> {len(ordered)} PDF in {outdir}")
    if a.merge:
        name = a.merge if a.merge.lower().endswith(".pdf") else a.merge + ".pdf"
        finalize([r.out for r in ordered], name)
        print(f"-> {outdir / name}")
    shutil.rmtree(tmp, ignore_errors=True)

    undetected = [Path(r.src).name for r in results.values()
                  if not r.detected and not r.error]
    errors = [Path(r.src).name for r in results.values() if r.error]
    if undetected:
        print(f"\nSheet outline NOT found ({len(undetected)}): processed as the full "
              f"frame, check them:\n  " + "\n  ".join(undetected))
    if errors:
        print(f"\nErrors ({len(errors)}):\n  " + "\n  ".join(errors))
    print(f"\nPage images (for touch-ups) in: {outdir / PAGES_DIR}")
