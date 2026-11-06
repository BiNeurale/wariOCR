"""Build the README media from a real run of the `wariocr` command.

    python tools/demo/media.py [--photos examples] [--out docs/media] [--work build/demo]

Runs the CLI on the demo photos (with --debug), then assembles, from the files it wrote
and from the same library calls the pipeline makes:
  hero.jpg              photo -> final page
  pipeline.gif          photo, detected outline, rectified sheet, final page
  terminal.gif          the command's real output, replayed with its real timing
  social-preview.png    1280x640 card for the GitHub repository
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wariocr import detect_document, load_image
from wariocr.io import EXTS, natural_key
from wariocr.pose import rectify

FONT = "/usr/share/fonts/TTF/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/TTF/DejaVuSansMono.ttf"
BG = (24, 26, 30)
FG = (226, 228, 232)
ACCENT = (110, 190, 255)


def font(path: str, px: int) -> ImageFont.FreeTypeFont:
    """TrueType font at `px` pixels."""
    return ImageFont.truetype(path, px)


def rgb(bgr: np.ndarray) -> Image.Image:
    """OpenCV BGR array to a PIL image."""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def fit(im: Image.Image, w: int, h: int) -> Image.Image:
    """Scale `im` to fit inside w x h, keeping its aspect."""
    s = min(w / im.width, h / im.height)
    return im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))),
                     Image.LANCZOS)


def run_cli(photos: Path, scans: Path, args: list[str]) -> tuple[list, float]:
    """Run `wariocr` and return its stdout lines with arrival times, and the total time."""
    cmd = [sys.executable, "-u", "-m", "wariocr", str(photos), "-o", str(scans), *args]
    t0 = time.perf_counter()
    lines = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as p:
        assert p.stdout is not None
        for ln in p.stdout:
            lines.append((time.perf_counter() - t0, ln.rstrip("\n")))
    if p.returncode:
        sys.exit(f"wariocr failed ({p.returncode})")
    return lines, time.perf_counter() - t0


def stages(photo: Path, scans: Path) -> list[tuple[str, Image.Image]]:
    """The four pipeline stages of one photo, from real program output."""
    img = load_image(photo)
    quad = detect_document(img)
    flat = rectify(img, quad, "a4", 300)[0]
    return [
        ("the phone photo", rgb(img)),
        ("sheet outline found", Image.open(scans / "debug" / f"{photo.stem}_detect.jpg")),
        ("pose, perspective and curl undone", rgb(flat)),
        ("shading removed, true A4 at 300 dpi", Image.open(scans / "pages" / f"{photo.stem}.jpg")),
    ]


def card(im: Image.Image, caption: str, step: int, n: int, w: int, h: int) -> Image.Image:
    """One GIF frame: the image on a dark canvas with a caption bar."""
    bar = 54
    c = Image.new("RGB", (w, h), BG)
    t = fit(im.convert("RGB"), w - 24, h - bar - 24)
    c.paste(t, ((w - t.width) // 2, 12 + (h - bar - 24 - t.height) // 2))
    d = ImageDraw.Draw(c)
    d.text((16, h - bar / 2), f"{step}/{n}", font=font(FONT_B, 20), fill=ACCENT, anchor="lm")
    d.text((64, h - bar / 2), caption, font=font(FONT, 20), fill=FG, anchor="lm")
    return c


def pipeline_gif(photos: list[Path], scans: Path, out: Path, w: int = 520,
                 h: int = 700) -> None:
    """Crossfading stage-by-stage animation over all demo photos."""
    frames, durs = [], []
    for ph in photos:
        st = stages(ph, scans)
        cards = [card(im, cap, i + 1, len(st), w, h) for i, (cap, im) in enumerate(st)]
        for i, cd in enumerate(cards):
            frames.append(cd)
            durs.append(2600 if i == len(cards) - 1 else 1500)
            nxt = cards[i + 1] if i + 1 < len(cards) else None
            if nxt is not None:
                for k in range(1, 4):
                    frames.append(Image.blend(cd, nxt, k / 4))
                    durs.append(80)
    pal = [f.quantize(colors=200, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
           for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durs, loop=0,
                optimize=True, disposal=1)


def terminal_gif(cmd: str, lines: list, total: float, out: Path, speed: float) -> None:
    """Replay the captured output at `speed` x real time, in a terminal-like frame."""
    f = font(MONO, 17)
    lh, pad, w = 24, 18, 980
    shown = [f"$ {cmd}", *(ln for _, ln in lines)]
    h = pad * 2 + 40 + lh * len(shown)
    title = "wariocr - real run" + (f" (played {speed:g}x faster)" if speed != 1 else "")

    def frame(n: int, clock: float) -> Image.Image:
        im = Image.new("RGB", (w, h), (16, 17, 20))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, w, 32], fill=(40, 42, 48))
        for i, col in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
            d.ellipse([14 + 22 * i, 10, 26 + 22 * i, 22], fill=col)
        d.text((w / 2, 16), title, font=font(FONT, 15), fill=(170, 172, 178), anchor="mm")
        d.text((w - 14, 16), f"{clock:5.1f} s", font=font(MONO, 15), fill=(170, 172, 178),
               anchor="rm")
        for i, ln in enumerate(shown[:n]):
            col = ACCENT if i == 0 else FG
            if ln.lstrip().startswith("[") and " OK " in ln:
                col = (150, 220, 140)
            d.text((pad, pad + 32 + i * lh), ln, font=f, fill=col)
        return im

    # the prompt is held 0.9 s before the run starts; then a frame per output line and
    # clock ticks every `tick` seconds of real time in between
    tick = 0.5
    frames, durs = [frame(1, 0.0)], [900]
    clock = 0.0
    for i, (t, _) in enumerate(lines):
        while clock + tick < t:
            clock += tick
            frames.append(frame(i + 1, clock))
            durs.append(round(tick * 1000 / speed))
        durs[-1] += round((t - clock) * 1000 / speed)
        frames.append(frame(i + 2, t))
        durs.append(0)
        clock = t
    frames[-1] = frame(len(shown), total)
    durs[-1] = 4000
    durs = [max(20, d) for d in durs]  # browsers stretch shorter GIF delays to 100 ms
    pal = [fr.quantize(colors=48, dither=Image.Dither.NONE) for fr in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durs, loop=0,
                optimize=True)


def arrow(d: ImageDraw.ImageDraw, x: int, y: int, size: int, col: tuple) -> None:
    """Right-pointing arrow centred on (x, y)."""
    s = size
    d.line([(x - s, y), (x + s * 0.6, y)], fill=col, width=max(3, s // 6))
    d.polygon([(x + s, y), (x + s * 0.4, y - s * 0.5), (x + s * 0.4, y + s * 0.5)], fill=col)


def before_after(before: Image.Image, after: Image.Image, h: int, gap: int,
                 bg: tuple) -> Image.Image:
    """Photo and page side by side at height h, with an arrow between."""
    b = fit(before.convert("RGB"), 10_000, h)
    a = fit(after.convert("RGB"), 10_000, h)
    w = b.width + a.width + gap * 3 + 2 * gap
    c = Image.new("RGB", (w, h + 2 * gap), bg)
    c.paste(b, (gap, gap))
    c.paste(a, (w - gap - a.width, gap))
    # thin frame so the white page stands out from a light background
    d = ImageDraw.Draw(c)
    d.rectangle([w - gap - a.width - 1, gap - 1, w - gap, gap + h], outline=(150, 150, 150))
    arrow(d, b.width + gap + (gap * 3 + 2 * gap) // 2, gap + h // 2, gap, (90, 96, 110))
    return c


def social(before: Image.Image, after: Image.Image, out: Path) -> None:
    """1280x640 repository card: name, one line, before -> after."""
    c = Image.new("RGB", (1280, 640), BG)
    pair = before_after(before, after, 520, 40, BG)
    pair = fit(pair, 720, 600)
    c.paste(pair, (1280 - pair.width - 20, (640 - pair.height) // 2))
    d = ImageDraw.Draw(c)
    d.text((60, 230), "wariOCR", font=font(FONT_B, 76), fill=(255, 255, 255))
    for i, ln in enumerate(["Phone photos of documents", "to true-scale, clean PDFs",
                            "with OCR."]):
        d.text((62, 340 + i * 44), ln, font=font(FONT, 32), fill=FG)
    c.save(out, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--photos", type=Path, default=Path("examples"))
    ap.add_argument("--out", type=Path, default=Path("docs/media"))
    ap.add_argument("--work", type=Path, default=Path("build/demo"))
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed of the terminal recording")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    scans = a.work / "scans"
    photos = sorted((p for p in a.photos.iterdir() if p.suffix.lower() in EXTS),
                    key=natural_key)

    cli_args = ["--debug", "--lang", "eng", "--merge", "examples.pdf"]
    lines, total = run_cli(a.photos, scans, cli_args)
    (a.work / "run.json").write_text(json.dumps({"seconds": total, "lines": lines}, indent=1))
    shown_cmd = f"wariocr {a.photos} -o {scans} {' '.join(cli_args)}"
    terminal_gif(shown_cmd, lines, total, a.out / "terminal.gif", a.speed)
    pipeline_gif(photos, scans, a.out / "pipeline.gif")

    first = photos[0]
    final = Image.open(scans / "pages" / f"{first.stem}.jpg")
    hero = before_after(Image.open(first), final, 900, 48, (238, 239, 242))
    hero.save(a.out / "hero.jpg", quality=86, optimize=True, progressive=True)
    social(Image.open(first), final, a.out / "social-preview.png")
    print(f"{len(photos)} photos, CLI {total:.1f} s; media in {a.out}")


if __name__ == "__main__":
    main()
