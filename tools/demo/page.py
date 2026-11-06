"""Draw the demo A4 sheet (fictional notice, self-written text) as a 300 dpi image.

    python tools/demo/page.py OUT.png

The text, names and stamp are invented for the demo: nothing here is a real document.
"""
from __future__ import annotations

import math
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 2480, 3508  # A4 at 300 dpi
MM = 300 / 25.4
FONTS = "/usr/share/fonts/liberation/"
INK = (28, 28, 30)

BODY = [
    "Starting from Monday 3 March the reading room moves to the ground floor of the old "
    "school building in Via dei Mulini, while the upper floor is repaired after last "
    "winter's leak. The lending desk, the children's corner and the two public computers "
    "move with it; the local history archive stays closed until further notice.",
    "Books borrowed before the move can be returned at the new desk or in the box next to "
    "the bakery. Loans due during the closing week (24 February - 1 March) are extended "
    "automatically by fourteen days: there is nothing to renew and no late fee.",
    "We are still looking for volunteers to carry boxes on Saturday 1 March from nine "
    "o'clock. Gloves and coffee are provided; children are welcome if they bring a "
    "grown-up who lifts the heavy ones.",
]

HOURS = [("Day", "Morning", "Afternoon"),
         ("Monday", "closed", "15:00 - 18:30"),
         ("Tuesday", "9:00 - 12:30", "15:00 - 18:30"),
         ("Wednesday", "9:00 - 12:30", "closed"),
         ("Thursday", "9:00 - 12:30", "15:00 - 19:30"),
         ("Friday", "9:00 - 12:30", "15:00 - 18:30"),
         ("Saturday", "9:30 - 12:30", "closed")]


def font(name: str, pt: float) -> ImageFont.FreeTypeFont:
    """Liberation font at `pt` points for 300 dpi."""
    return ImageFont.truetype(FONTS + name, round(pt * 300 / 72))


def stamp(size: int) -> Image.Image:
    """Round red rubber stamp with uneven ink, as an RGBA layer."""
    s = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(s)
    c = size / 2
    d.ellipse([8, 8, size - 8, size - 8], outline=255, width=14)
    d.ellipse([70, 70, size - 70, size - 70], outline=255, width=6)
    f = font("LiberationSans-Bold.ttf", 15)
    text = "VALBRUNA COMMUNITY LIBRARY * "
    for i, ch in enumerate(text):
        a = -math.pi / 2 + 2 * math.pi * i / len(text)
        glyph = Image.new("L", (120, 120), 0)
        ImageDraw.Draw(glyph).text((60, 60), ch, font=f, fill=255, anchor="mm")
        glyph = glyph.rotate(-math.degrees(a) - 90, resample=Image.BICUBIC)
        r = c - 42
        s.paste(255, (int(c + r * math.cos(a)) - 60, int(c + r * math.sin(a)) - 60), glyph)
    ImageDraw.Draw(s).text((c, c), "1 MAR", font=font("LiberationSans-Bold.ttf", 22),
                           fill=255, anchor="mm")
    rng = np.random.default_rng(7)
    ink = np.asarray(s, np.float32) / 255
    blot = np.asarray(Image.fromarray((rng.random(ink.shape) * 255).astype(np.uint8))
                      .filter(ImageFilter.GaussianBlur(6)), np.float32) / 255
    alpha = np.clip(ink * (0.55 + 1.6 * (blot - 0.5)) * 1.0, 0, 1)
    layer = np.zeros((size, size, 4), np.uint8)
    layer[..., :3] = (196, 36, 48)
    layer[..., 3] = (alpha * 255).astype(np.uint8)
    return Image.fromarray(layer, "RGBA").rotate(-12, resample=Image.BICUBIC)


def signature(d: ImageDraw.ImageDraw, x0: float, y0: float) -> None:
    """Blue ballpoint scribble."""
    t = np.linspace(0, 1, 900)
    xs = x0 + 520 * t + 38 * np.sin(t * 31)
    ys = y0 - 60 * np.sin(t * 9.5) * np.exp(-2 * t) + 26 * np.cos(t * 23)
    pts = list(zip(xs.tolist(), ys.tolist(), strict=True))
    d.line(pts, fill=(30, 52, 150), width=7, joint="curve")
    d.line([(x0 + 40, y0 + 70), (x0 + 560, y0 + 52)], fill=(30, 52, 150), width=6)


def draw() -> Image.Image:
    """The full sheet."""
    img = Image.new("RGB", (W, H), (250, 249, 245))
    d = ImageDraw.Draw(img)
    left, right = round(25 * MM), W - round(25 * MM)
    y = round(22 * MM)
    d.text((left, y), "VALBRUNA COMMUNITY LIBRARY", font=font("LiberationSans-Bold.ttf", 20),
           fill=(24, 70, 120))
    y += 100
    d.text((left, y), "Via dei Mulini 4  -  open to everyone, no card needed to read",
           font=font("LiberationSans-Regular.ttf", 10), fill=(90, 90, 90))
    y += 70
    d.line([(left, y), (right, y)], fill=(24, 70, 120), width=6)
    y += 90
    d.text((left, y), "Notice: the reading room is moving",
           font=font("LiberationSerif-Bold.ttf", 17), fill=INK)
    y += 130
    body = font("LiberationSerif-Regular.ttf", 11.5)
    for par in BODY:
        for line in textwrap.wrap(par, 90):
            d.text((left, y), line, font=body, fill=INK)
            y += 66
        y += 40
    y += 20
    d.text((left, y), "Opening hours at the new address", font=font("LiberationSerif-Bold.ttf", 13),
           fill=INK)
    y += 90
    cols = [left, left + 600, left + 1200, right]
    row_h = 78
    cell = font("LiberationSans-Regular.ttf", 11)
    for i, row in enumerate(HOURS):
        if i == 0:
            d.rectangle([cols[0], y, cols[-1], y + row_h], fill=(226, 234, 243))
        for j, txt in enumerate(row):
            d.text((cols[j] + 24, y + row_h / 2), txt, font=cell, fill=INK, anchor="lm")
        y += row_h
    top = y - row_h * len(HOURS)
    for k in range(len(HOURS) + 1):
        d.line([(cols[0], top + k * row_h), (cols[-1], top + k * row_h)], fill=INK, width=3)
    for c in cols:
        d.line([(c, top), (c, y)], fill=INK, width=3)
    y += 110
    d.text((left, y), "Questions: ask at the desk or leave a note in the box by the door.",
           font=body, fill=INK)
    y += 200
    d.text((right - 700, y), "The librarian", font=body, fill=INK)
    signature(d, right - 720, y + 170)
    st = stamp(560)
    img.paste(st, (left + 120, y - 40), st)
    d.text((W / 2, H - round(15 * MM)), "page 1 of 1", font=font("LiberationSans-Regular.ttf", 9),
           fill=(110, 110, 110), anchor="mm")
    return img


if __name__ == "__main__":
    draw().save(sys.argv[1], dpi=(300, 300))
