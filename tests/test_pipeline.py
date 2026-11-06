import re
import zlib

import cv2
import numpy as np
import pytest
import synth
from PIL import Image

import wariocr.pipeline
from wariocr import process
from wariocr.cli import build_parser, main
from wariocr.pdf import make_pdf

DPI = 150
A4_PX = (round(210 / 25.4 * DPI), round(297 / 25.4 * DPI))
A4_PT = (210 / 25.4 * 72, 297 / 25.4 * 72)
QUAD = [[120, 110], [800, 130], [790, 1090], [110, 1080]]


@pytest.fixture(scope="module")
def photo(tmp_path_factory):
    p = tmp_path_factory.mktemp("in") / "sheet.png"
    cv2.imwrite(str(p), synth.scene(QUAD))
    return p


@pytest.fixture(autouse=True)
def no_osd(monkeypatch):
    # keep results independent of an installed tesseract
    monkeypatch.setattr(wariocr.pipeline, "osd_rotation", lambda gray: 0)


def run(photo, tmp_path, mode):
    r = process(str(photo), str(tmp_path), mode=mode, dpi=DPI)
    assert not r.error, r.error
    assert r.detected and not r.manual
    return r


def test_color_page_is_a4_at_dpi(photo, tmp_path):
    r = run(photo, tmp_path, "color")
    with Image.open(r.out) as im:
        assert im.mode == "RGB"
        # scale comes from the sheet area, aspect from the measured edges
        assert (im.width, im.height) == pytest.approx(A4_PX, rel=0.01)
        assert round(im.info["dpi"][0]) == DPI


def test_pdf_page_is_true_scale(photo, tmp_path):
    r = run(photo, tmp_path, "color")
    pdf = tmp_path / "out.pdf"
    make_pdf([r.out], pdf, "a4")
    data = pdf.read_bytes()
    box = [float(v) for v in re.search(rb"/MediaBox \[([\d. ]+)\]", data).group(1).split()]
    assert box[2:] == pytest.approx(A4_PT, abs=0.1)
    # drawn at its own DPI (1:1 print), shrunk only if it overflows the sheet
    content = b"".join(zlib.decompressobj().decompress(m) for m in
                       re.findall(rb"/FlateDecode.*?stream\r?\n(.*?)endstream", data, re.S)
                       if m[:1] == b"x" and len(m) < 200)
    w_pt, h_pt = (float(v) for v in re.search(rb"([\d.]+) 0 0 ([\d.]+) [\d.]+ [\d.]+ cm",
                                                content).groups())
    with Image.open(r.out) as im:
        nat = (im.width / DPI * 72, im.height / DPI * 72)
    k = min(1.0, A4_PT[0] / nat[0], A4_PT[1] / nat[1])
    assert k > 0.99
    assert (w_pt, h_pt) == pytest.approx((nat[0] * k, nat[1] * k), abs=0.05)


def test_bw_output_is_one_bit(photo, tmp_path):
    r = run(photo, tmp_path, "bw")
    with Image.open(r.out) as im:
        assert r.out.endswith(".tif")
        assert im.mode == "1"
        assert im.info.get("compression") == "group4"
        ink = 1 - np.asarray(im, dtype=np.float32).mean()
    assert 0.005 < ink < 0.3  # text survived, page not black


def test_gray_output(photo, tmp_path):
    r = run(photo, tmp_path, "gray")
    with Image.open(r.out) as im:
        assert im.mode == "L"


def test_cli_italian_aliases():
    a = build_parser().parse_args(["x", "--bordi", "b", "--sbiancamento", "0.2",
                                   "--ordine", "o.txt"])
    assert (str(a.corners), a.whiten, str(a.order)) == ("b", 0.2, "o.txt")
    a = build_parser().parse_args(["x"])
    assert (a.corners, a.whiten, a.order) == (None, 0.5, None)


def test_cli_end_to_end(photo, tmp_path, capsys):
    out = tmp_path / "out"
    main([str(photo.parent), "-o", str(out), "--no-ocr", "--dpi", "100", "--jobs", "1",
          "--merge", "all"])
    assert (out / "pages" / "sheet.jpg").is_file()
    assert (out / "sheet.pdf").is_file() and (out / "all.pdf").is_file()
    assert "OK sheet.png" in capsys.readouterr().out
