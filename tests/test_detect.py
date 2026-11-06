import numpy as np
import pytest
import synth

from wariocr import detect_document

CASES = {
    "frontal": [[120, 110], [800, 130], [790, 1090], [110, 1080]],
    "rotated_20deg": [[250, 80], [840, 300], [650, 1150], [60, 930]],
    "perspective": [[220, 150], [690, 160], [850, 1120], [60, 1110]],
    "out_of_frame": [[150, 200], [860, 180], [960, 1260], [130, 1150]],
}
TOP = [[180, 160], [780, 190], [760, 1020], [150, 1000]]
UNDER = [[170, 200], [775, 220], [770, 1070], [150, 1050]]  # peeks out at the bottom


@pytest.mark.parametrize("name", CASES)
def test_detects_sheet(name):
    quad = np.array(CASES[name], np.float32)
    found = detect_document(synth.scene(quad))
    assert found is not None
    assert synth.iou(quad, found) >= 0.96


def test_stacked_sheet_regression():
    found = detect_document(synth.scene(TOP, under=UNDER))
    assert found is not None
    assert synth.iou(TOP, found) >= 0.90


@pytest.mark.xfail(strict=True, reason="known weakness: a sheet peeking out below is merged")
def test_stacked_sheet_top_only():
    found = detect_document(synth.scene(TOP, under=UNDER))
    assert synth.iou(TOP, found) >= 0.96


def test_blank_background_returns_none():
    assert detect_document(synth.background()) is None
