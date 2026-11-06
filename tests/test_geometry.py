import numpy as np
import pytest

from wariocr import order_corners

TL, TR, BR, BL = [10, 20], [300, 15], [310, 400], [5, 390]


@pytest.mark.parametrize("perm", [(0, 1, 2, 3), (2, 0, 3, 1), (3, 2, 1, 0), (1, 3, 0, 2)])
def test_order_corners_any_input_order(perm):
    pts = np.array([TL, TR, BR, BL], np.float32)[list(perm)]
    out = order_corners(pts)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, np.array([TL, TR, BR, BL], np.float32))


def test_order_corners_tilted_quad():
    # 20 degree rotation of an axis-aligned rectangle about its centre
    a = np.radians(20)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rect = np.array([[-100, -150], [100, -150], [100, 150], [-100, 150]], float)
    quad = rect @ r.T + 500
    out = order_corners(quad[::-1].copy())
    np.testing.assert_allclose(out, quad, atol=1e-3)
