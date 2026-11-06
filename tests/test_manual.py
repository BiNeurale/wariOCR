import cv2
import numpy as np

from wariocr import manual_quad

BLUE = (255, 0, 0)  # BGR pure blue, as drawn in an image editor
W, H = 600, 800


def canvas() -> np.ndarray:
    img = np.full((H, W, 3), 170, np.uint8)
    cv2.rectangle(img, (100, 100), (500, 700), (235, 235, 235), -1)
    return img


def save(tmp_path, img):
    p = tmp_path / "photo_detect.jpg"
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return p


def test_closed_outline_scaled_to_photo(tmp_path):
    quad = np.array([[90, 80], [520, 110], [505, 720], [70, 690]], np.int32)
    img = canvas()
    cv2.polylines(img, [quad], True, BLUE, 4)
    # a dark ballpoint-blue stroke inside the sheet must be ignored
    cv2.line(img, (150, 300), (450, 320), (120, 40, 30), 4)
    out = manual_quad(save(tmp_path, img), (2 * H, 2 * W, 3))
    assert out is not None
    np.testing.assert_allclose(out, 2 * quad, atol=2 * 2.0)


def test_undrawn_side_on_the_border(tmp_path):
    # the sheet runs off the bottom: only three sides are drawn
    pts = np.array([[80, H - 1], [100, 120], [480, 100], [520, H - 1]], np.int32)
    img = canvas()
    cv2.polylines(img, [pts], False, BLUE, 4)
    out = manual_quad(save(tmp_path, img), (H, W, 3))
    assert out is not None
    np.testing.assert_allclose(out[:2], pts[1:3], atol=2.0)
    assert (out[2:, 1] >= H - 4).all()  # bottom corners on the frame border


def test_no_blue_returns_none(tmp_path):
    assert manual_quad(save(tmp_path, canvas()), (H, W, 3)) is None
