"""Small numeric helpers shared by several stages."""
from __future__ import annotations

import numpy as np


def odd(n: float) -> int:
    """Round to the nearest odd integer >= 3 (OpenCV kernel sizes)."""
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    """Hermite ramp from 0 at `e0` to 1 at `e1`."""
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)
