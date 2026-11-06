"""wariOCR: phone photos of documents to true-scale, typographic-quality PDFs."""
from .detect import detect_document
from .geometry import PAPER_MM, order_corners
from .io import EXTS, load_image
from .manual import manual_quad
from .pipeline import Result, process

__version__ = "0.1.0"

__all__ = [
    "EXTS",
    "PAPER_MM",
    "Result",
    "detect_document",
    "load_image",
    "manual_quad",
    "order_corners",
    "process",
]
