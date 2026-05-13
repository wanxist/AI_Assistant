"""Image text extraction — OCR for image files."""

import logging
from pathlib import Path

from src.parsing.loader import ParsedDocument
from src.parsing.ocr import OCRParser

logger = logging.getLogger(__name__)


class ImageParser:
    """Parse image files using OCR.

    For standalone images (.png, .jpg, etc.) — delegates to OCRParser internally.
    """

    def __init__(self):
        self._ocr = OCRParser()

    def parse(self, file_path: str) -> list[ParsedDocument]:
        return self._ocr.parse(file_path)
