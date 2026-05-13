"""OCR via PaddleOCR — for scanned PDFs and images."""

import logging
from pathlib import Path

from src.parsing.loader import ParsedDocument

logger = logging.getLogger(__name__)


class OCRParser:
    """Extract text from images and scanned PDFs via PaddleOCR.

    Best for: scanned documents, photos of text, Chinese text.
    Auto-detects text language, strong on Chinese.
    """

    def __init__(self, lang: str = "ch"):
        self.lang = lang
        self._ocr = None

    def _ensure_ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(lang=self.lang, use_angle_cls=True)
        return self._ocr

    def parse(self, file_path: str) -> list[ParsedDocument]:
        ocr = self._ensure_ocr()
        file_path_obj = Path(file_path)

        if file_path_obj.suffix.lower() == ".pdf":
            return self._parse_pdf(str(file_path), ocr)
        else:
            return self._parse_image(str(file_path), ocr)

    def _parse_image(self, file_path: str, ocr) -> list[ParsedDocument]:
        result = ocr.ocr(file_path)
        if not result or not result[0]:
            logger.warning("No text found in %s", file_path)
            return []

        lines = []
        for line_info in result[0]:
            if line_info and len(line_info) >= 2:
                text = line_info[1][0] if isinstance(line_info[1], (list, tuple)) else str(line_info[1])
                lines.append(text)

        content = "\n".join(lines)
        logger.info("PaddleOCR extracted %d lines from %s", len(lines), file_path)

        return [ParsedDocument(
            file_path=file_path,
            file_type="image",
            content=content,
            metadata={"source": Path(file_path).stem},
            parser_used="paddleocr",
        )]

    def _parse_pdf(self, file_path: str, ocr) -> list[ParsedDocument]:
        """For scanned PDFs: convert each page to image then OCR."""
        import fitz

        docs = []
        pdf = fitz.open(file_path)
        file_stem = Path(file_path).stem

        for page_num in range(len(pdf)):
            page = pdf[page_num]
            pix = page.get_pixmap(dpi=200)
            img_path = f"/tmp/_ocr_page_{page_num}.png"
            pix.save(img_path)

            parsed = self._parse_image(img_path, ocr)
            if parsed:
                parsed[0].metadata["page"] = page_num + 1
                parsed[0].metadata["total_pages"] = len(pdf)
                parsed[0].metadata["source"] = file_stem
                parsed[0].file_type = "pdf"
                docs.extend(parsed)

        pdf.close()
        return docs
