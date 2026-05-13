"""PDF text extraction via PyMuPDF (fitz) — fast, for text-based PDFs."""

import logging
from pathlib import Path

from src.parsing.loader import ParsedDocument

logger = logging.getLogger(__name__)


class PyMuPDFParser:
    """Extract text from text-based PDFs using PyMuPDF.

    Best for: PDFs where you can select/copy text.
    Not for: scanned/image PDFs (use OCR parser).
    """

    def parse(self, file_path: str) -> list[ParsedDocument]:
        import fitz

        docs = []
        doc = fitz.open(file_path)
        file_stem = Path(file_path).stem

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            if not text:
                continue

            docs.append(ParsedDocument(
                file_path=file_path,
                file_type="pdf",
                content=text,
                metadata={
                    "page": page_num + 1,
                    "total_pages": len(doc),
                    "source": file_stem,
                },
                parser_used="pymupdf",
            ))

        doc.close()
        logger.info("PyMuPDF extracted %d pages from %s", len(docs), file_path)
        return docs
