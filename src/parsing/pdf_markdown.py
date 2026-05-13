"""PDF→Markdown via marker — preserves structure for complex layouts."""

import logging
from pathlib import Path

from src.parsing.loader import ParsedDocument

logger = logging.getLogger(__name__)


class MarkerParser:
    """Convert PDF to Markdown using marker-pdf.

    Best for: PDFs with tables, multi-column layouts, or mixed content.
    Slower than PyMuPDF but preserves document structure.
    """

    def parse(self, file_path: str) -> list[ParsedDocument]:
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
        except ImportError:
            raise ImportError(
                "marker-pdf is required. Install with: pip install marker-pdf"
            )

        converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )

        rendered = converter(str(file_path))
        markdown_text = rendered.markdown

        logger.info("Marker converted %s to markdown (%d chars)", file_path, len(markdown_text))

        return [ParsedDocument(
            file_path=file_path,
            file_type="pdf",
            content=markdown_text,
            metadata={
                "source": Path(file_path).stem,
                "format": "markdown",
            },
            parser_used="marker",
        )]
