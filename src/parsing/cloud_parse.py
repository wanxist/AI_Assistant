"""Cloud-based high-precision PDF parsing via llama-parse."""

import logging
from pathlib import Path

from src.config import settings
from src.parsing.loader import ParsedDocument

logger = logging.getLogger(__name__)


class LlamaCloudParser:
    """Parse PDF with llama-parse cloud service.

    Best for: high-accuracy needs (legal/contracts/financial reports).
    Free tier: 1000 pages/day.
    Falls back to local parser on failure.
    """

    def parse(self, file_path: str) -> list[ParsedDocument]:
        from llama_parse import LlamaParse

        if not settings.llama_cloud_api_key:
            raise RuntimeError("LLAMA_CLOUD_API_KEY not set")

        parser = LlamaParse(
            api_key=settings.llama_cloud_api_key,
            result_type="markdown",
            verbose=False,
        )

        documents = parser.load_data(str(file_path))

        results = []
        for doc in documents:
            results.append(ParsedDocument(
                file_path=file_path,
                file_type="pdf",
                content=doc.text if hasattr(doc, "text") else str(doc),
                metadata={
                    "source": Path(file_path).stem,
                    "format": "markdown",
                    **getattr(doc, "metadata", {}),
                },
                parser_used="llama-parse",
            ))

        logger.info("llama-parse processed %s → %d documents", file_path, len(results))
        return results
