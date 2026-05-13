"""Text chunking strategies for downstream RAG ingestion."""

import re
from typing import Iterator

from src.parsing.loader import ParsedDocument


class Chunker:
    """Split parsed documents into smaller chunks for embedding.

    Supports:
    - fixed_size: split by character count with overlap
    - sentence: split on sentence boundaries (Chinese + English)
    - markdown_header: split on markdown headings (##, ###)
    """

    def __init__(
        self,
        strategy: str = "sentence",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, docs: list[ParsedDocument]) -> list[ParsedDocument]:
        chunks = []
        for doc in docs:
            texts = self._split(doc.content)
            for i, text in enumerate(texts):
                chunks.append(ParsedDocument(
                    file_path=doc.file_path,
                    file_type=doc.file_type,
                    content=text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                    },
                    parser_used=doc.parser_used,
                ))
        return chunks

    def _split(self, text: str) -> list[str]:
        if self.strategy == "fixed_size":
            return self._fixed_size_split(text)
        elif self.strategy == "sentence":
            return self._sentence_split(text)
        elif self.strategy == "markdown_header":
            return self._markdown_header_split(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

    def _fixed_size_split(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _sentence_split(self, text: str) -> list[str]:
        # Split on Chinese/English sentence boundaries
        sentences = re.split(r"(?<=[。！？.!?\n])\s*", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) <= self.chunk_size:
                current += sent
            else:
                if current:
                    chunks.append(current)
                current = sent
        if current:
            chunks.append(current)
        return chunks

    def _markdown_header_split(self, text: str) -> list[str]:
        # Split on ## or ### headers
        sections = re.split(r"\n(?=#{2,3}\s)", text)
        return [s.strip() for s in sections if s.strip()]
