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
    - recursive: hierarchical fallback: paragraph → sentence → fixed_size.
      Preserves semantic boundaries as much as possible.
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
        elif self.strategy == "recursive":
            return self._recursive_split(text)
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

    def _recursive_split(self, text: str) -> list[str]:
        """递归分块：段落 → 句子 → 字符滑动窗口，逐级降级。
        
        优先保持语义完整性，遇到超长文本自动降级到更细粒度。
        """
        # Level 1: 按段落切分（连续空行）
        paragraphs = re.split(r"\n\s*\n", text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        result = []
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                result.append(para)
                continue
            # Level 2: 段落太大，按句子切分
            sentences = re.split(r"(?<=[。！？.!?\n])\s*", para)
            sentences = [s.strip() for s in sentences if s.strip()]

            buffer = ""
            for sent in sentences:
                if len(sent) > self.chunk_size:
                    # Level 3: 句子太大，用字符滑动窗口
                    if buffer:
                        result.append(buffer)
                        buffer = ""
                    result.extend(self._fixed_size_split(sent))
                elif len(buffer) + len(sent) <= self.chunk_size:
                    buffer += sent
                else:
                    result.append(buffer)
                    buffer = sent
            if buffer:
                result.append(buffer)
        return result
