"""Plain text parser — .txt files."""

from pathlib import Path

from src.parsing.loader import ParsedDocument


class TextParser:
    """Parse plain text files."""

    def parse(self, file_path: str) -> list[ParsedDocument]:
        text = Path(file_path).read_text(encoding="utf-8")
        return [ParsedDocument(
            file_path=file_path,
            file_type="txt",
            content=text,
            metadata={"chars": len(text)},
            parser_used="text",
        )]
