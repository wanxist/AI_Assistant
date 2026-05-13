from src.parsing.chunker import Chunker
from src.parsing.loader import ParsedDocument


def make_doc(content: str) -> ParsedDocument:
    return ParsedDocument(
        file_path="test.pdf",
        file_type="pdf",
        content=content,
        parser_used="test",
    )


def test_sentence_chunker():
    chunker = Chunker(strategy="sentence", chunk_size=512, chunk_overlap=50)
    doc = make_doc("第一句话。第二句话！第三句话？第四句话。")
    chunks = chunker.chunk([doc])
    assert len(chunks) >= 1
    # All content should be preserved
    combined = "".join(c.content for c in chunks)
    assert "第一句话" in combined
    assert "第四句话" in combined


def test_fixed_size_chunker():
    chunker = Chunker(strategy="fixed_size", chunk_size=50, chunk_overlap=10)
    doc = make_doc("A" * 200)
    chunks = chunker.chunk([doc])
    assert len(chunks) > 1
    assert all(len(c.content) <= 50 for c in chunks)
