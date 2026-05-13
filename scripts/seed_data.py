"""Seed test data into the knowledge base.

Usage:
    python scripts/seed_data.py /path/to/sample.pdf
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsing.loader import DocumentLoader
from src.parsing.chunker import Chunker


def main(file_path: str):
    loader = DocumentLoader()
    chunker = Chunker(strategy="sentence", chunk_size=512, chunk_overlap=50)

    print(f"Parsing: {file_path}")
    docs = loader.load(file_path)
    print(f"  → {len(docs)} pages/segments")

    chunks = chunker.chunk(docs)
    print(f"  → {len(chunks)} chunks")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i} ---")
        print(chunk.metadata)
        print(chunk.content[:200] + "...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/seed_data.py <file_path>")
        sys.exit(1)
    main(sys.argv[1])
