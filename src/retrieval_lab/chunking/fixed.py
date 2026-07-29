"""Fixed-size chunker: a sliding character window with overlap (spec §I.7)."""

from __future__ import annotations

from retrieval_lab.chunking.base import Chunker
from retrieval_lab.models import Chunk, Document


class FixedSizeChunker(Chunker):
    """Fixed-width character windows stepping by ``chunk_size - overlap``.

    The step is ``<= chunk_size``, so windows tile the document with no gaps (overlap only
    adds coverage). Character units keep offsets exact for span-coverage scoring.
    """

    def __init__(self, chunk_size: int = 400, overlap: int = 0) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    @property
    def spec(self) -> str:
        return f"fixed:size={self.chunk_size}:overlap={self.overlap}"

    def chunk(self, doc: Document) -> list[Chunk]:
        n = len(doc.text)
        if n == 0:
            return []
        step = self.chunk_size - self.overlap
        spans: list[tuple[int, int]] = []
        start = 0
        while start < n:
            end = min(start + self.chunk_size, n)
            spans.append((start, end))
            if end == n:
                break
            start += step
        return self._emit(doc, spans)
