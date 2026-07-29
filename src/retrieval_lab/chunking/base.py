"""Chunker base class (spec §I.7).

Every chunk records ``(source_id, start, end)`` so gold — defined over source spans — can be
scored by coverage independently per chunker (spec §I.8). The concrete chunkers here
**tile** each document (their chunks' union covers the whole text), so the representation
stage of attribution only fails on genuine *text loss*, never on a boundary artifact.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from retrieval_lab.models import Chunk, Document


class Chunker(ABC):
    @property
    @abstractmethod
    def spec(self) -> str:
        """Stable descriptor of this chunker + its parameters; part of every chunk id."""

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """Split one document into chunks covering its text with no gaps."""

    def chunk_corpus(self, docs: Iterable[Document]) -> list[Chunk]:
        """Chunk every document, preserving document order."""
        out: list[Chunk] = []
        for doc in docs:
            out.extend(self.chunk(doc))
        return out

    def _emit(self, doc: Document, spans: list[tuple[int, int]]) -> list[Chunk]:
        """Build ``Chunk`` objects from ``[start, end)`` spans, dropping empties."""
        chunks: list[Chunk] = []
        for start, end in spans:
            if end <= start:
                continue
            chunks.append(
                Chunk.make(doc.id, start, end, doc.text[start:end], chunker_spec=self.spec)
            )
        return chunks
