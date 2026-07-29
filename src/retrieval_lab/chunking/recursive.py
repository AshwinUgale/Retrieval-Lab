"""Recursive chunker: split on a separator hierarchy, then merge up to a size (spec §I.7).

Classic recursive character splitting — try the highest-priority separator (paragraph, then
line, then sentence, then space), recursing into any piece still larger than ``chunk_size``,
and finally merging adjacent pieces greedily up to the size. The emitted chunks are then
**stitched contiguous** so their union covers the whole document with no gaps — separators
that fall on a merge boundary are never dropped, keeping span-coverage math exact.
"""

from __future__ import annotations

from retrieval_lab.chunking.base import Chunker
from retrieval_lab.models import Chunk, Document

DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


class RecursiveChunker(Chunker):
    def __init__(
        self,
        chunk_size: int = 400,
        separators: tuple[str, ...] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self.separators = separators

    @property
    def spec(self) -> str:
        return f"recursive:size={self.chunk_size}"

    def chunk(self, doc: Document) -> list[Chunk]:
        n = len(doc.text)
        if n == 0:
            return []
        segments = self._split(doc.text, 0, self.separators)
        merged = self._merge(segments)
        stitched = self._stitch(merged, n)
        return self._emit(doc, stitched)

    def _split(self, text: str, offset: int, separators: tuple[str, ...]) -> list[tuple[int, int]]:
        """Recursively split ``text`` (at absolute ``offset``) into ``[start, end)`` segments."""
        if len(text) <= self.chunk_size or not separators:
            return [(offset, offset + len(text))]

        sep = separators[0]
        rest = separators[1:]

        if sep == "":
            # Hard character split for text with no usable separator left.
            return [
                (offset + i, offset + min(i + self.chunk_size, len(text)))
                for i in range(0, len(text), self.chunk_size)
            ]

        segments: list[tuple[int, int]] = []
        pos = 0
        for part in text.split(sep):
            seg_start = offset + pos
            seg_end = seg_start + len(part)
            if part:
                if len(part) <= self.chunk_size:
                    segments.append((seg_start, seg_end))
                else:
                    segments.extend(self._split(part, seg_start, rest))
            pos += len(part) + len(sep)
        return segments

    def _merge(self, segments: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Greedily merge adjacent segments while the span stays within ``chunk_size``."""
        merged: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = None
        for s, e in segments:
            if cur is None:
                cur = (s, e)
            elif e - cur[0] <= self.chunk_size:
                cur = (cur[0], e)
            else:
                merged.append(cur)
                cur = (s, e)
        if cur is not None:
            merged.append(cur)
        return merged

    @staticmethod
    def _stitch(spans: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
        """Make spans contiguous over ``[0, n)`` so their union covers the whole document."""
        if not spans:
            return [(0, n)]
        stitched: list[tuple[int, int]] = []
        for i in range(len(spans)):
            start = 0 if i == 0 else stitched[-1][1]
            end = spans[i + 1][0] if i + 1 < len(spans) else n
            stitched.append((start, end))
        return stitched
