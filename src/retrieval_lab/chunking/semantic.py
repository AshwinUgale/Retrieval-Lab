"""Semantic chunker: boundaries where adjacent-sentence meaning shifts (spec §I.7).

Split the document into sentences, embed each, and place a chunk boundary where the distance
between consecutive sentences is unusually large — i.e. where the topic shifts. The threshold
is a **percentile of the document's own distances** (default 75th), so it adapts to whatever
embedder is in use instead of assuming an absolute cosine scale (the keyless embedder and a
real model live on very different scales). A ``max_chars`` cap keeps chunks bounded even
through long stretches with no strong boundary.

Sentences tile the document contiguously, so the emitted chunks do too — representation-stage
coverage stays full.
"""

from __future__ import annotations

import re

import numpy as np

from retrieval_lab.chunking.base import Chunker
from retrieval_lab.embedding.base import Embedder
from retrieval_lab.models import Chunk, Document

_SENT_END = re.compile(r"[.!?]+(?:\s+|$)|\n+")


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Contiguous ``[start, end)`` spans, one per sentence, covering the whole text."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_END.finditer(text):
        spans.append((start, m.end()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if e > s]


class SemanticChunker(Chunker):
    def __init__(
        self,
        embedder: Embedder,
        breakpoint_percentile: float = 75.0,
        max_chars: int = 1000,
    ) -> None:
        if not 0 <= breakpoint_percentile <= 100:
            raise ValueError("breakpoint_percentile must be in [0, 100]")
        self.embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile
        self.max_chars = max_chars

    @property
    def spec(self) -> str:
        return (
            f"semantic:pct={self.breakpoint_percentile}:max={self.max_chars}"
            f":emb={self.embedder.name}"
        )

    def chunk(self, doc: Document) -> list[Chunk]:
        n = len(doc.text)
        if n == 0:
            return []
        sentences = _sentence_spans(doc.text)
        if len(sentences) <= 1:
            return self._emit(doc, sentences or [(0, n)])

        vecs = self.embedder.embed([doc.text[s:e] for s, e in sentences])
        # Distance between consecutive sentences (1 - cosine; vectors are L2-normalized).
        dists = np.array([1.0 - float(vecs[i] @ vecs[i + 1]) for i in range(len(sentences) - 1)])
        threshold = float(np.percentile(dists, self.breakpoint_percentile)) if dists.size else 1.0

        # Group consecutive sentences into chunks, closing at a high-distance boundary or the
        # size cap. Sentences tile the document, so the resulting spans do too.
        spans: list[tuple[int, int]] = []
        cur_start = sentences[0][0]
        for i, (_s, e) in enumerate(sentences):
            is_last = i == len(sentences) - 1
            boundary = (not is_last) and dists[i] >= threshold
            too_big = (e - cur_start) >= self.max_chars
            if is_last or boundary or too_big:
                spans.append((cur_start, e))
                if not is_last:
                    cur_start = sentences[i + 1][0]
        return self._emit(doc, spans)
