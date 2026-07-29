"""Rerankers: re-score a shortlist more precisely than first-stage retrieval (spec §I.7).

A reranker scores ``(query, chunk)`` jointly and reorders the ``candidate_n`` shortlist,
keeping the top ``top_k``. Two implementations:

- ``LexicalReranker`` — keyless and deterministic (shared-term overlap). The default for the
  core/CI path. Because it scores *exact* tokens while the dense embedder also uses char
  n-grams, the two can disagree — which is what lets the validation corpus plant a reranker
  demotion (a chunk dense ranked well that the reranker pushes past the cutoff).
- ``CrossEncoderReranker`` — a real cross-encoder via sentence-transformers, behind the
  ``[rerank]`` extra. More accurate, but a heavy download; lazily imported so the core never
  depends on it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from retrieval_lab.models import Chunk
from retrieval_lab.text import word_tokens


class Reranker(ABC):
    name: str

    @abstractmethod
    def rerank(self, query: str, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Return ``chunks`` reordered most-relevant first (stable on ties)."""


class LexicalReranker(Reranker):
    """Deterministic reranker scoring by count of shared unique query terms."""

    name = "lexical"

    def rerank(self, query: str, chunks: Sequence[Chunk]) -> list[Chunk]:
        q = set(word_tokens(query))
        scored = [
            (i, chunk, len(q & set(word_tokens(chunk.text))))
            for i, chunk in enumerate(chunks)
        ]
        scored.sort(key=lambda t: (-t[2], t[0]))  # score desc, then original order
        return [chunk for _, chunk, _ in scored]


class CrossEncoderReranker(Reranker):
    """Real cross-encoder reranker (sentence-transformers). Requires the ``[rerank]`` extra."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - exercised only with the extra
            raise ImportError(
                "CrossEncoderReranker needs the '[rerank]' extra: pip install "
                "'retrieval-lab[rerank]'"
            ) from exc
        self._model = CrossEncoder(model_name)
        self.name = f"ce:{model_name}"

    def rerank(self, query: str, chunks: Sequence[Chunk]) -> list[Chunk]:  # pragma: no cover
        if not chunks:
            return []
        scores = self._model.predict([(query, c.text) for c in chunks])
        order = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), i))
        return [chunks[i] for i in order]
