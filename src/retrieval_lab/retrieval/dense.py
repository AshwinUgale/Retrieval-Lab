"""Dense retrieval: embed the query, rank chunks by cosine similarity (spec §I.7).

Vectors are L2-normalized at the embedder, so cosine is a plain dot product and the whole
ranking is one matrix-vector multiply. This is the *exact* dense retriever (numpy, no ANN):
for the corpus sizes Retrieval Lab targets it is fast and has no recall loss, so a dense miss
is never an index artifact. The ANN option (which *can* lose recall, and reports when it
does) arrives in Phase 7.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from retrieval_lab.embedding.base import Embedder
from retrieval_lab.models import Chunk


class DenseRetriever:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def index(self, chunks: Iterable[Chunk]) -> DenseRetriever:
        """Embed and store the chunk corpus (as passages). Returns self for chaining."""
        self._chunks = list(chunks)
        self._matrix = self.embedder.embed_passage([c.text for c in self._chunks])
        return self

    def retrieve_scored(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Top-``k`` chunks by cosine, as ``(chunk, score)``, highest first.

        Ties break by original corpus order (stable sort) so results are deterministic.
        """
        if self._matrix is None:
            raise RuntimeError("index() must be called before retrieve")
        if not self._chunks or k <= 0:
            return []
        qv = self.embedder.embed_query([query])[0]
        scores = self._matrix @ qv
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self._chunks[i], float(scores[i])) for i in order]

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        """Top-``k`` chunks by cosine, highest first."""
        return [c for c, _ in self.retrieve_scored(query, k)]

    @property
    def index_nbytes(self) -> int:
        """Rough index size: the stored vector matrix in bytes (0 before indexing)."""
        return int(self._matrix.nbytes) if self._matrix is not None else 0
