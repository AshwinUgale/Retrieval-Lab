"""Approximate nearest-neighbour dense retrieval via HNSW (spec §I.7).

Exact dense retrieval (``DenseRetriever``) has no recall loss but is linear in the corpus
size. HNSW is sub-linear, trading a little recall for speed. That trade matters for
attribution: an ANN miss can be an *index* artifact rather than a representation or ranking
failure — so Retrieval Lab always reports **which** dense retriever was used, and
``ann_vs_exact_recall`` quantifies how much recall the approximation costs on your queries.

``ANNDenseRetriever`` mirrors ``DenseRetriever``'s interface, so it drops straight into the
pipeline. It needs the ``[ann]`` extra (hnswlib); the import is lazy so the core never
depends on it. The recall diagnostic itself is dependency-free and works on any two
retrievers.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from retrieval_lab.embedding.base import Embedder
from retrieval_lab.models import Chunk


class ANNDenseRetriever:
    """Approximate dense retriever backed by an HNSW index (cosine space)."""

    def __init__(
        self,
        embedder: Embedder,
        m: int = 16,
        ef_construction: int = 200,
        ef: int = 50,
        random_seed: int = 100,
    ) -> None:
        if m <= 0 or ef_construction <= 0 or ef <= 0:
            raise ValueError("HNSW m, ef_construction, and ef must be positive")
        self.embedder = embedder
        self.m = m
        self.ef_construction = ef_construction
        self.ef = ef
        self.random_seed = random_seed
        self.name = f"hnsw(M={m},ef={ef})"
        self._chunks: list[Chunk] = []
        self._index = None

    def index(self, chunks: Iterable[Chunk]) -> ANNDenseRetriever:
        try:
            import hnswlib
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "ANNDenseRetriever needs the '[ann]' extra: pip install 'retrieval-lab[ann]'"
            ) from exc
        self._chunks = list(chunks)
        vectors = self.embedder.embed_passage([c.text for c in self._chunks])
        n, dim = vectors.shape if vectors.size else (0, self.embedder.dim)
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(
            max_elements=max(1, n),
            ef_construction=self.ef_construction,
            M=self.m,
            random_seed=self.random_seed,
        )
        if n:
            index.add_items(np.asarray(vectors, dtype=np.float32), np.arange(n))
        index.set_ef(max(self.ef, 1))
        self._index = index
        return self

    def retrieve_scored(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if self._index is None:
            raise RuntimeError("index() must be called before retrieve")
        if not self._chunks or k <= 0:
            return []
        qv = np.asarray(self.embedder.embed_query([query])[0], dtype=np.float32)
        labels, distances = self._index.knn_query(qv, k=min(k, len(self._chunks)))
        # hnswlib cosine 'distance' is 1 - cosine similarity.
        return [
            (self._chunks[int(i)], 1.0 - float(d))
            for i, d in zip(labels[0], distances[0], strict=False)
        ]

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        return [c for c, _ in self.retrieve_scored(query, k)]

    @property
    def index_nbytes(self) -> int:
        """Serialized HNSW graph size (a stable approximation of index memory)."""
        if self._index is None:
            return 0
        size = getattr(self._index, "index_file_size", None)
        return int(size()) if size is not None else 0


def ann_vs_exact_recall(
    ann_retriever,
    exact_retriever,
    queries: Sequence[str],
    k: int,
) -> dict:
    """Fraction of each query's exact top-k that the ANN retriever also returns (spec §I.7).

    Both retrievers must be indexed on the same chunks. Returns mean recall over the queries
    (1.0 = the ANN index lost nothing) plus the per-query values. Works with any retriever
    pair, so it is testable without hnswlib.
    """
    if not queries:
        return {"k": k, "mean_recall": 1.0, "per_query": [], "n": 0}
    per_query: list[float] = []
    for q in queries:
        exact_ids = {c.id for c in exact_retriever.retrieve(q, k)}
        if not exact_ids:
            per_query.append(1.0)
            continue
        ann_ids = {c.id for c in ann_retriever.retrieve(q, k)}
        per_query.append(len(exact_ids & ann_ids) / len(exact_ids))
    return {
        "k": k,
        "mean_recall": float(np.mean(per_query)),
        "per_query": per_query,
        "n": len(queries),
    }
