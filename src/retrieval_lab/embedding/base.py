"""Embedder adapter + content-addressed cache (spec §I.7).

The adapter contract is deliberately small: ``embed(texts) -> np.ndarray`` returning one
L2-normalized row per text, so cosine similarity is a plain dot product. Every embedder
routes through a **content-addressed cache** keyed by ``hash(model, text)`` so that
re-embedding the same chunk across a sweep is free (spec §I.7) — the same chunk text under
the same model is embedded exactly once, ever.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from retrieval_lab.hashing import stable_hash


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """L2-normalize rows; zero rows (e.g. empty text) stay zero instead of dividing by zero."""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        norm = np.linalg.norm(mat)
        return mat / norm if norm > 0 else mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class EmbeddingCache:
    """In-memory content-addressed cache of individual embedding vectors.

    Keyed by ``hash(model, text)`` so it is stable across runs and safe to share across a
    whole sweep. A persistent on-disk variant is added in the sweep phase.
    """

    def __init__(self) -> None:
        self._store: dict[str, np.ndarray] = {}

    @staticmethod
    def key(model: str, text: str) -> str:
        return stable_hash(model, text)

    def get(self, model: str, text: str) -> np.ndarray | None:
        return self._store.get(self.key(model, text))

    def put(self, model: str, text: str, vec: np.ndarray) -> None:
        self._store[self.key(model, text)] = np.asarray(vec, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._store)


class Embedder(ABC):
    """Base embedder: subclasses implement ``_embed_raw``; the cache is handled here.

    ``name`` identifies the model in the cache key and in config ids, so two distinct models
    never collide and the same model always reuses cached vectors.
    """

    name: str
    dim: int

    def __init__(self, name: str, dim: int, cache: EmbeddingCache | None = None) -> None:
        self.name = name
        self.dim = dim
        self.cache = cache

    @abstractmethod
    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        """Embed every text; return an ``(n, dim)`` array of L2-normalized rows."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` (order-preserving), consulting/filling the cache when present."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.cache is None:
            return self._embed_raw(texts)

        out: list[np.ndarray | None] = [self.cache.get(self.name, t) for t in texts]
        missing_idx = [i for i, v in enumerate(out) if v is None]
        if missing_idx:
            fresh = self._embed_raw([texts[i] for i in missing_idx])
            for j, i in enumerate(missing_idx):
                vec = fresh[j]
                self.cache.put(self.name, texts[i], vec)
                out[i] = vec
        return np.vstack([np.asarray(v, dtype=np.float32) for v in out])

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience: embed a single text and return its 1-D vector."""
        return self.embed([text])[0]
