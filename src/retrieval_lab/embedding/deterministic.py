"""Keyless, deterministic embedder — the default and the CI/validation embedder (decision D1).

This is the **hashing trick** (a.k.a. feature hashing): tokenize a text into features (word
unigrams + character n-grams), hash each feature to a bucket with a signed contribution, and
L2-normalize. It needs no model, no network, and is byte-stable across machines, so the
constructed-ground-truth recovery suite (spec §I.13) stays reproducible forever.

It is not a semantic model, but it produces *meaningful, controllable* similarity: texts
that share words score high, and the char-n-gram features give it subword overlap that pure
exact-token BM25 lacks — which is exactly what lets the validation corpus construct
dense-vs-sparse divergence on purpose. Real e5/bge models arrive in Phase 5 behind an extra.
"""

from __future__ import annotations

import numpy as np

from retrieval_lab.embedding.base import Embedder, EmbeddingCache, l2_normalize
from retrieval_lab.hashing import stable_hash
from retrieval_lab.text import char_ngrams, word_tokens


class DeterministicEmbedder(Embedder):
    def __init__(
        self,
        dim: int = 512,
        char_ngram_sizes: tuple[int, ...] = (3, 4, 5),
        use_words: bool = True,
        name: str = "det-hash",
        cache: EmbeddingCache | None = None,
    ) -> None:
        super().__init__(name=name, dim=dim, cache=cache)
        self.char_ngram_sizes = char_ngram_sizes
        self.use_words = use_words

    def _features(self, text: str) -> list[str]:
        feats: list[str] = []
        if self.use_words:
            feats.extend(f"w:{t}" for t in word_tokens(text))
        feats.extend(f"c:{g}" for g in char_ngrams(text, self.char_ngram_sizes))
        return feats

    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = int(stable_hash(feat), 16)
                idx = h % self.dim
                sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
                mat[row, idx] += sign
        return l2_normalize(mat)
