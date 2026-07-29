"""Embedding adapters + content-addressed cache (spec §I.7)."""

from retrieval_lab.embedding.base import Embedder, EmbeddingCache, l2_normalize
from retrieval_lab.embedding.deterministic import DeterministicEmbedder

__all__ = ["Embedder", "EmbeddingCache", "l2_normalize", "DeterministicEmbedder"]
