"""Embedding adapters + content-addressed cache (spec §I.7)."""

from retrieval_lab.embedding.base import Embedder, EmbeddingCache, l2_normalize
from retrieval_lab.embedding.deterministic import DeterministicEmbedder

# Real-model embedders live behind the ``[real-embed]`` extra; import lazily so the core
# never imports sentence-transformers.
__all__ = [
    "Embedder",
    "EmbeddingCache",
    "l2_normalize",
    "DeterministicEmbedder",
    "SentenceTransformerEmbedder",
    "e5_embedder",
    "bge_embedder",
]


def __getattr__(name: str):
    if name in {"SentenceTransformerEmbedder", "e5_embedder", "bge_embedder"}:
        from retrieval_lab.embedding import sentence_transformer as _st

        return getattr(_st, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
