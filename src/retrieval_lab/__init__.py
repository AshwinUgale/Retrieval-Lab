"""Retrieval Lab — corpus-specific retrieval benchmark and deterministic stage-level
failure-attribution engine for RAG pipelines.

See ``PROJECTS-TECHNICAL-SPEC.md`` Part I for the authoritative design.
"""

from retrieval_lab.chunking import Chunker, FixedSizeChunker, RecursiveChunker
from retrieval_lab.embedding import DeterministicEmbedder, Embedder, EmbeddingCache
from retrieval_lab.gold import (
    EvidenceSet,
    GoldAnswer,
    GoldSpan,
    OffsetVerificationError,
    Query,
    coverage,
    gold_completion_rank,
    satisfies_gold,
    single_chunk_coverage_by_span,
)
from retrieval_lab.models import Chunk, Config, Document, QueryResult, compute_chunk_id
from retrieval_lab.retrieval import DenseRetriever
from retrieval_lab.scoring import score_query

__all__ = [
    "Document",
    "Chunk",
    "Config",
    "QueryResult",
    "compute_chunk_id",
    "GoldSpan",
    "EvidenceSet",
    "GoldAnswer",
    "Query",
    "OffsetVerificationError",
    "coverage",
    "single_chunk_coverage_by_span",
    "satisfies_gold",
    "gold_completion_rank",
    "Chunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "Embedder",
    "EmbeddingCache",
    "DeterministicEmbedder",
    "DenseRetriever",
    "score_query",
]

__version__ = "0.0.0"
