"""Retrieval Lab — corpus-specific retrieval benchmark and deterministic stage-level
failure-attribution engine for RAG pipelines.

See ``PROJECTS-TECHNICAL-SPEC.md`` Part I for the authoritative design.
"""

from retrieval_lab.attribution import AttributionResult, StageOutputs, attribute
from retrieval_lab.budget import pack_by_budget
from retrieval_lab.chunking import (
    Chunker,
    FixedSizeChunker,
    ParentChildChunker,
    RecursiveChunker,
    SemanticChunker,
)
from retrieval_lab.embedding import DeterministicEmbedder, Embedder, EmbeddingCache
from retrieval_lab.geometry import geometry_report
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
from retrieval_lab.metrics import (
    ConfigMetrics,
    aggregate_config,
    compare_configs,
    validity_report,
    wilson_interval,
)
from retrieval_lab.models import Chunk, Config, Document, QueryResult, compute_chunk_id
from retrieval_lab.pipeline import RetrievalPipeline, evaluate_query
from retrieval_lab.report import (
    pareto_frontier,
    read_json,
    render_explain,
    render_pareto,
    render_report,
    write_json,
)
from retrieval_lab.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    LexicalReranker,
    Reranker,
    reciprocal_rank_fusion,
)
from retrieval_lab.scoring import score_query
from retrieval_lab.sweep import SweepResult, SweepSpec, run_sweep

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
    "SemanticChunker",
    "ParentChildChunker",
    "Embedder",
    "EmbeddingCache",
    "DeterministicEmbedder",
    "DenseRetriever",
    "BM25Retriever",
    "reciprocal_rank_fusion",
    "Reranker",
    "LexicalReranker",
    "CrossEncoderReranker",
    "pack_by_budget",
    "score_query",
    "StageOutputs",
    "AttributionResult",
    "attribute",
    "RetrievalPipeline",
    "evaluate_query",
    "aggregate_config",
    "compare_configs",
    "validity_report",
    "wilson_interval",
    "ConfigMetrics",
    "SweepSpec",
    "SweepResult",
    "run_sweep",
    "render_report",
    "render_pareto",
    "render_explain",
    "pareto_frontier",
    "write_json",
    "read_json",
    "geometry_report",
]

__version__ = "0.0.0"
