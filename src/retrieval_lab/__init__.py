"""Retrieval Lab — a corpus-specific retrieval benchmark and deterministic, stage-level
failure-attribution engine for RAG pipelines.

The names re-exported here are the **supported public API** and follow semantic versioning.
Everything else remains importable from its submodule (e.g. ``retrieval_lab.attribution``,
``retrieval_lab.metrics``) but is considered internal and may change between minor versions.

Real embedding models (e5/bge) live behind the ``[real-embed]`` extra and are imported lazily
from ``retrieval_lab.embedding`` (``from retrieval_lab.embedding import e5_embedder``) so the
top-level import never requires sentence-transformers.

See ``PROJECTS-TECHNICAL-SPEC.md`` Part I for the authoritative design.
"""

# ruff: noqa: I001 — imports are grouped by role (matching __all__), not alphabetically.

# --- Data model ---------------------------------------------------------------------
from retrieval_lab.models import Chunk, Config, Document, QueryResult

# --- Ground truth: source-span gold + the coverage predicate ------------------------
from retrieval_lab.gold import (
    EvidenceSet,
    GoldAnswer,
    GoldSpan,
    OffsetVerificationError,
    Query,
    coverage,
    gold_completion_rank,
    load_documents,
    load_queries,
    satisfies_gold,
)

# --- Author gold from answer quotes (no hand-computed offsets) -----------------------
from retrieval_lab.authoring import build_gold, build_query, load_authoring_spec, make_span

# --- Import real labeled datasets ---------------------------------------------------
from retrieval_lab.datasets import DatasetImport, load_beir, load_squad

# --- Components: chunkers, embedders, retrievers, rerankers -------------------------
from retrieval_lab.chunking import (
    Chunker,
    FixedSizeChunker,
    ParentChildChunker,
    RecursiveChunker,
    SemanticChunker,
)
from retrieval_lab.embedding import DeterministicEmbedder, Embedder, EmbeddingCache
from retrieval_lab.retrieval import (
    ANNDenseRetriever,
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    LexicalReranker,
    Reranker,
)

# --- The sweep workflow -------------------------------------------------------------
from retrieval_lab.pipeline import RetrievalPipeline, evaluate_query
from retrieval_lab.sweep import SweepResult, SweepSpec, run_sweep
from retrieval_lab.metrics import ANNDiagnostic, ConfigMetrics

# --- Reporting ----------------------------------------------------------------------
from retrieval_lab.report import (
    pareto_frontier,
    read_json,
    render_explain,
    render_html,
    render_pareto,
    render_report,
    write_html,
    write_json,
)

# --- Embedding-space diagnostics ----------------------------------------------------
from retrieval_lab.geometry import geometry_report

__all__ = [
    # Data model
    "Document", "Chunk", "Config", "QueryResult",
    # Ground truth
    "GoldSpan", "EvidenceSet", "GoldAnswer", "Query", "OffsetVerificationError",
    "coverage", "satisfies_gold", "gold_completion_rank",
    "load_documents", "load_queries",
    # Authoring
    "make_span", "build_gold", "build_query", "load_authoring_spec",
    # Datasets
    "DatasetImport", "load_squad", "load_beir",
    # Chunkers
    "Chunker", "FixedSizeChunker", "RecursiveChunker", "SemanticChunker",
    "ParentChildChunker",
    # Embedders
    "Embedder", "DeterministicEmbedder", "EmbeddingCache",
    # Retrievers / rerankers
    "DenseRetriever", "BM25Retriever", "ANNDenseRetriever",
    "Reranker", "LexicalReranker", "CrossEncoderReranker",
    # Workflow
    "SweepSpec", "SweepResult", "run_sweep", "RetrievalPipeline", "evaluate_query",
    "ConfigMetrics", "ANNDiagnostic",
    # Reporting
    "render_report", "render_pareto", "render_explain", "render_html",
    "write_json", "write_html", "read_json", "pareto_frontier",
    # Diagnostics
    "geometry_report",
]

__version__ = "0.1.0"
