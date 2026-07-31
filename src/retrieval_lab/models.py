"""Core data model (spec §I.6).

These types are deliberately plain: a ``Document`` is source text with an id, a ``Chunk`` is
a retrievable slice that *always* remembers where it came from in the source
(``source_id`` + ``[start, end)``), and a ``Config`` names one point in the sweep grid. The
gold types (``GoldSpan``/``EvidenceSet``/``GoldAnswer``/``Query``) live in ``gold.py``
alongside the coverage predicate they are evaluated by.

The load-bearing invariant (spec §I.8, "chunk-identity problem"): gold is defined against
source-document character spans, never against chunk indices, because changing the chunker
changes the chunks. Every ``Chunk`` therefore carries a source pointer so a hit can be
recomputed by coverage, independently per chunker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from retrieval_lab.hashing import stable_hash


@dataclass(frozen=True)
class Document:
    """A source document. ``text`` is the exact string gold offsets index into."""

    id: str
    text: str
    meta: dict = field(default_factory=dict)


def compute_chunk_id(source_id: str, start: int, end: int, chunker_spec: str) -> str:
    """Stable chunk id = hash(source_id, start, end, chunker_spec) (spec §I.6).

    ``chunker_spec`` is included so the same span produced by two different chunkers gets
    two different ids — chunk identity is chunker-relative, which is exactly why gold cannot
    be pinned to a chunk id.
    """
    return stable_hash(source_id, start, end, chunker_spec)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit derived from a document.

    ``start``/``end`` are character offsets into the source document's ``text`` with the
    half-open convention ``[start, end)``. ``parent_id`` links a small indexed child to the
    larger parent that is actually returned (parent-child chunking); ``None`` otherwise.
    """

    id: str
    source_id: str
    start: int
    end: int
    text: str
    parent_id: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid chunk span [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        """Character length of the chunk's source span."""
        return self.end - self.start

    @classmethod
    def make(
        cls,
        source_id: str,
        start: int,
        end: int,
        text: str,
        chunker_spec: str,
        parent_id: str | None = None,
    ) -> Chunk:
        """Construct a chunk, computing its stable id from its source pointer + chunker."""
        return cls(
            id=compute_chunk_id(source_id, start, end, chunker_spec),
            source_id=source_id,
            start=start,
            end=end,
            text=text,
            parent_id=parent_id,
        )


@dataclass(frozen=True)
class Config:
    """One point in the sweep grid (spec §I.6).

    ``retrieval`` is one of ``"dense"``, ``"sparse"``, ``"hybrid"``. ``rerank`` is a reranker
    name or ``None``. ``candidate_n`` is the shortlist size retrieved before fusion/rerank;
    ``top_k`` is the final cutoff.
    """

    embed_model: str
    chunker: str
    retrieval: str
    rerank: str | None = None
    top_k: int = 5
    candidate_n: int = 50
    budget_tokens: int | None = None
    dense_index: str = "exact"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef: int = 50

    def __post_init__(self) -> None:
        if self.retrieval not in {"dense", "sparse", "hybrid"}:
            raise ValueError("retrieval must be dense, sparse, or hybrid")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.candidate_n < self.top_k:
            raise ValueError("candidate_n must be greater than or equal to top_k")
        if self.budget_tokens is not None and self.budget_tokens < 0:
            raise ValueError("budget_tokens must be non-negative")
        if self.dense_index not in {"exact", "hnsw", "none"}:
            raise ValueError("dense_index must be exact, hnsw, or none")
        if self.dense_index == "hnsw" and (
            self.hnsw_m <= 0 or self.hnsw_ef_construction <= 0 or self.hnsw_ef <= 0
        ):
            raise ValueError("HNSW m, ef_construction, and ef must be positive")
        if self.retrieval == "sparse":
            if self.dense_index == "hnsw":
                raise ValueError("sparse retrieval cannot use a dense index")
            object.__setattr__(self, "dense_index", "none")
        elif self.dense_index == "none":
            raise ValueError("dense and hybrid retrieval need a dense index")

    @property
    def id(self) -> str:
        """Stable, human-readable-ish config id used to key results and caches."""
        rr = self.rerank or "none"
        budget = self.budget_tokens if self.budget_tokens is not None else "none"
        hnsw = (
            f"|hnsw_m={self.hnsw_m}|hnsw_ef={self.hnsw_ef}"
            f"|hnsw_efc={self.hnsw_ef_construction}"
            if self.dense_index == "hnsw" else ""
        )
        return (
            f"{self.embed_model}|{self.chunker}|{self.retrieval}|rerank={rr}"
            f"|k={self.top_k}|n={self.candidate_n}|budget={budget}|index={self.dense_index}"
            f"{hnsw}"
        )


@dataclass
class QueryResult:
    """Per-(query, config) outcome (spec §I.6).

    - ``single_chunk_coverage_by_span``: max coverage any *single* chunk gives each required
      gold span — a span satisfied only across multiple chunks shows up low here.
    - ``fragmented_spans``: spans whose union-coverage satisfies gold but whose best single
      chunk falls below threshold (the fragmentation signal).
    - ``gold_completion_rank``: smallest ``r`` where the top-``r`` chunks satisfy an
      EvidenceSet; ``None`` if never satisfied within the ranked list.
    - ``stage_attribution``: earliest failing DAG stage (filled in by ``attribution.py``).
    - ``branch_diag``: dense-vs-sparse diagnostics, reported never auto-promoted to a verdict.
    """

    query_id: str
    config_id: str
    hit: bool
    single_chunk_coverage_by_span: dict[str, float] = field(default_factory=dict)
    fragmented_spans: list[str] = field(default_factory=list)
    gold_completion_rank: int | None = None
    stage_attribution: str | None = None
    branch_diag: dict | None = None
    retrieved_tokens: int = 0
    refused: bool = False
    refused_reason: str | None = None
