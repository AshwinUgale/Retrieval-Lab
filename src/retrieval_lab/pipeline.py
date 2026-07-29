"""Orchestrator: run one config on a query, record every stage, score + attribute.

This is the seam the scorer and the attribution engine share. It runs the retrieval DAG for
a ``Config`` (dense / sparse / hybrid), captures the intermediate item sets into a
``StageOutputs``, scores the final cutoff, and asks ``attribution.attribute`` where a miss
happened. The sweep engine (Phase 5) drives many of these; reranker and budget stages plug
in here in Phase 3.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from retrieval_lab.attribution import StageOutputs, attribute
from retrieval_lab.budget import pack_by_budget
from retrieval_lab.gold import DEFAULT_MIN_GOLD_COVERAGE, Query
from retrieval_lab.models import Chunk, Config, QueryResult
from retrieval_lab.retrieval.bm25 import BM25Retriever
from retrieval_lab.retrieval.dense import DenseRetriever
from retrieval_lab.retrieval.fusion import DEFAULT_RRF_C, reciprocal_rank_fusion
from retrieval_lab.retrieval.rerank import Reranker
from retrieval_lab.scoring import score_query

_DENSE_MODES = {"dense", "hybrid"}
_SPARSE_MODES = {"sparse", "hybrid"}


def _dedup(chunks: Iterable[Chunk]) -> list[Chunk]:
    """De-duplicate chunks by id, preserving first-seen order."""
    seen: set[str] = set()
    out: list[Chunk] = []
    for c in chunks:
        if c.id not in seen:
            seen.add(c.id)
            out.append(c)
    return out


class RetrievalPipeline:
    """Runs a single ``Config`` over an indexed corpus, producing ``StageOutputs`` per query.

    ``dense`` / ``sparse`` are retrievers already indexed on the *same* chunk set as
    ``chunks``. Only the retrievers the config's mode needs are required.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        config: Config,
        dense: DenseRetriever | None = None,
        sparse: BM25Retriever | None = None,
        reranker: Reranker | None = None,
        rrf_c: int = DEFAULT_RRF_C,
    ) -> None:
        self.chunks = list(chunks)
        self.config = config
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.rrf_c = rrf_c
        if config.retrieval in _DENSE_MODES and dense is None:
            raise ValueError(f"{config.retrieval!r} needs a dense retriever")
        if config.retrieval in _SPARSE_MODES and sparse is None:
            raise ValueError(f"{config.retrieval!r} needs a sparse retriever")
        if config.retrieval not in (_DENSE_MODES | _SPARSE_MODES):
            raise ValueError(f"unknown retrieval mode {config.retrieval!r}")
        if config.rerank and reranker is None:
            raise ValueError(f"config requests reranker {config.rerank!r} but none supplied")

    def run(self, query_text: str) -> StageOutputs:
        cfg = self.config
        n = cfg.candidate_n
        dense_c = self.dense.retrieve(query_text, n) if cfg.retrieval in _DENSE_MODES else None
        sparse_c = self.sparse.retrieve(query_text, n) if cfg.retrieval in _SPARSE_MODES else None

        fused = None
        if cfg.retrieval == "dense":
            union = list(dense_c or [])
            shortlist = list(dense_c or [])
        elif cfg.retrieval == "sparse":
            union = list(sparse_c or [])
            shortlist = list(sparse_c or [])
        else:  # hybrid
            union = _dedup([*(dense_c or []), *(sparse_c or [])])
            fused = reciprocal_rank_fusion([dense_c or [], sparse_c or []], self.rrf_c)[:n]
            shortlist = fused

        # Optional reranker reorders the whole candidate_n shortlist before the cutoff.
        reranker_input = None
        reranked = None
        if cfg.rerank and self.reranker is not None:
            reranker_input = shortlist
            reranked = self.reranker.rerank(query_text, shortlist)
            pre_final = reranked
        else:
            pre_final = shortlist

        final = pre_final[: cfg.top_k]

        # Optional budget packing produces the delivered subset of the final top_k.
        budget_packed = None
        if cfg.budget_tokens is not None:
            budget_packed, _ = pack_by_budget(final, cfg.budget_tokens)

        return StageOutputs(
            all_chunks=self.chunks,
            candidate_union=union,
            pre_final=pre_final,
            final=final,
            dense_candidates=dense_c,
            sparse_candidates=sparse_c,
            fused=fused,
            reranker_input=reranker_input,
            reranked=reranked,
            budget_packed=budget_packed,
        )


def evaluate_query(
    query: Query,
    pipeline: RetrievalPipeline,
    min_gold_coverage: float = DEFAULT_MIN_GOLD_COVERAGE,
) -> QueryResult:
    """Run, score, and attribute one query. Returns a fully populated ``QueryResult``."""
    outs = pipeline.run(query.text)
    # The delivered context is the budget-packed subset when a budget applies, else top_k.
    delivered = outs.budget_packed if outs.budget_packed is not None else outs.final
    result = score_query(
        query,
        outs.pre_final,
        pipeline.config.id,
        pipeline.config.top_k,
        min_gold_coverage,
        final=delivered,
    )
    attr = attribute(outs, query.gold, pipeline.config, min_gold_coverage)
    result.stage_attribution = attr.stage
    result.branch_diag = attr.branch_diag
    return result
