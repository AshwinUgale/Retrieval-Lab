"""Orchestrator: run one config on a query, record every stage, score + attribute.

This is the seam the scorer and the attribution engine share. It runs the retrieval DAG for
a ``Config`` (dense / sparse / hybrid), captures the intermediate item sets into a
``StageOutputs``, scores the final cutoff, and asks ``attribution.attribute`` where a miss
happened. The sweep engine (Phase 5) drives many of these; reranker and budget stages plug
in here in Phase 3.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from retrieval_lab.attribution import StageOutputs, attribute
from retrieval_lab.budget import pack_by_budget
from retrieval_lab.gold import DEFAULT_MIN_GOLD_COVERAGE, Query
from retrieval_lab.models import Chunk, Config, QueryResult
from retrieval_lab.retrieval.ann import ANNDenseRetriever
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
        dense: DenseRetriever | ANNDenseRetriever | None = None,
        dense_reference: DenseRetriever | None = None,
        sparse: BM25Retriever | None = None,
        reranker: Reranker | None = None,
        return_expander: Callable[[Sequence[Chunk]], list[Chunk]] | None = None,
        rrf_c: int = DEFAULT_RRF_C,
    ) -> None:
        self.chunks = list(chunks)
        self.config = config
        self.dense = dense
        self.dense_reference = dense_reference
        self.sparse = sparse
        self.reranker = reranker
        # Maps ranked indexed units to the units actually returned (e.g. parent-child:
        # children -> parents). Identity when None. Applied to every stage set so coverage,
        # attribution, and budgeting all operate on the returned unit.
        self.return_expander = return_expander
        self.rrf_c = rrf_c
        if config.retrieval in _DENSE_MODES and dense is None:
            raise ValueError(f"{config.retrieval!r} needs a dense retriever")
        if config.retrieval in _SPARSE_MODES and sparse is None:
            raise ValueError(f"{config.retrieval!r} needs a sparse retriever")
        if config.retrieval not in (_DENSE_MODES | _SPARSE_MODES):
            raise ValueError(f"unknown retrieval mode {config.retrieval!r}")
        if config.rerank and reranker is None:
            raise ValueError(f"config requests reranker {config.rerank!r} but none supplied")

    def run(self, query_text: str, include_reference: bool = True) -> StageOutputs:
        cfg = self.config
        n = cfg.candidate_n
        dense_c = self.dense.retrieve(query_text, n) if cfg.retrieval in _DENSE_MODES else None
        sparse_c = self.sparse.retrieve(query_text, n) if cfg.retrieval in _SPARSE_MODES else None
        exact_dense_c = (
            self.dense_reference.retrieve(query_text, n)
            if (
                include_reference
                and cfg.retrieval in _DENSE_MODES
                and self.dense_reference is not None
            )
            else None
        )

        fused = None
        exact_union = None
        if cfg.retrieval == "dense":
            union = list(dense_c or [])
            shortlist = list(dense_c or [])
            if exact_dense_c is not None:
                exact_union = list(exact_dense_c)
        elif cfg.retrieval == "sparse":
            union = list(sparse_c or [])
            shortlist = list(sparse_c or [])
        else:  # hybrid
            union = _dedup([*(dense_c or []), *(sparse_c or [])])
            if exact_dense_c is not None:
                exact_union = _dedup([*exact_dense_c, *(sparse_c or [])])
            fused = reciprocal_rank_fusion([dense_c or [], sparse_c or []], self.rrf_c)[:n]
            shortlist = fused

        # Optional reranker reorders the whole candidate_n shortlist before the cutoff.
        reranker_input_children = None
        reranked_children = None
        if cfg.rerank and self.reranker is not None:
            reranker_input_children = shortlist
            reranked_children = self.reranker.rerank(query_text, shortlist)
            pre_children = reranked_children
        else:
            pre_children = shortlist

        # Expand every set to the returned unit (identity unless a return_expander is set,
        # e.g. parent-child: children -> parents), so coverage/attribution/budget are all
        # computed on what is actually returned.
        expand = self.return_expander or (lambda xs: list(xs))
        pre_final = expand(pre_children)
        final = pre_final[: cfg.top_k]

        # Optional budget packing produces the delivered subset of the final top_k.
        budget_packed = None
        if cfg.budget_tokens is not None:
            budget_packed, _ = pack_by_budget(final, cfg.budget_tokens)

        return StageOutputs(
            all_chunks=expand(self.chunks),
            candidate_union=expand(union),
            pre_final=pre_final,
            final=final,
            dense_candidates=None if dense_c is None else expand(dense_c),
            sparse_candidates=None if sparse_c is None else expand(sparse_c),
            exact_candidate_union=None if exact_union is None else expand(exact_union),
            fused=None if fused is None else expand(fused),
            reranker_input=(
                None if reranker_input_children is None else expand(reranker_input_children)
            ),
            reranked=None if reranked_children is None else expand(reranked_children),
            budget_packed=budget_packed,
        )

    def attach_exact_reference(self, query_text: str, outs: StageOutputs) -> None:
        """Attach ANN's exact counterfactual after timed retrieval has completed."""
        if self.dense_reference is None or self.config.retrieval not in _DENSE_MODES:
            return
        expand = self.return_expander or (lambda xs: list(xs))
        exact_dense = expand(self.dense_reference.retrieve(query_text, self.config.candidate_n))
        if self.config.retrieval == "hybrid":
            outs.exact_candidate_union = _dedup(
                [*exact_dense, *(outs.sparse_candidates or [])]
            )
        else:
            outs.exact_candidate_union = exact_dense


def score_and_attribute(
    query: Query,
    outs: StageOutputs,
    config: Config,
    min_gold_coverage: float = DEFAULT_MIN_GOLD_COVERAGE,
) -> QueryResult:
    """Score + attribute already-computed stage outputs (retrieval already done).

    Split out from ``evaluate_query`` so a caller that times retrieval can reuse the run
    instead of executing it twice.
    """
    # The delivered context is the budget-packed subset when a budget applies, else top_k.
    delivered = outs.budget_packed if outs.budget_packed is not None else outs.final
    result = score_query(
        query, outs.pre_final, config.id, config.top_k, min_gold_coverage, final=delivered
    )
    attr = attribute(outs, query.gold, config, min_gold_coverage)
    result.stage_attribution = attr.stage
    result.branch_diag = attr.branch_diag
    return result


def evaluate_query(
    query: Query,
    pipeline: RetrievalPipeline,
    min_gold_coverage: float = DEFAULT_MIN_GOLD_COVERAGE,
) -> QueryResult:
    """Run, score, and attribute one query. Returns a fully populated ``QueryResult``."""
    outs = pipeline.run(query.text)
    return score_and_attribute(query, outs, pipeline.config, min_gold_coverage)
