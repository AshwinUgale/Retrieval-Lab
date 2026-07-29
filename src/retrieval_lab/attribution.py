"""DAG failure attribution (spec §I.7).

When a query misses, this assigns the failure to the **earliest stage of the retrieval DAG
whose item set can no longer satisfy gold** — using the *exact same* ``satisfies_gold``
union-coverage predicate the scorer uses, so scorer and attribution can never disagree. Two
chunks each covering half a span jointly satisfy it, and attribution treats that stage as a
success, not a failure.

Dense and sparse are **parallel branches**, so a "dense miss" is not a root cause when sparse
found the answer — branch results are reported as *diagnostics*, never auto-promoted to the
verdict. The DAG is **conditional on the config**: the fusion stage only exists for hybrid,
and reranker / budget stages (added in Phase 3) only when configured.

Stages (spec §I.7), checked earliest-first:

1. ``representation``        — the chunker's full chunk set can't satisfy gold (text loss).
                              (Tiling chunkers never trip this; a lossy chunker does.)
2. ``candidate_generation``  — chunks satisfy, but the raw candidate union (dense ∪ sparse)
                              does not. Branch diagnostics say which branch missed.
3. ``fusion``  (hybrid)      — the candidate union satisfies, but the fused shortlist doesn't.
4. ``reranker_demotion`` (rerank) — the reranker's full candidate_n input satisfies, but the
                              reranked top_k does not (the reranker pushed gold past the cutoff).
5. ``final_cutoff``          — (no reranker) the pre-final ranking satisfies, but top_k doesn't.
6. ``budget_cutoff`` (budget) — the final top_k satisfies, but the budget-packed subset does
                              not (lost only to the packing policy, not to retrieval).

The DAG is conditional on the config: the reranker stage exists only when a reranker is
configured (and it subsumes the final cutoff for that case, since reranking is what moved the
gold chunk); the budget stage exists only under budget-normalized evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from retrieval_lab.gold import DEFAULT_MIN_GOLD_COVERAGE, GoldAnswer, satisfies_gold
from retrieval_lab.models import Chunk, Config

STAGE_REPRESENTATION = "representation"
STAGE_CANDIDATE_GENERATION = "candidate_generation"
STAGE_FUSION = "fusion"
STAGE_RERANKER_DEMOTION = "reranker_demotion"
STAGE_FINAL_CUTOFF = "final_cutoff"
STAGE_BUDGET_CUTOFF = "budget_cutoff"


@dataclass
class StageOutputs:
    """The intermediate item sets of one config's run on one query — attribution's input.

    Populated by ``pipeline.RetrievalPipeline``. Fields that a config's DAG doesn't use are
    ``None`` (e.g. ``sparse_candidates`` for a dense-only config).
    """

    all_chunks: Sequence[Chunk]
    candidate_union: Sequence[Chunk]
    pre_final: Sequence[Chunk]
    final: Sequence[Chunk]
    dense_candidates: Sequence[Chunk] | None = None
    sparse_candidates: Sequence[Chunk] | None = None
    fused: Sequence[Chunk] | None = None
    reranker_input: Sequence[Chunk] | None = None  # candidate_n shortlist fed to the reranker
    reranked: Sequence[Chunk] | None = None        # reranker output (full, reordered)
    budget_packed: Sequence[Chunk] | None = None   # final after budget packing (delivered set)


@dataclass
class AttributionResult:
    stage: str | None  # None means the query was a hit — no failure to attribute.
    branch_diag: dict | None


def _branch_diag(
    outs: StageOutputs, gold: GoldAnswer, min_gold_coverage: float
) -> dict | None:
    """Per-branch hit diagnostics (dense/sparse), reported never used as a verdict."""
    diag: dict[str, bool] = {}
    if outs.dense_candidates is not None:
        diag["dense"] = satisfies_gold(outs.dense_candidates, gold, min_gold_coverage)
    if outs.sparse_candidates is not None:
        diag["sparse"] = satisfies_gold(outs.sparse_candidates, gold, min_gold_coverage)
    return diag or None


def attribute(
    outs: StageOutputs,
    gold: GoldAnswer,
    config: Config,
    min_gold_coverage: float = DEFAULT_MIN_GOLD_COVERAGE,
) -> AttributionResult:
    """Attribute a query outcome to the earliest failing DAG stage (or ``None`` on a hit)."""

    def sg(items: Sequence[Chunk] | None) -> bool:
        return items is not None and satisfies_gold(items, gold, min_gold_coverage)

    branch = _branch_diag(outs, gold, min_gold_coverage)

    # The delivered context is the budget-packed subset when a budget applies, else top_k.
    delivered = outs.budget_packed if outs.budget_packed is not None else outs.final

    # A hit: nothing to attribute. Branch diagnostics still travel with the result.
    if sg(delivered):
        return AttributionResult(None, branch)

    # 1. Representation — the answer text isn't fully present in the chunk set at all.
    if not sg(outs.all_chunks):
        return AttributionResult(STAGE_REPRESENTATION, branch)

    # 2. Candidate generation — retrieval's raw union dropped it (which branch? see diag).
    if not sg(outs.candidate_union):
        return AttributionResult(STAGE_CANDIDATE_GENERATION, branch)

    # 3. Fusion (hybrid only) — the union had it but the fused shortlist lost it.
    if config.retrieval == "hybrid" and outs.fused is not None and not sg(outs.fused):
        return AttributionResult(STAGE_FUSION, branch)

    # 4. Reranker demotion — the reranker's full candidate_n input had it, but its top_k
    #    output does not (the reranker pushed the gold chunk past the cutoff). Tested against
    #    the FULL shortlist input, not a pre-rerank top-k.
    if config.rerank and outs.reranker_input is not None:
        if sg(outs.reranker_input) and not sg(outs.final):
            return AttributionResult(STAGE_RERANKER_DEMOTION, branch)

    # 5. Final cutoff — (no reranker) ranking placed a gold chunk just past top_k.
    if not sg(outs.final):
        return AttributionResult(STAGE_FINAL_CUTOFF, branch)

    # 6. Budget cutoff — top_k had it, but the budget packing policy dropped it.
    if outs.budget_packed is not None and not sg(outs.budget_packed):
        return AttributionResult(STAGE_BUDGET_CUTOFF, branch)

    return AttributionResult(None, branch)  # defensive; unreachable given the hit check
