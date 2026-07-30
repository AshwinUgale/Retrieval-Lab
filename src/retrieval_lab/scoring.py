"""Scorer: turn a ranked chunk list into a ``QueryResult`` (spec §I.6–I.7).

The scorer is thin on purpose — all the hit logic lives in ``gold.satisfies_gold`` and its
helpers, which the attribution engine also calls. The scorer just applies them at the right
cutoffs: hit and single-chunk / fragmentation signals over the **delivered** context (the
returned ``top_k``, or the budget-packed subset when a budget applies), and completion rank
over the full ranked list.
"""

from __future__ import annotations

from collections.abc import Sequence

from retrieval_lab.gold import (
    DEFAULT_MIN_GOLD_COVERAGE,
    Query,
    fragmented_spans,
    gold_completion_rank,
    satisfies_gold,
    single_chunk_coverage_by_span,
)
from retrieval_lab.models import Chunk, QueryResult
from retrieval_lab.text import count_tokens


def score_query(
    query: Query,
    ranked: Sequence[Chunk],
    config_id: str,
    top_k: int,
    min_gold_coverage: float = DEFAULT_MIN_GOLD_COVERAGE,
    final: Sequence[Chunk] | None = None,
) -> QueryResult:
    """Score one query given the chunks a config ranked (deepest-first, length >= top_k).

    ``ranked`` is the full ranked shortlist (e.g. ``candidate_n`` deep); completion rank is
    measured over it so it can exceed ``top_k``. The **delivered context** — over which hit,
    coverage, fragmentation, and retrieved-token count are computed — is ``final`` when given
    (e.g. a budget-packed subset), otherwise ``ranked[:top_k]``.
    """
    delivered = list(final) if final is not None else list(ranked[:top_k])
    return QueryResult(
        query_id=query.id,
        config_id=config_id,
        hit=satisfies_gold(delivered, query.gold, min_gold_coverage),
        single_chunk_coverage_by_span=single_chunk_coverage_by_span(delivered, query.gold),
        fragmented_spans=fragmented_spans(delivered, query.gold, min_gold_coverage),
        gold_completion_rank=gold_completion_rank(ranked, query.gold, min_gold_coverage),
        retrieved_tokens=sum(count_tokens(c.text) for c in delivered),
    )
