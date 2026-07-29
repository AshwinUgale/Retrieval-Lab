"""Scorer: turn a ranked chunk list into a ``QueryResult`` (spec §I.6–I.7).

The scorer is thin on purpose — all the hit logic lives in ``gold.satisfies_gold`` and its
helpers, which the attribution engine also calls. The scorer just applies them at the right
cutoffs: hit and single-chunk / fragmentation signals over the returned ``top_k`` context,
completion rank over the full ranked list.
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
) -> QueryResult:
    """Score one query given the chunks a config ranked (deepest-first, length >= top_k).

    ``ranked`` should be the full ranked shortlist (e.g. ``candidate_n`` deep); ``top_k`` is
    the final returned cutoff. Completion rank is measured over ``ranked`` so it can exceed
    ``top_k`` (a gold chunk that ranking placed just past the cutoff), which is exactly the
    signal a later "final cutoff" attribution reads.
    """
    top = list(ranked[:top_k])
    return QueryResult(
        query_id=query.id,
        config_id=config_id,
        hit=satisfies_gold(top, query.gold, min_gold_coverage),
        single_chunk_coverage_by_span=single_chunk_coverage_by_span(top, query.gold),
        fragmented_spans=fragmented_spans(top, query.gold, min_gold_coverage),
        gold_completion_rank=gold_completion_rank(ranked, query.gold, min_gold_coverage),
        retrieved_tokens=sum(count_tokens(c.text) for c in top),
    )
