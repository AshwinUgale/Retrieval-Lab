"""Reciprocal Rank Fusion (RRF) — combine ranked lists by rank, not score (spec §I.7).

Dense and sparse scores live on different, incomparable scales, so they are fused by
*rank*: each chunk accumulates ``1 / (c + rank)`` (1-based rank) across the lists it appears
in, and the fused ranking sorts by that sum. ``c`` (~60) damps the influence of the very top
ranks so a single list cannot dominate. A chunk missing from a list simply contributes
nothing from it.
"""

from __future__ import annotations

from collections.abc import Sequence

from retrieval_lab.models import Chunk

DEFAULT_RRF_C = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Chunk]],
    c: int = DEFAULT_RRF_C,
) -> list[Chunk]:
    """Fuse several ranked chunk lists into one ranking, highest fused score first.

    Ties break deterministically by chunk id so the output is reproducible.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (c + rank)
            first_seen.setdefault(chunk.id, chunk)
    order = sorted(first_seen.values(), key=lambda ch: (-scores[ch.id], ch.id))
    return order
