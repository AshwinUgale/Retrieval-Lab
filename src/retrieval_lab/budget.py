"""Deterministic retrieved-token budget packing (spec §I.10).

Comparing configs only at ``top_k`` is unfair — five 1,500-token chunks are not five
200-token chunks in context, latency, or cost. So configs are also compared at fixed
retrieved-token budgets, and the packing policy must be exact and deterministic:

- process chunks in ranked order;
- include only **whole** chunks (no truncation — a partial chunk changes the retrievable
  unit and breaks span-coverage math);
- **stop before** the first chunk that would exceed the budget;
- **never skip** an oversized chunk to admit a lower-ranked smaller one;
- report both the configured budget and the actual retrieved-token count.

For parent-child retrieval the budget counts the **returned parent text**, supplied via
``text_of`` (defaults to the chunk's own text).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from retrieval_lab.models import Chunk
from retrieval_lab.text import count_tokens


def pack_by_budget(
    ranked: Sequence[Chunk],
    budget_tokens: int,
    text_of: Callable[[Chunk], str] | None = None,
) -> tuple[list[Chunk], int]:
    """Pack whole chunks in ranked order up to ``budget_tokens``.

    Returns ``(packed_chunks, actual_tokens)``. Stops at the first chunk that would overflow
    and does not look further, so the result is always a prefix of ``ranked``.
    """
    get_text = text_of or (lambda c: c.text)
    packed: list[Chunk] = []
    total = 0
    for chunk in ranked:
        t = count_tokens(get_text(chunk))
        if total + t > budget_tokens:
            break  # stop before overflow; never skip ahead to a smaller chunk
        packed.append(chunk)
        total += t
    return packed, total
