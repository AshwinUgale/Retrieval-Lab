"""Phase 3 — deterministic whole-chunk budget packing (spec §I.10)."""

from retrieval_lab.budget import pack_by_budget
from retrieval_lab.models import Chunk
from retrieval_lab.text import count_tokens


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, source_id="D", start=0, end=len(text), text=text)


def test_packs_whole_chunks_up_to_budget():
    a = _chunk("a", "one two three")     # 3 tokens
    b = _chunk("b", "four five")         # 2 tokens
    c = _chunk("c", "six seven eight")   # 3 tokens
    packed, total = pack_by_budget([a, b, c], budget_tokens=5)
    assert [x.id for x in packed] == ["a", "b"]  # 3 + 2 = 5; c would overflow
    assert total == 5


def test_stops_before_overflow_and_does_not_skip_ahead():
    big = _chunk("big", "w " * 20)       # ~20+ tokens
    small = _chunk("small", "tiny")      # 1 token
    # Even though 'small' would fit, packing stops at the first overflowing chunk.
    packed, total = pack_by_budget([big, small], budget_tokens=5)
    assert packed == []
    assert total == 0


def test_result_is_always_a_prefix():
    chunks = [_chunk(str(i), f"tok{i} more words here") for i in range(5)]
    packed, _ = pack_by_budget(chunks, budget_tokens=10)
    assert [c.id for c in packed] == [c.id for c in chunks[: len(packed)]]


def test_exact_fit_is_admitted():
    a = _chunk("a", "one two")   # 2 tokens
    b = _chunk("b", "three")     # 1 token
    packed, total = pack_by_budget([a, b], budget_tokens=3)
    assert [x.id for x in packed] == ["a", "b"]
    assert total == 3


def test_text_of_counts_parent_text():
    # Parent-child: budget should count the returned parent text, not the indexed child.
    child = _chunk("c", "child")  # 1 token if counted by own text
    parent_text = "this is the much larger parent passage that is returned instead"
    packed, total = pack_by_budget([child], budget_tokens=5, text_of=lambda _c: parent_text)
    assert packed == []  # parent text exceeds the budget
    assert count_tokens(parent_text) > 5
