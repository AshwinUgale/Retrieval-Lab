"""Phase 0 — the ground-truth foundation: coverage predicate + offset verification.

These tests pin the two decisions the tool lives or dies on (spec §I.7–I.8): union-coverage
semantics (fragmentation is real, mere overlap is not a hit) and fail-closed offset
verification.
"""

import pytest

from retrieval_lab.gold import (
    DEFAULT_MIN_GOLD_COVERAGE,
    EvidenceSet,
    GoldAnswer,
    GoldSpan,
    OffsetVerificationError,
    coverage,
    fragmented_spans,
    gold_completion_rank,
    load_queries,
    satisfies_gold,
    single_chunk_coverage_by_span,
    verify_query,
)
from retrieval_lab.hashing import content_hash
from retrieval_lab.models import Chunk, Document


def chunk(source_id: str, start: int, end: int) -> Chunk:
    """A chunk with arbitrary text — coverage math only reads the source offsets."""
    return Chunk.make(source_id, start, end, text="x" * (end - start), chunker_spec="test")


def span(source_id: str, start: int, end: int, doc_text: str | None = None) -> GoldSpan:
    quoted = doc_text[start:end] if doc_text is not None else "?" * (end - start)
    return GoldSpan(source_id=source_id, start=start, end=end, quoted_text=quoted)


def gold_of(*spans: GoldSpan) -> GoldAnswer:
    """A single-alternative gold requiring all given spans."""
    return GoldAnswer((EvidenceSet(tuple(spans)),))


# --------------------------------------------------------------------------------------
# Coverage semantics
# --------------------------------------------------------------------------------------


def test_full_coverage_by_one_chunk():
    s = span("D1", 10, 20)
    assert coverage([chunk("D1", 0, 100)], s) == 1.0


def test_partial_coverage_is_fraction():
    s = span("D1", 0, 10)
    # A chunk covering [0, 8) covers 8/10 of the span.
    assert coverage([chunk("D1", 0, 8)], s) == pytest.approx(0.8)


def test_chunk_from_other_document_contributes_nothing():
    s = span("D1", 0, 10)
    assert coverage([chunk("D2", 0, 100)], s) == 0.0


def test_union_of_two_half_covering_chunks_satisfies():
    # THE load-bearing case (spec §I.7): two chunks each covering half a span jointly
    # satisfy it. The predicate must see this identically everywhere.
    s = span("D1", 0, 10)
    items = [chunk("D1", 0, 5), chunk("D1", 5, 10)]
    assert coverage(items, s) == 1.0
    assert satisfies_gold(items, gold_of(s))


def test_overlapping_chunks_are_not_double_counted():
    s = span("D1", 0, 10)
    # [0,7) and [3,10) overlap on [3,7); union is the whole span, not 140%.
    assert coverage([chunk("D1", 0, 7), chunk("D1", 3, 10)], s) == 1.0


def test_mere_overlap_below_threshold_is_not_a_hit():
    # An "any overlap" rule would call this a hit; coverage-based gold does not (spec §I.7).
    s = span("D1", 0, 100)
    items = [chunk("D1", 0, 5)]  # 5% coverage
    assert not satisfies_gold(items, gold_of(s))


# --------------------------------------------------------------------------------------
# EvidenceSet (conjunction) and GoldAnswer (disjunction)
# --------------------------------------------------------------------------------------


def test_evidence_set_requires_all_spans():
    s1, s2 = span("D1", 0, 10), span("D1", 50, 60)
    g = gold_of(s1, s2)
    assert not satisfies_gold([chunk("D1", 0, 10)], g)  # only s1
    assert satisfies_gold([chunk("D1", 0, 10), chunk("D1", 50, 60)], g)  # both


def test_any_alternative_suffices():
    alt1 = EvidenceSet((span("D1", 0, 10),))
    alt2 = EvidenceSet((span("D2", 0, 10),))
    g = GoldAnswer((alt1, alt2))
    assert satisfies_gold([chunk("D2", 0, 10)], g)  # second alternative alone


def test_threshold_boundary_is_inclusive():
    s = span("D1", 0, 10)
    at_threshold = [chunk("D1", 0, 8)]  # exactly 0.8
    assert coverage(at_threshold, s) == pytest.approx(DEFAULT_MIN_GOLD_COVERAGE)
    assert satisfies_gold(at_threshold, gold_of(s))
    below = [chunk("D1", 0, 7)]  # 0.7
    assert not satisfies_gold(below, gold_of(s))


# --------------------------------------------------------------------------------------
# Completion rank
# --------------------------------------------------------------------------------------


def test_completion_rank_is_smallest_prefix_that_satisfies():
    s = span("D1", 0, 10)
    ranked = [chunk("D2", 0, 10), chunk("D1", 0, 5), chunk("D1", 5, 10)]
    # Needs both halves -> first satisfied at the 3-chunk prefix.
    assert gold_completion_rank(ranked, gold_of(s)) == 3


def test_completion_rank_none_when_never_satisfied():
    s = span("D1", 0, 100)
    ranked = [chunk("D1", 0, 5), chunk("D2", 0, 100)]
    assert gold_completion_rank(ranked, gold_of(s)) is None


# --------------------------------------------------------------------------------------
# Fragmentation signal
# --------------------------------------------------------------------------------------


def test_fragmentation_signal_when_only_union_satisfies():
    s = span("D1", 0, 10)
    items = [chunk("D1", 0, 5), chunk("D1", 5, 10)]  # each covers 0.5 alone
    single = single_chunk_coverage_by_span(items, gold_of(s))
    assert single[s.id] == pytest.approx(0.5)
    assert fragmented_spans(items, gold_of(s)) == [s.id]


def test_no_fragmentation_when_a_single_chunk_satisfies():
    s = span("D1", 0, 10)
    items = [chunk("D1", 0, 10), chunk("D1", 0, 5)]
    assert single_chunk_coverage_by_span(items, gold_of(s))[s.id] == 1.0
    assert fragmented_spans(items, gold_of(s)) == []


# --------------------------------------------------------------------------------------
# Offset verification — fail closed (spec §I.8, §I.11)
# --------------------------------------------------------------------------------------


def _doc(text: str) -> Document:
    return Document(id="D1", text=text)


def _query_with_span(s: GoldSpan):
    from retrieval_lab.gold import Query

    return Query(id="Q1", text="q", gold=gold_of(s))


def test_verify_accepts_matching_quoted_text_and_version():
    text = "the quick brown fox"
    doc = _doc(text)
    s = GoldSpan("D1", 4, 9, quoted_text="quick", source_version=content_hash(text))
    verify_query(_query_with_span(s), {"D1": doc})  # no raise


def test_verify_rejects_quoted_text_mismatch():
    doc = _doc("the quick brown fox")
    s = GoldSpan("D1", 4, 9, quoted_text="slow!")  # wrong text at those offsets
    with pytest.raises(OffsetVerificationError, match="quoted_text mismatch"):
        verify_query(_query_with_span(s), {"D1": doc})


def test_verify_rejects_stale_source_version():
    doc = _doc("the quick brown fox")
    s = GoldSpan("D1", 4, 9, quoted_text="quick", source_version="deadbeef" * 8)
    with pytest.raises(OffsetVerificationError, match="changed since labeling"):
        verify_query(_query_with_span(s), {"D1": doc})


def test_verify_rejects_unknown_source():
    s = GoldSpan("MISSING", 0, 3, quoted_text="abc")
    with pytest.raises(OffsetVerificationError, match="unknown source"):
        verify_query(_query_with_span(s), {"D1": _doc("abc")})


def test_verify_rejects_out_of_range_offsets():
    doc = _doc("short")
    s = GoldSpan("D1", 0, 999, quoted_text="short" + "?" * 994)
    with pytest.raises(OffsetVerificationError, match="out of range"):
        verify_query(_query_with_span(s), {"D1": doc})


# --------------------------------------------------------------------------------------
# JSONL loading round-trip
# --------------------------------------------------------------------------------------


def test_load_queries_strict_raises_on_drift(tmp_path):
    import json

    docs = {"D1": _doc("the quick brown fox")}
    bad = {
        "id": "Q1",
        "text": "q",
        "gold": {"alternatives": [{"required_spans": [
            {"source_id": "D1", "start": 4, "end": 9, "quoted_text": "WRONG"}
        ]}]},
    }
    p = tmp_path / "queries.jsonl"
    p.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(OffsetVerificationError):
        load_queries(p, docs, strict=True)
    # Non-strict skips the drifted query instead of raising.
    assert load_queries(p, docs, strict=False) == []


def test_load_queries_accepts_valid_gold(tmp_path):
    import json

    text = "the quick brown fox"
    docs = {"D1": _doc(text)}
    good = {
        "id": "Q1",
        "text": "who is quick?",
        "gold": {"alternatives": [{"required_spans": [
            {"source_id": "D1", "start": 4, "end": 9, "quoted_text": "quick",
             "source_version": content_hash(text)}
        ]}]},
    }
    p = tmp_path / "queries.jsonl"
    p.write_text(json.dumps(good) + "\n", encoding="utf-8")
    queries = load_queries(p, docs, strict=True)
    assert len(queries) == 1
    assert queries[0].id == "Q1"
    assert satisfies_gold([chunk("D1", 0, 19)], queries[0].gold)
