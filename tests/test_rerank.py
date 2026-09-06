"""Phase 3 — reranker reorders a shortlist (spec §I.7)."""

import pytest

from retrieval_lab.models import Chunk
from retrieval_lab.retrieval import LexicalReranker, MMRReranker


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, source_id="D", start=0, end=len(text), text=text)


def test_lexical_reranker_orders_by_shared_query_terms():
    q = "reset the password"
    chunks = [
        _chunk("none", "completely unrelated content about gardening"),
        _chunk("all", "reset the password now please"),
        _chunk("some", "password recovery instructions"),
    ]
    ranked = LexicalReranker().rerank(q, chunks)
    assert ranked[0].id == "all"  # shares reset/the/password
    assert ranked[-1].id == "none"  # shares nothing


def test_lexical_reranker_is_stable_on_ties():
    q = "alpha"
    chunks = [_chunk("c1", "beta gamma"), _chunk("c2", "delta epsilon")]  # both score 0
    ranked = LexicalReranker().rerank(q, chunks)
    assert [c.id for c in ranked] == ["c1", "c2"]  # original order preserved


def test_lexical_reranker_empty_input():
    assert LexicalReranker().rerank("q", []) == []


def test_mmr_reranker_diversifies_near_duplicate_candidates():
    chunks = [
        _chunk("first", "python retrieval indexing tutorial"),
        _chunk("duplicate", "python retrieval indexing tutorial guide"),
        _chunk("diverse", "python reranking evaluation benchmark"),
    ]

    ranked = MMRReranker(lambda_=0.5).rerank("python retrieval reranking", chunks)

    assert [chunk.id for chunk in ranked] == ["first", "diverse", "duplicate"]


def test_mmr_reranker_is_deterministic_and_stable_on_ties():
    chunks = [_chunk("first", "alpha"), _chunk("second", "alpha")]
    reranker = MMRReranker()

    assert [chunk.id for chunk in reranker.rerank("alpha", chunks)] == ["first", "second"]
    assert reranker.rerank("alpha", chunks) == reranker.rerank("alpha", chunks)


def test_mmr_reranker_validates_lambda_and_handles_empty_input():
    assert MMRReranker().rerank("q", []) == []
    with pytest.raises(ValueError, match="between 0 and 1"):
        MMRReranker(lambda_=1.1)
