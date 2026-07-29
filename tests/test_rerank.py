"""Phase 3 — reranker reorders a shortlist (spec §I.7)."""

from retrieval_lab.models import Chunk
from retrieval_lab.retrieval import LexicalReranker


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
    assert ranked[0].id == "all"      # shares reset/the/password
    assert ranked[-1].id == "none"    # shares nothing


def test_lexical_reranker_is_stable_on_ties():
    q = "alpha"
    chunks = [_chunk("c1", "beta gamma"), _chunk("c2", "delta epsilon")]  # both score 0
    ranked = LexicalReranker().rerank(q, chunks)
    assert [c.id for c in ranked] == ["c1", "c2"]  # original order preserved


def test_lexical_reranker_empty_input():
    assert LexicalReranker().rerank("q", []) == []
