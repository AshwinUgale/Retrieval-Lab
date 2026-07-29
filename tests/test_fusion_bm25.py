"""Phase 2 — BM25 sparse retrieval and RRF fusion (spec §I.7)."""

from retrieval_lab.chunking import FixedSizeChunker
from retrieval_lab.corpora.constructed import build_basic_corpus
from retrieval_lab.models import Chunk
from retrieval_lab.retrieval import BM25Retriever, reciprocal_rank_fusion


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, source_id="D", start=0, end=len(text), text=text)


# --------------------------------- BM25 ----------------------------------------------


def test_bm25_ranks_exact_term_match_first():
    chunks = FixedSizeChunker(chunk_size=200).chunk_corpus(build_basic_corpus()[0].values())
    bm25 = BM25Retriever().index(chunks)
    top = bm25.retrieve("404 status code not found", k=1)
    assert "404" in top[0].text


def test_bm25_rare_term_outweighs_common_term():
    docs = [
        _chunk("c1", "the the the the common common words appear frequently here"),
        _chunk("c2", "a passage mentioning the rare token zqrx once among common words"),
    ]
    bm25 = BM25Retriever().index(docs)
    # A query with a rare term should surface the doc that actually contains it.
    top = bm25.retrieve("zqrx", k=2)
    assert top[0].id == "c2"


def test_bm25_excludes_zero_score_and_empty_query():
    docs = [_chunk("c1", "alpha beta"), _chunk("c2", "gamma delta")]
    bm25 = BM25Retriever().index(docs)
    assert bm25.retrieve("nonexistentterm", k=5) == []
    assert bm25.retrieve("", k=5) == []
    hits = bm25.retrieve("alpha", k=5)
    assert [c.id for c in hits] == ["c1"]  # only the matching doc, not both


def test_bm25_deterministic_tie_break_by_corpus_order():
    docs = [_chunk("c1", "same words here"), _chunk("c2", "same words here")]
    bm25 = BM25Retriever().index(docs)
    top = bm25.retrieve("same words here", k=2)
    assert [c.id for c in top] == ["c1", "c2"]


# --------------------------------- RRF -----------------------------------------------


def test_rrf_rewards_appearing_in_multiple_lists():
    a, b, c = _chunk("a", "a"), _chunk("b", "b"), _chunk("c", "c")
    dense = [a, b, c]
    sparse = [b, a, c]
    fused = reciprocal_rank_fusion([dense, sparse])
    # 'b' is rank1+rank2, 'a' is rank2+rank1 -> tie on score; both above 'c' (rank3+rank3).
    assert fused[-1].id == "c"
    assert {fused[0].id, fused[1].id} == {"a", "b"}


def test_rrf_consensus_top_beats_split():
    a, b, c, d = (_chunk(x, x) for x in "abcd")
    # 'a' is #1 in both lists -> highest fused score.
    fused = reciprocal_rank_fusion([[a, b, c], [a, d, b]])
    assert fused[0].id == "a"


def test_rrf_is_deterministic_across_calls():
    a, b = _chunk("a", "a"), _chunk("b", "b")
    r1 = reciprocal_rank_fusion([[a, b], [b, a]])
    r2 = reciprocal_rank_fusion([[a, b], [b, a]])
    assert [c.id for c in r1] == [c.id for c in r2]
