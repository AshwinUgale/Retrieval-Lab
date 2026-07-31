"""Phase 0 — data model invariants (spec §I.6)."""

import pytest

from retrieval_lab.models import Chunk, Config, compute_chunk_id


def test_chunk_id_is_stable_and_chunker_relative():
    # Same source span, same chunker -> identical id, stable across calls.
    a = compute_chunk_id("D1", 0, 10, "fixed:100")
    b = compute_chunk_id("D1", 0, 10, "fixed:100")
    assert a == b
    # Same span, different chunker -> different id (chunk identity is chunker-relative).
    assert compute_chunk_id("D1", 0, 10, "recursive:100") != a
    # Different span -> different id.
    assert compute_chunk_id("D1", 0, 11, "fixed:100") != a


def test_chunk_make_computes_id_and_length():
    c = Chunk.make("D1", 5, 20, "some text", chunker_spec="fixed:100")
    assert c.id == compute_chunk_id("D1", 5, 20, "fixed:100")
    assert c.length == 15
    assert c.parent_id is None


def test_chunk_rejects_invalid_span():
    with pytest.raises(ValueError):
        Chunk(id="x", source_id="D1", start=10, end=5, text="")


def test_config_id_is_deterministic_and_distinct():
    c1 = Config(embed_model="e5", chunker="fixed", retrieval="hybrid", rerank="ce", top_k=5)
    c2 = Config(embed_model="e5", chunker="fixed", retrieval="hybrid", rerank="ce", top_k=5)
    c3 = Config(embed_model="bge", chunker="fixed", retrieval="hybrid", rerank="ce", top_k=5)
    assert c1.id == c2.id
    assert c1.id != c3.id
    assert "rerank=ce" in c1.id
    # rerank=None renders explicitly, not as an empty string.
    assert "rerank=none" in Config("e5", "fixed", "dense").id
    assert "index=exact" in c1.id
    hnsw = Config(
        embed_model="e5",
        chunker="fixed",
        retrieval="hybrid",
        rerank="ce",
        top_k=5,
        dense_index="hnsw",
    )
    assert c1.id != hnsw.id
    assert "hnsw_m=16" in hnsw.id and "hnsw_ef=50" in hnsw.id


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"top_k": 5, "candidate_n": 4},
        {"budget_tokens": -1},
        {"dense_index": "unknown"},
    ],
)
def test_config_rejects_invalid_cutoffs_and_index(kwargs):
    with pytest.raises(ValueError):
        Config("e5", "fixed", "dense", **kwargs)


def test_sparse_config_has_no_dense_index():
    config = Config("none", "fixed", "sparse")
    assert config.dense_index == "none"
    assert "index=none" in config.id
