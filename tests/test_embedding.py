"""Phase 1 — the keyless deterministic embedder + content-addressed cache (spec §I.7)."""

import numpy as np

from retrieval_lab.embedding import DeterministicEmbedder, EmbeddingCache


def cos(a, b):
    return float(a @ b)


def test_deterministic_and_reproducible():
    e1 = DeterministicEmbedder(dim=256)
    e2 = DeterministicEmbedder(dim=256)
    v1 = e1.embed(["the quick brown fox"])
    v2 = e2.embed(["the quick brown fox"])
    assert np.array_equal(v1, v2)  # byte-stable across instances


def test_rows_are_l2_normalized():
    e = DeterministicEmbedder(dim=256)
    mat = e.embed(["some text here", "another different sentence"])
    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_empty_text_is_zero_vector_not_nan():
    e = DeterministicEmbedder(dim=64)
    v = e.embed([""])[0]
    assert np.all(v == 0.0)
    assert not np.any(np.isnan(v))


def test_similar_texts_score_higher_than_dissimilar():
    e = DeterministicEmbedder(dim=1024)
    q = e.embed_one("what is the powerhouse of the cell")
    related = e.embed_one("the mitochondria is the powerhouse of the cell")
    unrelated = e.embed_one("bring a pot of salted water to a boil for pasta")
    assert cos(q, related) > cos(q, unrelated)
    assert cos(q, related) > 0.2


def test_char_ngrams_capture_morphological_overlap():
    # Exact tokens differ ("configure" vs "configuration") but char n-grams overlap, so the
    # keyless embedder still sees them as related — the lever the validation corpus uses to
    # build dense-vs-sparse divergence.
    e = DeterministicEmbedder(dim=1024, use_words=True)
    a = e.embed_one("how to configure the server")
    b = e.embed_one("server configuration settings")
    assert cos(a, b) > 0.05


def test_cache_populates_and_returns_same_vectors():
    cache = EmbeddingCache()
    e = DeterministicEmbedder(dim=128, cache=cache)
    first = e.embed(["alpha", "beta"])
    assert len(cache) == 2
    # Second call is served from cache and is identical.
    second = e.embed(["alpha", "beta"])
    assert np.array_equal(first, second)
    assert len(cache) == 2  # no new entries


def test_cache_keyed_by_model_name():
    cache = EmbeddingCache()
    DeterministicEmbedder(dim=64, name="m1", cache=cache).embed(["x"])
    DeterministicEmbedder(dim=64, name="m2", cache=cache).embed(["x"])
    assert len(cache) == 2  # same text, two models -> two entries


def test_empty_input_returns_empty_matrix():
    e = DeterministicEmbedder(dim=32)
    out = e.embed([])
    assert out.shape == (0, 32)
