"""Phase 7 — embedding-space geometry lenses (spec §I.5, §I.7)."""

import numpy as np

from retrieval_lab.embedding.base import l2_normalize
from retrieval_lab.geometry import (
    anisotropy,
    cross_model_mismatch,
    geometry_report,
    hubness,
    render_geometry,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_anisotropy_low_for_spread_vectors_high_for_clustered():
    spread = l2_normalize(_rng(0).normal(size=(200, 32)))
    # Clustered: all near a single direction.
    base = _rng(1).normal(size=(1, 32))
    clustered = l2_normalize(base + 0.05 * _rng(2).normal(size=(200, 32)))
    assert anisotropy(spread)["mean_random_cosine"] < 0.2
    assert anisotropy(clustered)["mean_random_cosine"] > 0.8


def test_hubness_skew_higher_with_a_planted_hub():
    v = l2_normalize(_rng(0).normal(size=(100, 16)))
    base_skew = hubness(v, k=5)["skewness"]
    # Plant many near-duplicates of one direction -> that region becomes a hub.
    hub_dir = v[0]
    hubbed = np.vstack([v, l2_normalize(hub_dir + 0.01 * _rng(3).normal(size=(40, 16)))])
    assert hubness(hubbed, k=5)["skewness"] >= base_skew


def test_cross_model_mismatch_detects_disjoint_query_region():
    corpus = l2_normalize(_rng(0).normal(size=(50, 32)))
    aligned = corpus[:10]  # queries drawn from the corpus itself
    disjoint = l2_normalize(_rng(9).normal(size=(10, 32)) + 5.0)  # shifted region
    aligned_nn = cross_model_mismatch(aligned, corpus)["mean_query_to_corpus_nn"]
    disjoint_nn = cross_model_mismatch(disjoint, corpus)["mean_query_to_corpus_nn"]
    assert aligned_nn > disjoint_nn


def test_small_inputs_are_degenerate_not_errors():
    assert hubness(np.zeros((2, 4)))["skewness"] == 0.0
    assert anisotropy(np.zeros((1, 4)))["isotropy"] == 1.0


def test_geometry_report_and_render():
    corpus = l2_normalize(_rng(0).normal(size=(60, 16)))
    queries = corpus[:5]
    report = geometry_report(corpus, queries, k=5)
    assert "hubness" in report.to_dict()
    assert report.cross_model_mismatch is not None
    text = render_geometry(report)
    assert "geometry" in text.lower()
    assert "not verdicts" in text.lower() or "not per-query" in text.lower()
