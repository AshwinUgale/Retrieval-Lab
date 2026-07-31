"""Phase 5 — the grid sweep engine + user-story regression breakdown (spec §I.4, user story)."""

import os

import pytest

from retrieval_lab.attribution import STAGE_FINAL_CUTOFF
from retrieval_lab.chunking import FixedSizeChunker
from retrieval_lab.corpora.constructed import build_basic_corpus
from retrieval_lab.corpora.planted import build_candidate_miss_corpus, build_fragmented_corpus
from retrieval_lab.embedding import DeterministicEmbedder, EmbeddingCache
from retrieval_lab.sweep import SweepSpec, run_sweep


def test_sweep_runs_the_full_grid_and_aggregates():
    docs, queries = build_basic_corpus()
    cache = EmbeddingCache()
    spec = SweepSpec(
        embedders={"det": DeterministicEmbedder(dim=2048, cache=cache)},
        chunkers={
            "fixed400": FixedSizeChunker(chunk_size=400),
            "fixed120": FixedSizeChunker(chunk_size=120, overlap=20),
        },
        retrieval_modes=("dense", "hybrid"),
        top_k=3,
        candidate_n=10,
    )
    result = run_sweep(docs, queries, spec, min_sample=1)

    assert result.n_queries == 4
    assert len(result.metrics) == spec.n_configs() == 4  # 1 embed × 2 chunk × 2 mode
    assert set(result.results_by_config) == {m.config_id for m in result.metrics}
    # Ranking and best selection work.
    best = result.best(by="hit_rate")
    assert best is not None
    assert result.ranked()[0].hit_rate >= result.ranked()[-1].hit_rate


def test_sweep_user_story_regression_breakdown():
    # The spec's user story: a chunking change lifts overall recall but regresses some
    # queries; the tool attributes each regression to a stage. Here a large-chunk config hits
    # the fragmented answer, while a fine-chunk config splits it and drops it at top_k=1.
    docs, frag_query, split = build_fragmented_corpus()
    spec = SweepSpec(
        embedders={"det": DeterministicEmbedder(dim=2048)},
        chunkers={
            "whole": FixedSizeChunker(chunk_size=400),   # answer stays in one chunk
            "split": FixedSizeChunker(chunk_size=split),  # answer fragments across chunks
        },
        retrieval_modes=("dense",),
        top_k=1,
        candidate_n=10,
    )
    result = run_sweep(docs, [frag_query], spec, min_sample=1)

    whole_id = next(c for c in result.results_by_config if "|whole|" in c)
    split_id = next(c for c in result.results_by_config if "|split|" in c)
    whole_res = result.results_by_config[whole_id][0]
    split_res = result.results_by_config[split_id][0]

    assert whole_res.hit and not split_res.hit               # the regression
    assert split_res.stage_attribution == STAGE_FINAL_CUTOFF  # attributed to a stage
    # Paired comparison shows the whole-chunk config strictly better on this query.
    cmp = result.compare(whole_id, split_id, metric="hit")
    assert cmp.mean_diff == 1.0


def test_sweep_flags_baseline_broken_when_all_configs_miss():
    docs, query = build_candidate_miss_corpus()
    spec = SweepSpec(
        embedders={"det": DeterministicEmbedder(dim=2048)},
        chunkers={"c": FixedSizeChunker(chunk_size=200)},
        retrieval_modes=("hybrid",),
        top_k=1,
        candidate_n=1,  # both branches surface only the distractor -> every config misses
    )
    result = run_sweep(docs, [query], spec, min_sample=1)
    assert all(m.hit_rate == 0.0 for m in result.metrics)
    assert result.validity.baseline_broken


def test_shared_cache_is_reused_across_configs():
    docs, queries = build_basic_corpus()
    cache = EmbeddingCache()
    spec = SweepSpec(
        embedders={"det": DeterministicEmbedder(dim=512, cache=cache)},
        chunkers={"c": FixedSizeChunker(chunk_size=400)},
        retrieval_modes=("dense", "hybrid"),  # dense index reused across both modes
        top_k=3,
        candidate_n=10,
    )
    run_sweep(docs, queries, spec, min_sample=1)
    assert len(cache) > 0  # chunk + query embeddings were cached


def test_sparse_only_sweep_does_not_use_or_duplicate_embedders():
    class ExplodingEmbedder:
        def embed_passage(self, _texts):
            raise AssertionError("sparse retrieval must not build a dense index")

    docs, queries = build_basic_corpus()
    spec = SweepSpec(
        embedders={"unused-a": ExplodingEmbedder(), "unused-b": ExplodingEmbedder()},
        chunkers={"c": FixedSizeChunker(chunk_size=400)},
        retrieval_modes=("sparse",),
        top_k=3,
        candidate_n=10,
    )
    result = run_sweep(docs, queries, spec, min_sample=1)

    assert len(result.metrics) == spec.n_configs() == 1
    assert result.metrics[0].config_id.startswith("unused-a|c|sparse|")


def test_best_is_suppressed_below_minimum_sample():
    docs, queries = build_basic_corpus()
    spec = SweepSpec(
        embedders={"det": DeterministicEmbedder(dim=512)},
        chunkers={"c": FixedSizeChunker(chunk_size=400)},
        retrieval_modes=("dense",),
    )
    result = run_sweep(docs, queries, spec, min_sample=len(queries) + 1)

    assert result.validity.verdicts_suppressed
    assert result.best() is None


@pytest.mark.skipif(
    os.environ.get("RLAB_REAL_EMBED") != "1",
    reason="real embedder test is opt-in (needs a model download); set RLAB_REAL_EMBED=1",
)
def test_real_e5_embedder_smoke():  # pragma: no cover - opt-in, needs network
    from retrieval_lab.embedding import e5_embedder
    from retrieval_lab.retrieval import DenseRetriever

    docs, queries = build_basic_corpus()
    chunks = FixedSizeChunker(chunk_size=400).chunk_corpus(docs.values())
    retriever = DenseRetriever(e5_embedder()).index(chunks)
    top = retriever.retrieve(queries[0].text, k=1)
    assert top and "404" in top[0].text
