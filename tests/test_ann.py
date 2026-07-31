"""Phase 7 — ANN option + ANN-vs-exact recall diagnostic (spec §I.7)."""

import importlib.util

import pytest

import retrieval_lab.sweep as sweep_module
from retrieval_lab.chunking import FixedSizeChunker
from retrieval_lab.corpora.constructed import build_basic_corpus
from retrieval_lab.embedding import DeterministicEmbedder
from retrieval_lab.report import render_html, sweep_from_dict, sweep_to_dict
from retrieval_lab.retrieval import DenseRetriever, ann_vs_exact_recall
from retrieval_lab.sweep import SweepSpec, run_sweep

HAVE_HNSWLIB = importlib.util.find_spec("hnswlib") is not None


class _DroppingRetriever:
    """A stub 'approximate' retriever that drops the last of the exact results.

    Simulates ANN recall loss deterministically so the diagnostic is testable without
    hnswlib.
    """

    def __init__(self, exact: DenseRetriever, drop: int = 1) -> None:
        self.exact = exact
        self.drop = drop

    def retrieve(self, query: str, k: int):
        hits = self.exact.retrieve(query, k)
        return hits[: max(0, len(hits) - self.drop)]


def _indexed_exact():
    docs, queries = build_basic_corpus()
    chunks = FixedSizeChunker(chunk_size=120, overlap=20).chunk_corpus(docs.values())
    exact = DenseRetriever(DeterministicEmbedder(dim=2048)).index(chunks)
    return exact, [q.text for q in queries]


def test_recall_is_one_when_ann_matches_exact():
    exact, queries = _indexed_exact()
    diag = ann_vs_exact_recall(exact, exact, queries, k=3)
    assert diag["mean_recall"] == 1.0
    assert diag["n"] == len(queries)


def test_recall_drops_when_ann_loses_results():
    exact, queries = _indexed_exact()
    lossy = _DroppingRetriever(exact, drop=1)
    diag = ann_vs_exact_recall(lossy, exact, queries, k=3)
    assert diag["mean_recall"] < 1.0  # the approximation lost some exact top-k members


def test_empty_queries_are_degenerate():
    exact, _ = _indexed_exact()
    assert ann_vs_exact_recall(exact, exact, [], k=3)["mean_recall"] == 1.0


def test_hnsw_is_wired_through_sweep_and_diagnostics(monkeypatch):
    class FakeANN(DenseRetriever):
        def __init__(self, embedder, **_kwargs):
            super().__init__(embedder)

        @property
        def index_nbytes(self):
            return 1234

    monkeypatch.setattr(sweep_module, "ANNDenseRetriever", FakeANN)
    docs, queries = build_basic_corpus()
    spec = SweepSpec(
        embedders={"det": DeterministicEmbedder(dim=256)},
        chunkers={"fixed": FixedSizeChunker(chunk_size=120, overlap=20)},
        retrieval_modes=("dense", "hybrid"),
        dense_indexes=("exact", "hnsw"),
        top_k=3,
        candidate_n=5,
    )
    sweep = run_sweep(docs, queries, spec, min_sample=1, measure_latency=True)

    assert len(sweep.metrics) == spec.n_configs() == 4
    hnsw_ids = [cid for cid in sweep.results_by_config if "|index=hnsw" in cid]
    assert len(hnsw_ids) == 2
    assert set(sweep.ann_diagnostics) == set(hnsw_ids)
    assert all(sweep.ann_diagnostics[cid].mean_recall == 1.0 for cid in hnsw_ids)
    assert all(sweep.cost[cid].index_bytes == 1234 for cid in hnsw_ids)
    restored = sweep_from_dict(sweep_to_dict(sweep))
    assert restored.ann_diagnostics == sweep.ann_diagnostics
    html = render_html(restored)
    assert "HNSW approximation check" in html
    assert "mean candidate recall@5" in html
    assert 'data-filter="index"' in html


def test_hnsw_rejects_search_width_below_candidate_count():
    docs, queries = build_basic_corpus()
    spec = SweepSpec(
        embedders={"det": DeterministicEmbedder(dim=64)},
        chunkers={"fixed": FixedSizeChunker(chunk_size=120)},
        retrieval_modes=("dense",),
        dense_indexes=("hnsw",),
        candidate_n=20,
        hnsw_ef=10,
    )
    with pytest.raises(ValueError, match="hnsw_ef"):
        run_sweep(docs, queries, spec, min_sample=1)


@pytest.mark.skipif(not HAVE_HNSWLIB, reason="hnswlib not installed ([ann] extra)")
def test_hnsw_retriever_recovers_most_exact_neighbours():  # pragma: no cover - needs extra
    from retrieval_lab.retrieval import ANNDenseRetriever

    docs, queries = build_basic_corpus()
    chunks = FixedSizeChunker(chunk_size=120, overlap=20).chunk_corpus(docs.values())
    emb = DeterministicEmbedder(dim=2048)
    exact = DenseRetriever(emb).index(chunks)
    ann = ANNDenseRetriever(emb, ef=64).index(chunks)
    diag = ann_vs_exact_recall(ann, exact, [q.text for q in queries], k=3)
    assert diag["mean_recall"] >= 0.6  # small corpus: HNSW should recover most neighbours
