"""Post-build #4 — opt-in latency/cost measurement (spec §I.10).

Wall-clock values are non-deterministic and environment-specific, so these tests assert
*structure and ordering* (present, sane, disclosed) — never absolute timings — and that the
default deterministic path is unaffected.
"""

from retrieval_lab.chunking import FixedSizeChunker
from retrieval_lab.corpora.constructed import build_basic_corpus
from retrieval_lab.embedding import DeterministicEmbedder
from retrieval_lab.metrics import latency_stats
from retrieval_lab.report import render_html, render_report, sweep_from_dict, sweep_to_dict
from retrieval_lab.sweep import SweepSpec, run_sweep


def _spec():
    return SweepSpec(
        embedders={"det": DeterministicEmbedder(dim=1024)},
        chunkers={"fixed": FixedSizeChunker(chunk_size=400)},
        retrieval_modes=("dense", "sparse", "hybrid"),
        top_k=3, candidate_n=10,
    )


def test_latency_stats_orders_p50_le_p95():
    stats = latency_stats(
        [1.0, 2.0, 3.0, 10.0, 4.0], index_bytes=2048, build_ms=12.5
    )
    assert stats.n == 5
    assert stats.p50_ms <= stats.p95_ms
    assert stats.index_bytes == 2048
    assert stats.build_ms == 12.5


def test_latency_stats_empty_is_degenerate():
    stats = latency_stats([])
    assert stats.n == 0 and stats.p50_ms == 0.0 and stats.p95_ms == 0.0


def test_sweep_without_measure_has_no_cost():
    docs, queries = build_basic_corpus()
    sweep = run_sweep(docs, queries, _spec(), min_sample=1)
    assert sweep.cost is None
    assert "ENVIRONMENT-SPECIFIC" not in render_report(sweep)


def test_sweep_with_measure_populates_cost_for_every_config():
    docs, queries = build_basic_corpus()
    sweep = run_sweep(docs, queries, _spec(), min_sample=1, measure_latency=True)
    assert sweep.cost is not None
    assert set(sweep.cost) == set(sweep.results_by_config)
    for c in sweep.cost.values():
        assert c.n == len(queries)
        assert c.p50_ms <= c.p95_ms
        assert c.mean_ms >= 0.0
        assert c.build_ms >= 0.0


def test_dense_configs_report_index_bytes_sparse_does_not():
    docs, queries = build_basic_corpus()
    sweep = run_sweep(docs, queries, _spec(), min_sample=1, measure_latency=True)
    dense_id = next(k for k in sweep.cost if "|dense|" in k)
    sparse_id = next(k for k in sweep.cost if "|sparse|" in k)
    assert sweep.cost[dense_id].index_bytes > 0   # the stored vector matrix
    assert sweep.cost[sparse_id].index_bytes == 0  # BM25 postings not sized here


def test_report_discloses_environment_specificity_when_measured():
    docs, queries = build_basic_corpus()
    sweep = run_sweep(docs, queries, _spec(), min_sample=1, measure_latency=True)
    report = render_report(sweep)
    assert "ENVIRONMENT-SPECIFIC" in report
    assert "p50 ms" in report and "p95 ms" in report


def test_html_explains_warm_latency_build_time_and_index_size():
    docs, queries = build_basic_corpus()
    sweep = run_sweep(docs, queries, _spec(), min_sample=1, measure_latency=True)
    html = render_html(sweep)
    assert "Runtime and index cost" in html
    assert "Warm p50/p95" in html
    assert "build ms" in html
    assert "not total process memory" in html


def test_cost_round_trips_through_json():
    docs, queries = build_basic_corpus()
    sweep = run_sweep(docs, queries, _spec(), min_sample=1, measure_latency=True)
    restored = sweep_from_dict(sweep_to_dict(sweep))
    assert restored.cost is not None
    assert set(restored.cost) == set(sweep.cost)
    a_id = next(iter(sweep.cost))
    assert restored.cost[a_id].index_bytes == sweep.cost[a_id].index_bytes
    assert restored.cost[a_id].build_ms == sweep.cost[a_id].build_ms
