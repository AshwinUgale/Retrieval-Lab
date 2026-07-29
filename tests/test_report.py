"""Phase 6 — reporting: JSON round-trip, Pareto frontier, renderers (spec §I.10, §I.12)."""

from retrieval_lab.chunking import FixedSizeChunker
from retrieval_lab.corpora.constructed import build_basic_corpus
from retrieval_lab.embedding import DeterministicEmbedder
from retrieval_lab.report import (
    pareto_frontier,
    render_explain,
    render_pareto,
    render_report,
    sweep_from_dict,
    sweep_to_dict,
    write_json,
)
from retrieval_lab.sweep import SweepSpec, run_sweep


def _sweep():
    docs, queries = build_basic_corpus()
    spec = SweepSpec(
        embedders={"det": DeterministicEmbedder(dim=2048)},
        chunkers={"fixed": FixedSizeChunker(chunk_size=400),
                  "fine": FixedSizeChunker(chunk_size=120, overlap=20)},
        retrieval_modes=("dense", "hybrid"),
        top_k=3, candidate_n=10,
    )
    return run_sweep(docs, queries, spec, min_sample=1)


def test_json_round_trip_preserves_results_and_metrics():
    sweep = _sweep()
    restored = sweep_from_dict(sweep_to_dict(sweep))
    assert restored.n_queries == sweep.n_queries
    assert set(restored.results_by_config) == set(sweep.results_by_config)
    assert len(restored.metrics) == len(sweep.metrics)
    # A concrete metric survives the trip.
    m0, r0 = sweep.metrics[0], restored.metrics_by_config()[sweep.metrics[0].config_id]
    assert r0.hit_rate == m0.hit_rate
    assert r0.hit_rate_ci == m0.hit_rate_ci


def test_write_json_creates_readable_file(tmp_path):
    from retrieval_lab.report import read_json

    sweep = _sweep()
    p = tmp_path / "out.json"
    write_json(sweep, p)
    assert p.exists()
    reloaded = read_json(p)
    assert reloaded.n_queries == sweep.n_queries


def test_pareto_frontier_is_nondominated():
    sweep = _sweep()
    frontier = pareto_frontier(sweep)
    assert frontier
    # No frontier point is dominated by another frontier point.
    for a in frontier:
        for b in frontier:
            if a is b:
                continue
            dominated = (b.hit_rate >= a.hit_rate
                         and b.avg_retrieved_tokens <= a.avg_retrieved_tokens
                         and (b.hit_rate > a.hit_rate
                              or b.avg_retrieved_tokens < a.avg_retrieved_tokens))
            assert not dominated
    # Sorted by hit-rate descending.
    assert frontier == sorted(frontier, key=lambda p: (-p.hit_rate, p.avg_retrieved_tokens))


def test_renderers_produce_meaningful_text():
    sweep = _sweep()
    report = render_report(sweep)
    assert "query set" in report and "hit@k" in report
    pareto = render_pareto(sweep)
    assert "Pareto" in pareto and "tokens" in pareto
    explain = render_explain(sweep, "Q1")
    assert "Q1" in explain
    assert "no query with id" in render_explain(sweep, "NOPE")
