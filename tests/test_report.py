"""Phase 6 — reporting: JSON round-trip, Pareto frontier, renderers (spec §I.10, §I.12)."""

from retrieval_lab.chunking import FixedSizeChunker
from retrieval_lab.corpora.constructed import build_basic_corpus
from retrieval_lab.embedding import DeterministicEmbedder
from retrieval_lab.report import (
    pareto_frontier,
    render_explain,
    render_html,
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


def test_html_report_is_self_contained(tmp_path):
    from retrieval_lab.report import render_html, write_html

    sweep = _sweep()
    html = render_html(sweep)
    assert html.startswith("<!doctype html>")
    assert "Retrieval Lab" in html and "Pareto" in html
    # Self-contained: no external resource references.
    assert "http://" not in html and "https://" not in html
    assert "<script" not in html
    p = tmp_path / "nested" / "report.html"
    write_html(sweep, p)  # creates parent dirs
    assert p.exists() and p.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_html_report_is_readable_and_well_formed():
    """Regression: the report must not fall back to pipe-string ids, and the Pareto SVG
    attributes must be quoted (an unquoted ``class=x/>`` renders black/invisible)."""
    import re

    sweep = _sweep()
    html = render_html(sweep)
    # Readability: a glossary + colour-coded stage legend, and config chips (not raw ids).
    assert "How to read this" in html
    assert "Final cutoff" in html  # a stage explained in plain English
    assert "|dense|" not in html  # the pipe-string config id is never shown raw
    # The Pareto SVG renders marks, and no attribute has the unquoted-class-slash bug.
    assert "<svg" in html and "<circle" in html
    assert re.search(r'class=[a-z]+/', html) is None
    assert 'class="front"' in html and 'class="dom"' in html


def test_html_report_renders_when_no_cost_and_single_config():
    # Degenerate inputs must not crash the renderer.
    docs, queries = build_basic_corpus()
    from retrieval_lab.chunking import FixedSizeChunker
    from retrieval_lab.embedding import DeterministicEmbedder

    spec = SweepSpec(
        embedders={"det": DeterministicEmbedder(dim=512)},
        chunkers={"fixed": FixedSizeChunker(chunk_size=400)},
        retrieval_modes=("dense",),
        top_k=3, candidate_n=10,
    )
    html = render_html(run_sweep(docs, queries, spec, min_sample=1))
    assert html.startswith("<!doctype html>") and "<svg" in html


def test_html_does_not_declare_winner_below_minimum_sample():
    docs, queries = build_basic_corpus()
    spec = SweepSpec(
        embedders={"det": DeterministicEmbedder(dim=512)},
        chunkers={"fixed": FixedSizeChunker(chunk_size=400)},
        retrieval_modes=("dense",),
    )
    html = render_html(run_sweep(docs, queries, spec, min_sample=len(queries) + 1))

    assert ">BEST<" not in html
    assert "Best on your query set" not in html
    assert "no aggregate winner" in html
