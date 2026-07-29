"""Phase 4 — metrics, confidence intervals, and validity gates (spec §I.9, §I.11)."""

import pytest

from retrieval_lab.metrics import (
    aggregate_config,
    bootstrap_mean_ci,
    compare_configs,
    paired_bootstrap_diff,
    reciprocal_rank,
    validity_report,
    wilson_interval,
)
from retrieval_lab.models import QueryResult


def _result(qid: str, hit: bool, rank: int | None, config_id: str = "cfg",
            stage: str | None = None, fragmented: bool = False) -> QueryResult:
    return QueryResult(
        query_id=qid, config_id=config_id, hit=hit,
        gold_completion_rank=rank, stage_attribution=stage,
        fragmented_spans=["s"] if fragmented else [],
    )


# --------------------------------- Wilson ---------------------------------------------


def test_wilson_bounds_are_ordered_and_contain_estimate():
    lo, hi = wilson_interval(8, 10)
    assert 0.0 <= lo < 0.8 < hi <= 1.0


def test_wilson_zero_n_is_degenerate():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_all_success_upper_is_one_area():
    lo, hi = wilson_interval(10, 10)
    assert hi == pytest.approx(1.0, abs=1e-9)
    assert lo < 1.0  # Wilson does not collapse to a point at the boundary


# --------------------------------- Bootstrap ------------------------------------------


def test_bootstrap_ci_contains_mean_and_is_seed_reproducible():
    vals = [0.0, 0.5, 1.0, 0.25, 0.75, 1.0, 0.0, 0.5]
    lo, hi = bootstrap_mean_ci(vals, seed=0)
    mean = sum(vals) / len(vals)
    assert lo <= mean <= hi
    assert bootstrap_mean_ci(vals, seed=0) == (lo, hi)  # reproducible


def test_bootstrap_empty_is_degenerate():
    assert bootstrap_mean_ci([]) == (0.0, 0.0)


# --------------------------------- Paired comparison ----------------------------------


def test_paired_identical_arms_not_significant():
    a = [1.0, 0.0, 1.0, 1.0, 0.0]
    cmp = paired_bootstrap_diff(a, a, seed=0)
    assert cmp.mean_diff == 0.0
    assert not cmp.significant


def test_paired_clear_winner_is_significant():
    # A wins on every query -> difference is consistently positive.
    a = [1.0] * 40
    b = [0.0] * 40
    cmp = paired_bootstrap_diff(a, b, seed=0)
    assert cmp.mean_diff == 1.0
    assert cmp.significant
    assert cmp.ci[0] > 0


def test_paired_requires_aligned_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1.0, 0.0], [1.0], seed=0)


# --------------------------------- Aggregation ----------------------------------------


def test_reciprocal_rank_from_completion_rank():
    assert reciprocal_rank(_result("q", True, 1)) == 1.0
    assert reciprocal_rank(_result("q", True, 4)) == 0.25
    assert reciprocal_rank(_result("q", False, None)) == 0.0


def test_aggregate_reports_hit_rate_and_mrr():
    results = [_result(f"q{i}", hit=(i < 15), rank=(1 if i < 15 else None))
               for i in range(20)]
    m = aggregate_config(results, min_sample=20)
    assert m.n == 20
    assert m.hit_rate == pytest.approx(0.75)
    assert m.mrr == pytest.approx(0.75)
    assert m.hit_rate_ci is not None and m.mrr_ci is not None
    assert not m.verdict_suppressed


def test_aggregate_suppresses_verdict_below_min_sample():
    results = [_result(f"q{i}", hit=True, rank=1) for i in range(5)]
    m = aggregate_config(results, min_sample=20)
    assert m.verdict_suppressed
    assert m.hit_rate_ci is None and m.mrr_ci is None
    assert m.hit_rate == 1.0  # point estimate still reported
    assert any("min_sample" in n for n in m.notes)


def test_aggregate_stage_breakdown_and_fragmentation_counts():
    results = [
        _result("q1", hit=False, rank=None, stage="final_cutoff"),
        _result("q2", hit=False, rank=None, stage="final_cutoff"),
        _result("q3", hit=False, rank=None, stage="fusion"),
        _result("q4", hit=True, rank=2, fragmented=True),
        _result("q5", hit=True, rank=1),
    ]
    m = aggregate_config(results, min_sample=1)
    assert m.stage_breakdown == {"final_cutoff": 2, "fusion": 1}
    assert m.fragmented_queries == 1


def test_aggregate_ignores_refused_queries():
    good = _result("q1", hit=True, rank=1)
    refused = QueryResult(query_id="q2", config_id="cfg", hit=False, refused=True)
    m = aggregate_config([good, refused], min_sample=1)
    assert m.n == 1


def test_compare_configs_aligns_by_query_id():
    a = [_result("q1", True, 1, "A"), _result("q2", True, 1, "A"), _result("q3", False, None, "A")]
    b = [_result("q3", False, None, "B"), _result("q1", False, None, "B"),
         _result("q2", False, None, "B")]
    cmp = compare_configs(a, b, metric="hit", seed=0)
    assert cmp.n == 3
    assert cmp.mean_diff == pytest.approx(2 / 3)  # A hits 2 more of 3


# --------------------------------- Validity gates -------------------------------------


def test_validity_flags_baseline_broken():
    zero = [_result(f"q{i}", hit=False, rank=None) for i in range(20)]
    m = aggregate_config(zero, config_id="cfg", min_sample=20)
    report = validity_report([m])
    assert report.baseline_broken
    assert any("baseline broken" in n for n in report.notes)


def test_validity_flags_suppressed_and_blackbox():
    small = aggregate_config([_result("q", True, 1)], min_sample=20)
    report = validity_report([small], attribution_available=False)
    assert report.verdicts_suppressed
    assert not report.attribution_available
    assert any("black-box" in n for n in report.notes)


def test_validity_clean_when_healthy():
    healthy = aggregate_config(
        [_result(f"q{i}", hit=(i % 2 == 0), rank=(1 if i % 2 == 0 else None)) for i in range(20)],
        min_sample=20,
    )
    report = validity_report([healthy])
    assert not report.baseline_broken
    assert not report.verdicts_suppressed
    assert report.attribution_available
