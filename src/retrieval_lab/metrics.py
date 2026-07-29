"""Metrics, confidence intervals, and validity gates (spec §I.9, §I.11).

The overriding rule (spec §0): a diagnostic that emits a confident number without a
trustworthy reference is worse than no tool. So every aggregate carries ``n`` and a
metric-appropriate confidence interval, and below a minimum-sample gate **no aggregate
verdict is emitted at all** — the tool degrades to raw per-query evidence.

Metric-appropriate intervals (spec §I.9):
- **Recall@k / hit-rate** are proportions → **Wilson** score intervals (well-behaved at
  small n and near 0/1, unlike the normal approximation).
- **MRR** is a mean of reciprocal ranks, not a proportion → **bootstrap** interval. MRR uses
  ``1 / gold_completion_rank`` (the rank at which the top-r chunks first satisfy an
  EvidenceSet), which is well-defined under multi-chunk / multi-span gold.
- **Comparing two configs** → **paired** bootstrap over per-query outcomes, since every
  config is evaluated on the same queries (unpaired intervals overstate significance).

**nDCG is deliberately omitted** (spec §I.9): coverage-based gold is non-additive — two
chunks useless alone can be sufficient together — which per-item graded relevance cannot
model. Recall and completion-rank MRR suffice; nDCG would require explicit per-passage graded
judgments the query set does not supply.

Bootstraps are **seeded** (default 0) so every reported interval is reproducible.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from retrieval_lab.models import QueryResult

DEFAULT_MIN_SAMPLE = 20
"""Below this many labeled queries, aggregate verdicts are suppressed (spec §I.11).

A gate against declaring a winner on a handful of queries; configurable per call. Hand-
labeled sets are small, so this is a real and common condition, not an edge case.
"""

DEFAULT_Z = 1.96  # ~95% normal quantile, for Wilson intervals
DEFAULT_N_BOOT = 2000
DEFAULT_ALPHA = 0.05


# --------------------------------------------------------------------------------------
# Interval primitives
# --------------------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = DEFAULT_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (spec §I.9)."""
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values`` (seeded, reproducible)."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


@dataclass
class Comparison:
    """Paired comparison of a metric between two configs on the same queries."""

    mean_diff: float                 # metric(a) - metric(b), per-query paired
    ci: tuple[float, float]
    significant: bool                # True iff the CI excludes 0
    n: int


def paired_bootstrap_diff(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
) -> Comparison:
    """Paired bootstrap on per-query differences (spec §I.9).

    ``values_a[i]`` and ``values_b[i]`` must be the same query under two configs. Resampling
    *queries* (not the two arms independently) respects the pairing, so the interval doesn't
    overstate significance.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs equal-length, aligned per-query values")
    n = a.size
    if n == 0:
        return Comparison(0.0, (0.0, 0.0), False, 0)
    diff = a - b
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return Comparison(float(diff.mean()), (lo, hi), not (lo <= 0.0 <= hi), n)


# --------------------------------------------------------------------------------------
# Per-config aggregation
# --------------------------------------------------------------------------------------


def reciprocal_rank(result: QueryResult) -> float:
    """``1 / gold_completion_rank`` (0 if the config never satisfied gold)."""
    r = result.gold_completion_rank
    return 1.0 / r if r else 0.0


@dataclass
class ConfigMetrics:
    """Aggregate metrics for one config over a labeled query set (spec §I.9)."""

    config_id: str
    n: int
    hits: int
    hit_rate: float                              # == Recall@k under single-alternative gold
    hit_rate_ci: tuple[float, float] | None      # Wilson; None when the verdict is suppressed
    mrr: float
    mrr_ci: tuple[float, float] | None           # bootstrap; None when suppressed
    verdict_suppressed: bool
    stage_breakdown: dict[str, int] = field(default_factory=dict)  # miss counts by DAG stage
    fragmented_queries: int = 0                  # hits that were reconstructed across chunks
    notes: list[str] = field(default_factory=list)


def aggregate_config(
    results: Sequence[QueryResult],
    config_id: str | None = None,
    min_sample: int = DEFAULT_MIN_SAMPLE,
    seed: int = 0,
) -> ConfigMetrics:
    """Aggregate per-query results for one config, with CIs and the minimum-sample gate.

    Below ``min_sample`` the point estimates are still reported (they are facts about the
    sample) but the confidence intervals are ``None`` and ``verdict_suppressed`` is set — the
    caller must not present a winner (spec §I.11).
    """
    results = [r for r in results if not r.refused]
    n = len(results)
    hits = sum(1 for r in results if r.hit)
    hit_rate = hits / n if n else 0.0
    rr = [reciprocal_rank(r) for r in results]
    mrr = float(np.mean(rr)) if rr else 0.0

    suppressed = n < min_sample
    hit_rate_ci = None if suppressed else wilson_interval(hits, n)
    mrr_ci = None if suppressed else bootstrap_mean_ci(rr, seed=seed)

    stage_breakdown = dict(
        Counter(r.stage_attribution for r in results if not r.hit and r.stage_attribution)
    )
    fragmented = sum(1 for r in results if r.hit and r.fragmented_spans)

    notes: list[str] = []
    if suppressed:
        notes.append(
            f"n={n} < min_sample={min_sample}: aggregate verdict suppressed; "
            "per-query diffs only (spec §I.11)"
        )
    cid = config_id if config_id is not None else (results[0].config_id if results else "?")
    return ConfigMetrics(
        config_id=cid,
        n=n,
        hits=hits,
        hit_rate=hit_rate,
        hit_rate_ci=hit_rate_ci,
        mrr=mrr,
        mrr_ci=mrr_ci,
        verdict_suppressed=suppressed,
        stage_breakdown=stage_breakdown,
        fragmented_queries=fragmented,
        notes=notes,
    )


def compare_configs(
    results_a: Sequence[QueryResult],
    results_b: Sequence[QueryResult],
    metric: str = "hit",
    seed: int = 0,
) -> Comparison:
    """Paired comparison of two configs on the same queries (aligned by query id).

    ``metric`` is ``"hit"`` (per-query 0/1 hit) or ``"rr"`` (reciprocal rank). Queries are
    aligned by id and any query missing from either arm is dropped (with pairing preserved).
    """
    by_id_b = {r.query_id: r for r in results_b}
    paired = [(a, by_id_b[a.query_id]) for a in results_a if a.query_id in by_id_b]
    if metric == "hit":
        va = [1.0 if a.hit else 0.0 for a, _ in paired]
        vb = [1.0 if b.hit else 0.0 for _, b in paired]
    elif metric == "rr":
        va = [reciprocal_rank(a) for a, _ in paired]
        vb = [reciprocal_rank(b) for _, b in paired]
    else:
        raise ValueError(f"unknown metric {metric!r}; use 'hit' or 'rr'")
    return paired_bootstrap_diff(va, vb, seed=seed)


# --------------------------------------------------------------------------------------
# Validity gates (fail closed, spec §I.11)
# --------------------------------------------------------------------------------------


@dataclass
class ValidityReport:
    """Cross-config validity checks that gate whether verdicts may be presented."""

    baseline_broken: bool                 # zero recall under every config
    verdicts_suppressed: bool             # any config below the minimum-sample gate
    attribution_available: bool           # False for black-box (non-decomposable) pipelines
    notes: list[str] = field(default_factory=list)


def validity_report(
    per_config: Sequence[ConfigMetrics],
    attribution_available: bool = True,
) -> ValidityReport:
    """Aggregate the fail-closed gates of spec §I.11 across a sweep's configs."""
    notes: list[str] = []
    baseline_broken = bool(per_config) and all(m.hit_rate == 0.0 for m in per_config)
    if baseline_broken:
        notes.append("baseline broken: zero recall under all configs — check gold/corpus.")

    verdicts_suppressed = any(m.verdict_suppressed for m in per_config)
    if verdicts_suppressed:
        notes.append("at least one config below the minimum-sample gate — no aggregate winner.")

    if not attribution_available:
        notes.append("pipeline not stage-decomposable (black-box): attribution suppressed, "
                     "scores only (spec §I.11).")

    return ValidityReport(
        baseline_broken=baseline_broken,
        verdicts_suppressed=verdicts_suppressed,
        attribution_available=attribution_available,
        notes=notes,
    )
