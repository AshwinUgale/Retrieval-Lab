"""Reporting: JSON round-trip, ranked report, Pareto frontier, per-query explain.

Spec §I.10 (fair comparison / Pareto) and §I.12 (CLI/API).

Everything here reads a ``SweepResult`` and renders it — the machine-readable JSON (so
``explain`` / ``pareto`` can run against a prior ``run``), a ranked human report, a Pareto
frontier, and a per-query stage-attribution view.

Fair-comparison note (spec §I.10): the Pareto frontier here trades **quality (hit-rate)**
against **retrieved tokens** — the environment-independent cost axis. Latency and dollar cost
are real axes too, but they are measured under stated conditions and are *not* reproducible
across machines, so they are added only when actually measured (they are disclosed as
environment-specific, never as transferable quality).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from retrieval_lab.metrics import ConfigMetrics, ValidityReport
from retrieval_lab.models import QueryResult
from retrieval_lab.sweep import SweepResult

# --------------------------------------------------------------------------------------
# JSON round-trip
# --------------------------------------------------------------------------------------


def sweep_to_dict(sweep: SweepResult) -> dict:
    return {
        "n_docs": sweep.n_docs,
        "n_queries": sweep.n_queries,
        "results_by_config": {
            cid: [asdict(r) for r in results]
            for cid, results in sweep.results_by_config.items()
        },
        "metrics": [asdict(m) for m in sweep.metrics],
        "validity": asdict(sweep.validity),
    }


def _tuple_or_none(v) -> tuple[float, float] | None:
    return None if v is None else (float(v[0]), float(v[1]))


def sweep_from_dict(d: dict) -> SweepResult:
    results_by_config = {
        cid: [QueryResult(**r) for r in results]
        for cid, results in d["results_by_config"].items()
    }
    metrics = []
    for m in d["metrics"]:
        m = dict(m)
        m["hit_rate_ci"] = _tuple_or_none(m.get("hit_rate_ci"))
        m["mrr_ci"] = _tuple_or_none(m.get("mrr_ci"))
        metrics.append(ConfigMetrics(**m))
    validity = ValidityReport(**d["validity"])
    return SweepResult(
        n_docs=d["n_docs"],
        n_queries=d["n_queries"],
        results_by_config=results_by_config,
        metrics=metrics,
        validity=validity,
    )


def write_json(sweep: SweepResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(sweep_to_dict(sweep), indent=2), encoding="utf-8")


def read_json(path: str | Path) -> SweepResult:
    return sweep_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------------------
# Pareto frontier (quality × retrieved tokens)
# --------------------------------------------------------------------------------------


@dataclass
class ParetoPoint:
    config_id: str
    hit_rate: float
    mrr: float
    avg_retrieved_tokens: float


def avg_retrieved_tokens(results: Sequence[QueryResult]) -> float:
    live = [r for r in results if not r.refused]
    return sum(r.retrieved_tokens for r in live) / len(live) if live else 0.0


def _points(sweep: SweepResult) -> list[ParetoPoint]:
    by_id = sweep.metrics_by_config()
    points = []
    for cid, results in sweep.results_by_config.items():
        m = by_id.get(cid)
        points.append(
            ParetoPoint(
                config_id=cid,
                hit_rate=m.hit_rate if m else 0.0,
                mrr=m.mrr if m else 0.0,
                avg_retrieved_tokens=avg_retrieved_tokens(results),
            )
        )
    return points


def pareto_frontier(sweep: SweepResult) -> list[ParetoPoint]:
    """Non-dominated configs trading higher hit-rate against fewer retrieved tokens.

    A point is dominated when another has hit-rate ≥ and tokens ≤ it, with at least one
    strict. Returned sorted by hit-rate descending, then tokens ascending.
    """
    pts = _points(sweep)

    def dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
        no_worse = a.hit_rate >= b.hit_rate and a.avg_retrieved_tokens <= b.avg_retrieved_tokens
        strictly_better = (
            a.hit_rate > b.hit_rate or a.avg_retrieved_tokens < b.avg_retrieved_tokens
        )
        return no_worse and strictly_better

    frontier = [p for p in pts if not any(dominates(q, p) for q in pts if q is not p)]
    frontier.sort(key=lambda p: (-p.hit_rate, p.avg_retrieved_tokens))
    return frontier


# --------------------------------------------------------------------------------------
# Text rendering
# --------------------------------------------------------------------------------------


def _fmt_ci(ci: tuple[float, float] | None) -> str:
    return "  (n<gate)" if ci is None else f" [{ci[0]:.2f},{ci[1]:.2f}]"


def render_report(sweep: SweepResult, top: int | None = None, by: str = "hit_rate") -> str:
    """A ranked, human-readable report. Framed as best *on this query set*, never 'best'."""
    lines: list[str] = []
    lines.append(f"Retrieval Lab — {len(sweep.metrics)} configs on your query set "
                 f"({sweep.n_queries} queries, {sweep.n_docs} docs)")
    lines.append("Results are relative to YOUR query set — a thin/biased set biases the winner.")
    lines.append("")
    ranked = sweep.ranked(by)
    if top is not None:
        ranked = ranked[:top]
    lines.append(f"{'config':<58}{'n':>4}   {'hit@k (95% CI)':>22}{'MRR':>7}   stage breakdown")
    lines.append("-" * 110)
    for m in ranked:
        stages = ", ".join(f"{k}:{v}" for k, v in sorted(m.stage_breakdown.items())) or "-"
        frag = f"  frag:{m.fragmented_queries}" if m.fragmented_queries else ""
        hit = f"{m.hit_rate:.2f}{_fmt_ci(m.hit_rate_ci)}"
        lines.append(f"{m.config_id:<58}{m.n:>4}   {hit:>22}{m.mrr:>7.2f}   {stages}{frag}")
    lines.append("")
    if sweep.validity.notes:
        lines.append("Validity:")
        lines.extend(f"  - {n}" for n in sweep.validity.notes)
    else:
        lines.append("Validity: ok (aggregate verdicts permitted).")
    return "\n".join(lines)


def render_pareto(sweep: SweepResult) -> str:
    lines = ["Pareto frontier — quality (hit@k) vs retrieved tokens "
             "(the environment-independent cost axis; latency/cost added only when measured):",
             ""]
    lines.append(f"{'config':<58}{'hit@k':>7}{'MRR':>7}{'avg tokens':>12}")
    lines.append("-" * 84)
    for p in pareto_frontier(sweep):
        lines.append(f"{p.config_id:<58}{p.hit_rate:>7.2f}{p.mrr:>7.2f}"
                     f"{p.avg_retrieved_tokens:>12.1f}")
    return "\n".join(lines)


def render_explain(sweep: SweepResult, query_id: str) -> str:
    """Per-config outcome for one query: hit, the failing stage, branch diagnostics."""
    lines = [f"explain — query {query_id!r} across {len(sweep.results_by_config)} configs", ""]
    lines.append(f"{'config':<58}{'hit':>5}{'stage':>20}{'rank':>6}  diag")
    lines.append("-" * 100)
    found = False
    for cid, results in sweep.results_by_config.items():
        for r in results:
            if r.query_id != query_id:
                continue
            found = True
            stage = r.stage_attribution or ("-" if r.hit else "?")
            rank = "-" if r.gold_completion_rank is None else str(r.gold_completion_rank)
            diag = ""
            if r.branch_diag:
                diag = " ".join(f"{k}={'hit' if v else 'miss'}" for k, v in r.branch_diag.items())
            if r.fragmented_spans:
                diag += "  fragmented"
            lines.append(f"{cid:<58}{('yes' if r.hit else 'no'):>5}{stage:>20}{rank:>6}  {diag}")
    if not found:
        return f"explain — no query with id {query_id!r} in this run."
    return "\n".join(lines)
