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

import html
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from retrieval_lab.metrics import ANNDiagnostic, ConfigCost, ConfigMetrics, ValidityReport
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
        "cost": None if sweep.cost is None else {k: asdict(v) for k, v in sweep.cost.items()},
        "ann_diagnostics": {
            k: asdict(v) for k, v in sweep.ann_diagnostics.items()
        },
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
    cost_d = d.get("cost")
    cost = None if cost_d is None else {k: ConfigCost(**v) for k, v in cost_d.items()}
    ann_diagnostics = {
        k: ANNDiagnostic(**v) for k, v in d.get("ann_diagnostics", {}).items()
    }
    return SweepResult(
        n_docs=d["n_docs"],
        n_queries=d["n_queries"],
        results_by_config=results_by_config,
        metrics=metrics,
        validity=validity,
        cost=cost,
        ann_diagnostics=ann_diagnostics,
    )


def write_json(sweep: SweepResult, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sweep_to_dict(sweep), indent=2), encoding="utf-8")


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
    return "  (sample too small for CI)" if ci is None else f" [{ci[0]:.2f},{ci[1]:.2f}]"


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
    if sweep.ann_diagnostics:
        lines.append("HNSW approximation check (candidate recall versus exact dense search):")
        seen: set[tuple[str, str, int]] = set()
        for m in ranked:
            diag = sweep.ann_diagnostics.get(m.config_id)
            if diag is None:
                continue
            cfg = _parse_cfg(m.config_id)
            key = (cfg["embed"], cfg["chunker"], diag.k)
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"  {cfg['embed']} | {cfg['chunker']}: recall@{diag.k}="
                f"{diag.mean_recall:.3f}, changed={diag.queries_below_full_recall}/{diag.n}"
            )
        lines.append("")
    if sweep.cost:
        lines.append("Measured cost (ENVIRONMENT-SPECIFIC — not transferable across machines; "
                     "only quality + token budgets transfer):")
        lines.append(
            f"  {'config':<56}{'p50 ms':>9}{'p95 ms':>9}{'build ms':>10}{'index KB':>11}"
        )
        lines.append("  " + "-" * 95)
        for m in ranked:
            c = sweep.cost.get(m.config_id)
            if c is None:
                continue
            build = "—" if c.build_ms is None else f"{c.build_ms:.1f}"
            lines.append(f"  {m.config_id:<56}{c.p50_ms:>9.2f}{c.p95_ms:>9.2f}"
                         f"{build:>10}"
                         f"{c.index_bytes / 1024:>11.1f}")
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


# --------------------------------------------------------------------------------------
# Self-contained HTML report (inline CSS + SVG + tiny JS; open in any browser, no CDN)
# --------------------------------------------------------------------------------------

# Default table views for large sweeps: top-N first, then filtered + paginated "all".
_HTML_TOP_N = 15
_HTML_PAGE_SIZE = 12


def _h(x: object) -> str:
    return html.escape(str(x))


# The six DAG failure stages: short label + one-line plain-English meaning. Colours are
# assigned as CSS custom properties (see _REPORT_CSS) from the validated categorical palette.
_STAGE_META: dict[str, tuple[str, str]] = {
    "representation": ("Representation", "the answer text isn't in any chunk — ingestion lost it"),
    "ann_index": ("ANN approximation",
                  "exact dense search found it, but the HNSW approximation did not"),
    "candidate_generation": ("Not retrieved",
                             "neither dense nor sparse fetched it into the candidates"),
    "fusion": ("Fusion drop", "candidates had it, but combining dense + sparse lost it"),
    "reranker_demotion": ("Reranker demotion", "the reranker reordered it out of the top-k"),
    "final_cutoff": ("Final cutoff", "retrieved and ranked, but it landed just past the top-k"),
    "budget_cutoff": ("Budget cutoff", "it was in the top-k, but the token budget dropped it"),
}
_STAGE_ORDER = list(_STAGE_META)

# Theme-aware styling. Colours come from the validated data-viz categorical palette; the dark
# column is the same hues re-stepped for the dark surface. Single accent (blue) for the hit@k
# magnitude bars; the six categorical hues identify the failure stages (always with a label,
# so identity is never colour-alone).
_REPORT_CSS = """
:root{color-scheme:light dark;
 --page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
 --line:#e1e0d9;--baseline:#c3c2b7;--border:rgba(11,11,11,.10);--track:#eceae4;
 --accent:#2a78d6;--accent2:#256abf;
 --st-representation:#e34948;--st-ann-index:#bb4f9a;
 --st-candidate-generation:#eb6834;--st-fusion:#eda100;
 --st-reranker-demotion:#4a3aa7;--st-final-cutoff:#e87ba4;--st-budget-cutoff:#1baf7a;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;
 --page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
 --line:#2c2c2a;--baseline:#383835;--border:rgba(255,255,255,.10);--track:#2c2c2a;
 --accent:#3987e5;--accent2:#5598e7;
 --st-representation:#e66767;--st-ann-index:#d377c1;
 --st-candidate-generation:#d95926;--st-fusion:#c98500;
 --st-reranker-demotion:#9085e9;--st-final-cutoff:#d55181;--st-budget-cutoff:#199e70;}}
:root[data-theme="dark"]{color-scheme:dark;
 --page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
 --line:#2c2c2a;--baseline:#383835;--border:rgba(255,255,255,.10);--track:#2c2c2a;
 --accent:#3987e5;--accent2:#5598e7;
 --st-representation:#e66767;--st-ann-index:#d377c1;
 --st-candidate-generation:#d95926;--st-fusion:#c98500;
 --st-reranker-demotion:#9085e9;--st-final-cutoff:#d55181;--st-budget-cutoff:#199e70;}
*{box-sizing:border-box}
body{font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--page);
 color:var(--ink);margin:0 auto;max-width:1060px;padding:2rem 2.2rem 3rem}
header{position:relative;padding-right:6rem}
h1{font-size:1.5rem;margin:0 0 .2rem;letter-spacing:-.01em}
h2{font-size:1.05rem;margin:2.2rem 0 .6rem;font-weight:650}
.sub{color:var(--ink2);margin:.1rem 0 0;max-width:74ch}
.themebtn{position:absolute;right:0;top:0;font:inherit;font-size:.78rem;font-weight:600;
 color:var(--ink);background:var(--surface);border:1px solid var(--border);border-radius:8px;
 padding:.32rem .62rem;cursor:pointer}
.note,.warn{color:var(--muted);font-weight:400;font-size:.8rem}
em{font-style:italic}
.hero{display:flex;align-items:center;gap:1.4rem;margin:1.4rem 0 .2rem;padding:1.1rem 1.3rem;
 background:var(--surface);border:1px solid var(--border);border-radius:14px}
.herolabel{color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.08em}
.herofig{font-size:2.9rem;font-weight:680;line-height:1;color:var(--accent)}
.herometa{color:var(--ink2);font-size:.9rem}.herometa b{color:var(--ink)}
.guide{margin:1.1rem 0 .4rem;border:1px solid var(--border);border-radius:12px;
 background:var(--surface);padding:.1rem .9rem}
.guide summary{cursor:pointer;font-weight:650;padding:.7rem .1rem}
.guidebody{padding:.1rem .1rem .8rem;color:var(--ink2);font-size:.9rem}
.guidebody b{color:var(--ink)}
.guidegrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:.65rem}
.guidecard{padding:.75rem .8rem;border:1px solid var(--border);border-radius:9px;
 background:var(--page)}
.guidecard b{display:block;margin-bottom:.18rem}.guidecard p{margin:0}
.guidesub{margin:1rem 0 .5rem;color:var(--ink);font-weight:650}
.legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(285px,1fr));gap:.55rem}
.stg{display:inline-flex;align-items:baseline;gap:.4rem;font-size:.85rem;color:var(--ink2)}
.stg i{width:9px;height:9px;border-radius:3px;background:var(--c);flex:none;
 transform:translateY(1px)}
.stg b{color:var(--ink);font-weight:600}
.legend .stg{align-items:flex-start;padding:.6rem .7rem;border:1px solid var(--border);
 border-radius:9px;background:var(--page);line-height:1.4}
.legend .stg i{margin-top:.25rem}.legend .stg b{display:inline}
.trust{margin:1rem 0 .35rem;padding:.85rem 1rem;border:1px solid var(--border);
 border-left:4px solid var(--accent);border-radius:10px;background:var(--surface)}
.trust.warnbox{border-left-color:var(--st-candidate-generation)}
.trust h2{margin:0 0 .2rem;font-size:.95rem}.trust p{margin:.15rem 0;color:var(--ink2)}
.trust ul{margin:.35rem 0 0;padding-left:1.2rem;color:var(--ink2)}
.sectionintro{color:var(--ink2);max-width:80ch;margin:.15rem 0 .75rem}
.wrap{overflow-x:auto;border:1px solid var(--border);border-radius:12px}
table{border-collapse:collapse;width:100%;font-size:.87rem;background:var(--surface)}
th,td{text-align:left;padding:.6rem .8rem;border-bottom:1px solid var(--line);vertical-align:middle}
th{color:var(--muted);font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;
 white-space:nowrap}
tr:last-child td{border-bottom:none}
.num,.num2{font-variant-numeric:tabular-nums}
td.rank{color:var(--muted);font-weight:600;width:3rem;text-align:center;white-space:nowrap}
.best{display:inline-block;background:var(--accent);color:#fff;font-size:.6rem;font-weight:700;
 letter-spacing:.05em;padding:.16rem .38rem;border-radius:5px}
tr.win td{background:color-mix(in srgb,var(--accent) 8%,transparent)}
td.cfg{white-space:normal;line-height:1.95}
.chunk{font-weight:650;color:var(--ink)}
.chip{display:inline-block;font-size:.75rem;color:var(--ink2);
 background:color-mix(in srgb,var(--ink) 7%,transparent);padding:.08rem .42rem;border-radius:6px;
 white-space:nowrap}
.chip.embed{color:var(--accent2)}.chip.rr{color:var(--st-reranker-demotion)}
.chip.bud{color:var(--st-budget-cutoff)}.chip.idx{color:var(--st-fusion)}
.track{height:8px;border-radius:4px;background:var(--track);overflow:hidden;min-width:110px;max-width:170px}
.track i{display:block;height:100%;background:var(--accent);border-radius:4px}
.track.mrr i{background:var(--baseline)}
.lbl{font-size:.82rem;margin-top:.22rem}.lbl b{font-weight:660}
.ci{color:var(--muted);font-size:.72rem}.gate{color:var(--muted);font-style:italic}
td.stages{white-space:normal;line-height:2}td.stages .stg{margin-right:.75rem}
.none{color:var(--muted)}
.frag{color:var(--muted);font-size:.77rem;border:1px solid var(--border);border-radius:6px;
 padding:.05rem .35rem}
.pareto{margin:.4rem 0 0;padding:1rem 1.1rem;background:var(--surface);
 border:1px solid var(--border);border-radius:12px}
.pareto figcaption{color:var(--muted);font-size:.78rem;margin-top:.5rem}
.pareto svg{display:block;width:100%;height:auto;aspect-ratio:760/440}
svg .grid{stroke:var(--line);stroke-width:1}svg .tick{fill:var(--muted);font-size:11px}
svg .yr{text-anchor:end}svg .xr{text-anchor:middle}svg .axttl{fill:var(--muted);font-size:11px}
svg .dom{fill:var(--surface);stroke:var(--muted);stroke-width:1.5}
svg .front{fill:var(--accent);stroke:var(--surface);stroke-width:2}
svg .lead{stroke:var(--baseline);stroke-width:1}
svg .plabel{fill:var(--ink2);font-size:12px;font-variant-numeric:tabular-nums}
footer{margin-top:2.4rem;color:var(--muted);font-size:.78rem;border-top:1px solid var(--line);
 padding-top:1rem}
.toolbar{display:flex;flex-wrap:wrap;gap:.55rem .75rem;align-items:end;margin:.2rem 0 .75rem;
 padding:.85rem 1rem;background:var(--surface);border:1px solid var(--border);border-radius:12px}
.field{display:flex;flex-direction:column;gap:.22rem;min-width:7.5rem}
.field label{font-size:.68rem;font-weight:650;color:var(--muted);text-transform:uppercase;
 letter-spacing:.05em}
.field select{appearance:auto;font:inherit;font-size:.84rem;color:var(--ink);
 background:var(--page);border:1px solid var(--border);border-radius:8px;padding:.35rem .5rem}
.statusbar{display:flex;flex-wrap:wrap;gap:.6rem 1rem;align-items:center;
 justify-content:space-between;margin:0 0 .55rem;color:var(--ink2);font-size:.84rem}
.pager{display:flex;gap:.4rem;align-items:center}
.pager button{font:inherit;font-size:.8rem;font-weight:600;color:var(--ink);
 background:var(--surface);border:1px solid var(--border);border-radius:8px;
 padding:.3rem .65rem;cursor:pointer}
.pager button:disabled{opacity:.4;cursor:default}
.pager .pg{color:var(--muted);font-variant-numeric:tabular-nums;min-width:4.5rem;text-align:center}
tr.cfg-row[hidden],tr.cost-row[hidden]{display:none}
.noscript{padding:.65rem .8rem;border:1px solid var(--border);border-radius:9px;
 color:var(--ink2);background:var(--surface)}
"""


def _parse_cfg(config_id: str) -> dict[str, str]:
    parts = config_id.split("|")
    d = {
        "embed": parts[0] if parts else "",
        "chunker": parts[1] if len(parts) > 1 else "",
        "retrieval": parts[2] if len(parts) > 2 else "",
    }
    for p in parts[3:]:
        if "=" in p:
            k, v = p.split("=", 1)
            d[k] = v
    if "index" not in d:
        d["index"] = "none" if d["retrieval"] == "sparse" else "exact"
    return d


def _short_chunker(ch: str) -> str:
    kind, _, rest = ch.partition(":")
    label = {"fixed": "fixed", "recursive": "recursive",
             "parentchild": "parent-child", "semantic": "semantic"}.get(kind, kind)
    if kind in ("fixed", "recursive") and rest:
        label += f" {rest}"
    elif kind == "parentchild" and rest:
        label += f" {rest}"  # e.g. 800x200
    return label


def _short_rerank(rr: str) -> str:
    if rr in ("none", "", "None"):
        return "—"
    if rr == "lexical":
        return "lexical"
    if rr.startswith("ce"):
        model = rr.split(":", 1)[1] if ":" in rr else ""
        tail = model.split("/")[-1].replace("ms-marco-", "") if model else ""
        return f"ce·{tail}" if tail else "ce"
    return rr


def _cfg_chips(config_id: str, show_embed: bool, show_index: bool = False) -> str:
    c = _parse_cfg(config_id)
    chips = [f'<b class=chunk>{_h(_short_chunker(c["chunker"]))}</b>',
             f'<span class=chip>{_h(c["retrieval"])}</span>']
    if show_embed and c["embed"] != "none":
        chips.insert(0, f'<span class="chip embed">{_h(c["embed"])}</span>')
    index_name = c.get("index", "exact")
    if index_name == "hnsw" or (show_index and index_name != "none"):
        index_label = index_name.upper()
        if index_name == "hnsw":
            index_label += (
                f" M{c.get('hnsw_m', '?')} ef{c.get('hnsw_ef', '?')}"
            )
        chips.append(f'<span class="chip idx">{_h(index_label)} index</span>')
    rr = _short_rerank(c.get("rerank", "none"))
    if rr != "—":
        chips.append(f'<span class="chip rr">rerank {_h(rr)}</span>')
    if c.get("budget", "none") not in ("none", ""):
        chips.append(f'<span class="chip bud">≤{_h(c["budget"])} tok</span>')
    return " ".join(chips)


def _bar(rate: float, cls: str = "hit") -> str:
    pct = max(0.0, min(1.0, rate)) * 100
    label = "hit at k" if cls == "hit" else "mean reciprocal rank"
    return (
        f'<div class="track {cls}" role="progressbar" aria-label="{label}" '
        f'aria-valuemin="0" aria-valuemax="1" aria-valuenow="{rate:.3f}">'
        f'<i aria-hidden="true" style="width:{pct:.1f}%"></i></div>'
    )


def _format_bytes(size: int) -> str:
    if size <= 0:
        return "—"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit in {"B", "KB"} else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _short_config_label(config_id: str) -> str:
    c = _parse_cfg(config_id)
    parts = [
        c["embed"] if c["embed"] != "none" else None,
        _short_chunker(c["chunker"]),
        c["retrieval"],
        "HNSW" if c.get("index") == "hnsw" else None,
    ]
    rr = _short_rerank(c.get("rerank", "none"))
    if rr != "—":
        parts.append(rr)
    return " · ".join(p for p in parts if p)


def _stage_chips(breakdown: dict[str, int]) -> str:
    if not breakdown:
        return '<span class=none>— all hits</span>'
    out = []
    for stage in _STAGE_ORDER:
        n = breakdown.get(stage)
        if not n:
            continue
        label = _STAGE_META[stage][0]
        out.append(f'<span class=stg style="--c:var(--st-{stage.replace("_","-")})">'
                   f'<i></i>{_h(label)} {n}</span>')
    return " ".join(out)


def _ci(c: tuple[float, float] | None) -> str:
    return (
        '<span class=gate>sample too small for CI</span>'
        if c is None else f"[{c[0]:.2f}–{c[1]:.2f}]"
    )


def _pareto_svg(sweep: SweepResult) -> str:
    import math

    pts = _points(sweep)
    if not pts:
        return ""
    frontier = {p.config_id for p in pareto_frontier(sweep)}
    w, h, left, right, top, bot = 760, 440, 60, 208, 24, 52
    xs = [p.avg_retrieved_tokens for p in pts]
    ys = [p.hit_rate for p in pts]
    xlo, xhi = min(xs) * 0.90, (max(xs) or 1) * 1.06
    if xhi <= xlo:
        xhi = xlo + 1
    # y-domain hugs the data (round 0.05 step just below the lowest point) so points use the
    # full height instead of floating in an empty 0.80–0.90 band.
    ylo = max(0.0, math.floor((min(ys) - 0.011) / 0.05) * 0.05)
    yhi = 1.0
    if yhi - ylo < 0.05:
        ylo = max(0.0, yhi - 0.05)

    def sx(x: float) -> float:
        return left + (x - xlo) / (xhi - xlo) * (w - left - right)

    def sy(y: float) -> float:
        return top + (1 - (y - ylo) / (yhi - ylo)) * (h - top - bot)

    el: list[str] = []
    # y gridlines + round ticks every 0.05. NB: every attribute value is quoted — an unquoted
    # `class=x/>` absorbs the slash into the value and breaks the self-close, so the class never
    # matches and marks fall back to SVG's default black fill (invisible on a dark surface).
    yv = ylo
    while yv <= yhi + 1e-9:
        yy = sy(yv)
        el.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{w - right}" y2="{yy:.1f}" '
                  f'class="grid" />')
        el.append(f'<text x="{left - 8}" y="{yy + 4:.1f}" class="tick yr">{yv:.2f}</text>')
        yv += 0.05
    for xv in (xlo, (xlo + xhi) / 2, xhi):
        el.append(f'<text x="{sx(xv):.1f}" y="{h - bot + 18}" class="tick xr">{xv:.0f}</text>')
    el.append(f'<text x="{left}" y="{h - 6}" class="axttl">avg retrieved tokens &#8594;</text>')
    el.append(f'<text x="{left - 48}" y="{top - 9}" class="axttl">hit@k &#8593;</text>')

    for p in pts:  # dominated dots first, so the frontier sits on top
        if p.config_id in frontier:
            continue
        title = _h(
            f"{_short_config_label(p.config_id)} — hit@k {p.hit_rate:.3f}, "
            f"{p.avg_retrieved_tokens:.0f} avg tokens"
        )
        el.append(
            f'<circle cx="{sx(p.avg_retrieved_tokens):.1f}" '
            f'cy="{sy(p.hit_rate):.1f}" r="4.5" class="dom"><title>{title}</title></circle>'
        )

    # Frontier dots, plus labels stacked in a right-hand gutter with leader lines and a
    # minimum vertical gap, so labels never overlap each other or the points.
    front = sorted((p for p in pts if p.config_id in frontier), key=lambda p: -p.hit_rate)
    label_x, gap, prev = w - right + 16, 18.0, top - 18.0
    for p in front:
        cx, cy = sx(p.avg_retrieved_tokens), sy(p.hit_rate)
        ly = min(max(cy, prev + gap), h - bot - 2)
        prev = ly
        lbl = _short_config_label(p.config_id)
        title = _h(
            f"{lbl} — hit@k {p.hit_rate:.3f}, {p.avg_retrieved_tokens:.0f} avg tokens"
        )
        el.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" class="front">'
            f"<title>{title}</title></circle>"
        )
        el.append(f'<line x1="{cx + 7:.1f}" y1="{cy:.1f}" x2="{label_x - 5:.1f}" '
                  f'y2="{ly:.1f}" class="lead" />')
        el.append(f'<text x="{label_x:.1f}" y="{ly + 4:.1f}" class="plabel">'
                  f'{_h(lbl)} &#183; {p.hit_rate:.2f}</text>')
    # No width/height attributes — the viewBox provides the aspect ratio and CSS sizes it.
    return (f'<svg viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="Pareto frontier of hit@k versus retrieved tokens">'
            + "".join(el) + "</svg>")


def _unique_cfg_values(ranked: Sequence[ConfigMetrics], key: str) -> list[str]:
    """Stable unique values for a config dimension (for filter dropdowns)."""
    seen: list[str] = []
    for m in ranked:
        v = _parse_cfg(m.config_id).get(key, "")
        if key == "rerank" and v in ("", "None"):
            v = "none"
        if key == "budget" and v == "":
            v = "none"
        if v not in seen:
            seen.append(v)
    return seen


def _filter_select(name: str, label: str, values: Sequence[str],
                   display: dict[str, str] | None = None) -> str:
    if len(values) <= 1:
        return ""
    opts = ['<option value="">All</option>']
    for v in values:
        shown = (display or {}).get(v, v)
        opts.append(f'<option value="{_h(v)}">{_h(shown)}</option>')
    return (
        f'<div class=field><label for="f-{name}">{_h(label)}</label>'
        f'<select id="f-{name}" data-filter="{_h(name)}">{"".join(opts)}</select></div>'
    )


def _row_data_attrs(config_id: str, rank: int) -> str:
    """Filter/sync attributes. Numeric ``data-i`` keeps pipe-string ids out of HTML."""
    c = _parse_cfg(config_id)
    rr = c.get("rerank", "none") or "none"
    if rr == "None":
        rr = "none"
    bud = c.get("budget", "none") or "none"
    return (
        f'data-i="{rank}" data-rank="{rank}" '
        f'data-embed="{_h(c.get("embed", ""))}" '
        f'data-chunker="{_h(c.get("chunker", ""))}" '
        f'data-retrieval="{_h(c.get("retrieval", ""))}" '
        f'data-rerank="{_h(rr)}" data-budget="{_h(bud)}" '
        f'data-index="{_h(c.get("index", "exact"))}"'
    )


def _report_ui_script(top_n: int = _HTML_TOP_N, page_size: int = _HTML_PAGE_SIZE) -> str:
    """Inline browser script: filter configs, top-N vs all, paginate. No external deps."""
    # Keep this free of the literal sequence </script> so it can sit in a <script> tag.
    return f"""
<script>
(function () {{
  var TOP_N = {top_n}, PAGE = {page_size}, page = 1;
  function $(sel, root) {{
    return (root || document).querySelector(sel);
  }}
  function $$(sel, root) {{
    return Array.prototype.slice.call(
      (root || document).querySelectorAll(sel)
    );
  }}
  function filters() {{
    var out = {{}};
    $$("select[data-filter]").forEach(function (el) {{
      out[el.getAttribute("data-filter")] = el.value;
    }});
    return out;
  }}
  function match(row, f) {{
    return (!f.embed || row.dataset.embed === f.embed)
      && (!f.chunker || row.dataset.chunker === f.chunker)
      && (!f.retrieval || row.dataset.retrieval === f.retrieval)
      && (!f.index || row.dataset.index === f.index)
      && (!f.rerank || row.dataset.rerank === f.rerank)
      && (!f.budget || row.dataset.budget === f.budget);
  }}
  function apply() {{
    var f = filters();
    var view = ($("#view-mode") || {{value: "top"}}).value;
    var rows = $$("tr.cfg-row");
    var matched = rows.filter(function (r) {{ return match(r, f); }});
    var nMatch = matched.length;
    if (view === "top") matched = matched.slice(0, TOP_N);
    var pages = Math.max(1, Math.ceil(matched.length / PAGE));
    if (page > pages) page = pages;
    if (view === "top") page = 1;
    var start = view === "top" ? 0 : (page - 1) * PAGE;
    var end = view === "top" ? matched.length : Math.min(start + PAGE, matched.length);
    var visible = {{}};
    matched.slice(start, end).forEach(function (r) {{
      visible[r.dataset.i] = true;
    }});
    rows.forEach(function (r) {{ r.hidden = !visible[r.dataset.i]; }});
    $$("tr.cost-row").forEach(function (r) {{
      r.hidden = !visible[r.dataset.i];
    }});
    var status = $("#table-status");
    if (status) {{
      if (!nMatch) status.textContent = "No configurations match these filters.";
      else if (view === "top")
        status.textContent = "Showing top " + matched.length + " of " + nMatch
          + " matching (ranked by hit@k).";
      else
        status.textContent = "Showing " + (start + 1) + "\\u2013" + end + " of "
          + matched.length + " matching \\u00b7 page " + page + "/" + pages + ".";
    }}
    var prev = $("#pg-prev"), next = $("#pg-next");
    var label = $("#pg-label"), pager = $("#pager");
    if (pager) pager.hidden = view === "top" || matched.length <= PAGE;
    if (prev) prev.disabled = page <= 1;
    if (next) next.disabled = page >= pages;
    if (label) label.textContent = page + " / " + pages;
  }}
  document.addEventListener("DOMContentLoaded", function () {{
    var theme = $("#theme-toggle");
    function darkNow() {{
      var explicit = document.documentElement.getAttribute("data-theme");
      if (explicit) return explicit === "dark";
      return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    }}
    function themeLabel() {{
      if (theme) theme.textContent = darkNow() ? "Light mode" : "Dark mode";
    }}
    try {{
      var saved = localStorage.getItem("retrieval-lab-theme");
      if (saved === "dark" || saved === "light")
        document.documentElement.setAttribute("data-theme", saved);
    }} catch (_err) {{}}
    themeLabel();
    if (theme) theme.addEventListener("click", function () {{
      var nextTheme = darkNow() ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", nextTheme);
      try {{
        localStorage.setItem("retrieval-lab-theme", nextTheme);
      }} catch (_err) {{}}
      themeLabel();
    }});
    $$("select[data-filter], #view-mode").forEach(function (el) {{
      el.addEventListener("change", function () {{ page = 1; apply(); }});
    }});
    var prev = $("#pg-prev"), next = $("#pg-next");
    if (prev) prev.addEventListener("click", function () {{
      page -= 1; apply();
    }});
    if (next) next.addEventListener("click", function () {{
      page += 1; apply();
    }});
    apply();
  }});
}})();
</script>
"""


def render_html(sweep: SweepResult, by: str = "hit_rate") -> str:
    """A single self-contained HTML page (inline CSS + SVG + tiny JS) for a ``SweepResult``.

    Large sweeps stay readable: filter by config dimension, default to the top-N by hit@k,
    then browse all matching configs with pagination. Opens in any browser with no CDN.
    """
    ranked = sweep.ranked(by)
    embeds = _unique_cfg_values(ranked, "embed")
    chunkers = _unique_cfg_values(ranked, "chunker")
    retrievals = _unique_cfg_values(ranked, "retrieval")
    indexes = _unique_cfg_values(ranked, "index")
    reranks = _unique_cfg_values(ranked, "rerank")
    budgets = _unique_cfg_values(ranked, "budget")
    show_embed = len(embeds) > 1
    show_index = len(indexes) > 1
    winner = sweep.best(by)
    n_configs = len(ranked)
    use_browser_ui = n_configs > _HTML_TOP_N

    rows = []
    for i, m in enumerate(ranked):
        is_winner = winner is not None and m.config_id == winner.config_id
        win_cls = " win" if is_winner else ""
        badge = '<span class=best>BEST</span>' if is_winner else f"{i + 1}"
        frag = (
            f'<span class=frag title="hits reconstructed across chunks — fragile">'
            f"{m.fragmented_queries} fragile</span>"
            if m.fragmented_queries else ""
        )
        rows.append(
            f'<tr class="cfg-row{win_cls}" {_row_data_attrs(m.config_id, i + 1)}>'
            f'<td class=rank>{badge}</td>'
            f"<td class=cfg>{_cfg_chips(m.config_id, show_embed, show_index)}</td>"
            f'<td class=num>{_bar(m.hit_rate)}<div class=lbl><b>{m.hit_rate:.2f}</b> '
            f'<span class=ci>{_ci(m.hit_rate_ci)}</span></div></td>'
            f'<td class=num>{_bar(m.mrr, "mrr")}<div class=lbl>{m.mrr:.2f} '
            f'<span class=ci>{_ci(m.mrr_ci)}</span></div></td>'
            f"<td class=stages>{_stage_chips(m.stage_breakdown)} {frag}</td></tr>"
        )

    legend = " ".join(
        f'<span class=stg style="--c:var(--st-{s.replace("_","-")})"><i></i>'
        f'<b>{_h(_STAGE_META[s][0])}</b> — {_h(_STAGE_META[s][1])}</span>'
        for s in _STAGE_ORDER
    )

    chunker_labels = {c: _short_chunker(c) for c in chunkers}
    index_labels = {"none": "Not applicable", "exact": "Exact", "hnsw": "HNSW"}
    rerank_labels = {r: ("none" if r == "none" else _short_rerank(r)) for r in reranks}
    budget_labels = {b: ("none" if b == "none" else f"≤{b} tok") for b in budgets}
    filter_fields = "".join([
        _filter_select("embed", "Embedder", embeds),
        _filter_select("chunker", "Chunker", chunkers, chunker_labels),
        _filter_select("retrieval", "Retrieval", retrievals),
        _filter_select("index", "Dense index", indexes, index_labels),
        _filter_select("rerank", "Rerank", reranks, rerank_labels),
        _filter_select("budget", "Budget", budgets, budget_labels),
    ])
    if use_browser_ui or filter_fields:
        view_field = (
            '<div class=field><label for="view-mode">View</label>'
            '<select id="view-mode">'
            f'<option value="top" selected>Top {_HTML_TOP_N}</option>'
            '<option value="all">All matching</option>'
            "</select></div>"
        )
        toolbar = (
            f'<div class=toolbar role=search aria-label="Filter configurations">'
            f"{filter_fields}{view_field}</div>"
            '<div class=statusbar>'
            f'<div id="table-status" aria-live="polite">Showing top '
            f"{min(_HTML_TOP_N, n_configs)} of "
            f"{n_configs} configurations.</div>"
            '<div class=pager id=pager hidden>'
            '<button type=button id=pg-prev aria-label="Previous page">Prev</button>'
            '<span class=pg id=pg-label>1 / 1</span>'
            '<button type=button id=pg-next aria-label="Next page">Next</button>'
            "</div></div>"
        )
    else:
        toolbar = ""

    cost_section = ""
    if sweep.cost:
        crows = []
        for i, m in enumerate(ranked):
            c = sweep.cost.get(m.config_id)
            if not c:
                continue
            index_size = _format_bytes(c.index_bytes)
            build = "—" if c.build_ms is None else f"{c.build_ms:.1f}"
            crows.append(
                f'<tr class=cost-row {_row_data_attrs(m.config_id, i + 1)}>'
                f"<td class=cfg>{_cfg_chips(m.config_id, show_embed, show_index)}</td>"
                f"<td class=num2>{c.p50_ms:.1f}</td><td class=num2>{c.p95_ms:.1f}</td>"
                f"<td class=num2>{build}</td>"
                f"<td class=num2>{index_size}</td></tr>"
            )
        cost_section = (
            '<h2>Runtime and index cost</h2>'
            '<p class=sectionintro>Environment-specific measurements from this machine. '
            "Warm p50/p95 time retrieval with query embeddings already cached, so configs "
            "are comparable; cross-encoder time is included. Build time includes passage "
            "embedding and index construction. Index is the exact vector matrix or serialized "
            "HNSW graph size—not total process memory.</p>"
            '<div class=wrap><table class=cost><thead><tr><th scope=col>config</th>'
            '<th scope=col>warm p50 ms</th><th scope=col>warm p95 ms</th>'
            '<th scope=col>build ms</th><th scope=col>index</th></tr></thead>'
            f'<tbody>{"".join(crows)}</tbody></table></div>'
        )

    ann_section = ""
    if sweep.ann_diagnostics:
        first_ann_diag = next(iter(sweep.ann_diagnostics.values()))
        seen_ann: set[tuple[str, str, int]] = set()
        ann_rows = []
        for m in ranked:
            diag = sweep.ann_diagnostics.get(m.config_id)
            if diag is None:
                continue
            cfg = _parse_cfg(m.config_id)
            key = (cfg["embed"], cfg["chunker"], diag.k)
            if key in seen_ann:
                continue
            seen_ann.add(key)
            ann_rows.append(
                "<tr>"
                f'<td class=cfg><span class="chip embed">{_h(cfg["embed"])}</span> '
                f'<b class=chunk>{_h(_short_chunker(cfg["chunker"]))}</b> '
                f'<span class="chip idx">HNSW M{_h(cfg.get("hnsw_m", "?"))} '
                f'ef{_h(cfg.get("hnsw_ef", "?"))}</span></td>'
                f'<td class=num2>{diag.mean_recall:.3f}</td>'
                f'<td class=num2>{diag.queries_below_full_recall} / {diag.n}</td>'
                f'<td class=num2>{diag.min_recall:.3f}</td></tr>'
            )
        ann_section = (
            '<h2>HNSW approximation check</h2>'
            f'<p class=sectionintro>For each query, HNSW’s top-{first_ann_diag.k} candidates '
            "are compared "
            "with exact dense search. Mean candidate recall of 1.000 means no exact neighbors "
            "were lost in the sampled queries. This diagnoses approximation quality; it is "
            "separate from answer hit@k.</p>"
            '<div class=wrap><table><thead><tr>'
            '<th scope=col>embedding + chunking</th>'
            f'<th scope=col>mean candidate recall@{first_ann_diag.k}</th>'
            '<th scope=col>queries changed</th><th scope=col>worst query</th>'
            f'</tr></thead><tbody>{"".join(ann_rows)}</tbody></table></div>'
        )

    if sweep.validity.notes:
        validity_items = "".join(f"<li>{_h(n)}</li>" for n in sweep.validity.notes)
        trust_section = (
            '<section class="trust warnbox"><h2>Can you trust this comparison? '
            "Review needed</h2>"
            "<p>These checks protect against declaring a misleading winner:</p>"
            f"<ul>{validity_items}</ul></section>"
        )
    else:
        trust_section = (
            '<section class=trust><h2>Can you trust this comparison? Checks passed</h2>'
            "<p>The labeled-query sample is large enough for an aggregate comparison, "
            "and the retrieval baseline is working. Results still apply only to this "
            "corpus and query set.</p></section>"
        )

    hero = ""
    if winner is not None:
        wc = _parse_cfg(winner.config_id)
        recipe = " · ".join(filter(None, [
            wc["embed"], _short_chunker(wc["chunker"]), wc["retrieval"],
            None if _short_rerank(wc.get("rerank", "none")) == "—"
            else f"rerank {_short_rerank(wc['rerank'])}",
        ]))
        hero = (
            f'<div class=hero><div class=herolabel>Best on your query set</div>'
            f'<div class=herofig>{winner.hit_rate:.2f}</div>'
            f'<div class=herometa><b>hit@k</b> {_ci(winner.hit_rate_ci)} · '
            f'MRR {winner.mrr:.2f}<br><span class=recipe>{_h(recipe)}</span></div></div>'
        )

    script = _report_ui_script()
    noscript_notice = (
        "<noscript><p class=noscript>Interactive filters and pagination require JavaScript. "
        "All configurations are shown below, so no results are hidden.</p></noscript>"
        if use_browser_ui or filter_fields else ""
    )
    guide_open = " open" if n_configs <= _HTML_TOP_N else ""

    return (
        "<!doctype html>\n<html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width, initial-scale=1">'
        "<title>Retrieval Lab report</title><style>" + _REPORT_CSS + "</style></head><body>"
        '<header><button type=button class=themebtn id=theme-toggle '
        'aria-label="Toggle color theme">Dark mode</button><h1>Retrieval&nbsp;Lab</h1>'
        f"<p class=sub>{n_configs} configurations · {sweep.n_queries} queries · "
        f"{sweep.n_docs} documents. Ranked <em>on your query set</em> — never “best” in the "
        "abstract; a thin or biased query set biases the winner.</p></header>"
        + hero +
        trust_section +
        f"<details class=guide{guide_open}><summary>How to read this report</summary>"
        '<div class=guidebody><div class=guidegrid>'
        '<div class=guidecard><b>Hit@k — was the evidence found?</b>'
        "<p>The share of labeled queries whose answer evidence was covered by the final "
        "top-k chunks. Higher is better.</p></div>"
        '<div class=guidecard><b>Answer rank (MRR) — how early?</b>'
        "<p>Rewards placing the complete answer near the top. 1.00 means it was always "
        "complete at rank 1.</p></div>"
        '<div class=guidecard><b>95% confidence range — how certain?</b>'
        "<p>The plausible range around hit@k from this query sample. More labeled queries "
        "usually make it narrower.</p></div>"
        '<div class=guidecard><b>Fragile hit — worked across pieces</b>'
        "<p>No single chunk covered enough evidence; several returned chunks did so "
        "together. Small chunking changes may break it.</p></div>"
        '<div class=guidecard><b>Candidates vs. top-k</b>'
        "<p>Candidates are the larger shortlist retrieved first. Top-k is the smaller final "
        "set delivered after fusion or reranking.</p></div>"
        '<div class=guidecard><b>Token budget — context limit</b>'
        "<p>An optional cap on whole retrieved chunks sent onward. It compares configs at "
        "a similar context cost.</p></div></div>"
        "<p class=guidesub>Where a missed answer was first lost</p>"
        "<p>Each miss is counted once, at the earliest pipeline stage that could no longer "
        "cover the labeled answer evidence.</p>"
        f"<div class=legend>{legend}</div>"
        "<p>Use the filters to narrow a large sweep; the default view is the top "
        f"{_HTML_TOP_N} by hit@k. Switch to <b>All matching</b> to page through the rest."
        "</p></div></details>"
        "<h2>Ranked configurations</h2>"
        + noscript_notice
        + toolbar +
        '<div class=wrap><table class=main><thead><tr>'
        '<th scope=col aria-label="rank"></th><th scope=col>configuration</th>'
        '<th scope=col>hit@k</th><th scope=col>MRR</th>'
        '<th scope=col>where the misses were lost</th></tr></thead><tbody>'
        + "".join(rows) + "</tbody></table></div>"
        "<h2>Quality vs. context trade-offs "
        '<span class=note>(the Pareto frontier)</span></h2>'
        '<p class=sectionintro>“Pareto frontier” is the standard name for the outer edge of '
        "best trade-offs: a point is on it when no other tested config is both more accurate "
        "and uses the same or fewer retrieved tokens. There may be several good choices "
        "because improving quality can require more context.</p>"
        f"<figure class=pareto>{_pareto_svg(sweep)}"
        "<figcaption>Move up for more hits; move left for less context. Filled points are "
        "best trade-offs. Hollow points are dominated: another config is at least as good "
        "on both axes and better on one.</figcaption></figure>"
        + ann_section
        + cost_section +
        "<footer>Generated by Retrieval&nbsp;Lab. Open this file in any browser — "
        "no server required.</footer>"
        + script +
        "</body></html>"
    )


def write_html(sweep: SweepResult, path: str | Path, by: str = "hit_rate") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(sweep, by), encoding="utf-8")
