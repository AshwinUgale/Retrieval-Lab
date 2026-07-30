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

from retrieval_lab.metrics import ConfigCost, ConfigMetrics, ValidityReport
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
    return SweepResult(
        n_docs=d["n_docs"],
        n_queries=d["n_queries"],
        results_by_config=results_by_config,
        metrics=metrics,
        validity=validity,
        cost=cost,
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
    if sweep.cost:
        lines.append("Measured cost (ENVIRONMENT-SPECIFIC — not transferable across machines; "
                     "only quality + token budgets transfer):")
        lines.append(f"  {'config':<56}{'p50 ms':>9}{'p95 ms':>9}{'index KB':>11}")
        lines.append("  " + "-" * 85)
        for m in ranked:
            c = sweep.cost.get(m.config_id)
            if c is None:
                continue
            lines.append(f"  {m.config_id:<56}{c.p50_ms:>9.2f}{c.p95_ms:>9.2f}"
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
# Self-contained HTML report (inline CSS + SVG, theme-aware, no scripts, no external assets)
# --------------------------------------------------------------------------------------


def _h(x: object) -> str:
    return html.escape(str(x))


# The six DAG failure stages: short label + one-line plain-English meaning. Colours are
# assigned as CSS custom properties (see _REPORT_CSS) from the validated categorical palette.
_STAGE_META: dict[str, tuple[str, str]] = {
    "representation": ("Representation", "the answer text isn't in any chunk — ingestion lost it"),
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
 --st-representation:#e34948;--st-candidate-generation:#eb6834;--st-fusion:#eda100;
 --st-reranker-demotion:#4a3aa7;--st-final-cutoff:#e87ba4;--st-budget-cutoff:#1baf7a;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;
 --page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
 --line:#2c2c2a;--baseline:#383835;--border:rgba(255,255,255,.10);--track:#2c2c2a;
 --accent:#3987e5;--accent2:#5598e7;
 --st-representation:#e66767;--st-candidate-generation:#d95926;--st-fusion:#c98500;
 --st-reranker-demotion:#9085e9;--st-final-cutoff:#d55181;--st-budget-cutoff:#199e70;}}
:root[data-theme="dark"]{color-scheme:dark;
 --page:#0d0d0d;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
 --line:#2c2c2a;--baseline:#383835;--border:rgba(255,255,255,.10);--track:#2c2c2a;
 --accent:#3987e5;--accent2:#5598e7;
 --st-representation:#e66767;--st-candidate-generation:#d95926;--st-fusion:#c98500;
 --st-reranker-demotion:#9085e9;--st-final-cutoff:#d55181;--st-budget-cutoff:#199e70;}
*{box-sizing:border-box}
body{font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--page);
 color:var(--ink);margin:0 auto;max-width:1060px;padding:2rem 2.2rem 3rem}
h1{font-size:1.5rem;margin:0 0 .2rem;letter-spacing:-.01em}
h2{font-size:1.05rem;margin:2.2rem 0 .6rem;font-weight:650}
.sub{color:var(--ink2);margin:.1rem 0 0;max-width:74ch}
.note,.warn{color:var(--muted);font-weight:400;font-size:.8rem}
em{font-style:italic}
.hero{display:flex;align-items:center;gap:1.4rem;margin:1.4rem 0 .2rem;padding:1.1rem 1.3rem;
 background:var(--surface);border:1px solid var(--border);border-radius:14px}
.herolabel{color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.08em}
.herofig{font-size:2.9rem;font-weight:680;line-height:1;color:var(--accent)}
.herometa{color:var(--ink2);font-size:.9rem}.herometa b{color:var(--ink)}
.guide{margin:1.1rem 0 .4rem;border:1px solid var(--border);border-radius:12px;
 background:var(--surface);padding:.1rem .9rem}
.guide summary{cursor:pointer;font-weight:600;padding:.6rem .1rem}
.guidebody{padding:.1rem .1rem .8rem;color:var(--ink2);font-size:.9rem}
.guidebody b{color:var(--ink)}
.leghead{margin:.7rem 0 .5rem}
.legend{display:flex;flex-direction:column;gap:.4rem}
.stg{display:inline-flex;align-items:baseline;gap:.4rem;font-size:.85rem;color:var(--ink2)}
.stg i{width:9px;height:9px;border-radius:3px;background:var(--c);flex:none;
 transform:translateY(1px)}
.stg b{color:var(--ink);font-weight:600}
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
.chip.bud{color:var(--st-budget-cutoff)}
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
.validity{padding-left:1.1rem;color:var(--ink2)}.validity li{margin:.25rem 0}
footer{margin-top:2.4rem;color:var(--muted);font-size:.78rem;border-top:1px solid var(--line);
 padding-top:1rem}
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


def _cfg_chips(config_id: str, show_embed: bool) -> str:
    c = _parse_cfg(config_id)
    chips = [f'<b class=chunk>{_h(_short_chunker(c["chunker"]))}</b>',
             f'<span class=chip>{_h(c["retrieval"])}</span>']
    if show_embed:
        chips.insert(0, f'<span class="chip embed">{_h(c["embed"])}</span>')
    rr = _short_rerank(c.get("rerank", "none"))
    if rr != "—":
        chips.append(f'<span class="chip rr">rerank {_h(rr)}</span>')
    if c.get("budget", "none") not in ("none", ""):
        chips.append(f'<span class="chip bud">≤{_h(c["budget"])} tok</span>')
    return " ".join(chips)


def _bar(rate: float, cls: str = "hit") -> str:
    pct = max(0.0, min(1.0, rate)) * 100
    return f'<div class="track {cls}"><i style="width:{pct:.1f}%"></i></div>'


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
    return '<span class=gate>n&lt;min</span>' if c is None else f"[{c[0]:.2f}–{c[1]:.2f}]"


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
        el.append(f'<circle cx="{sx(p.avg_retrieved_tokens):.1f}" '
                  f'cy="{sy(p.hit_rate):.1f}" r="4.5" class="dom" />')

    # Frontier dots, plus labels stacked in a right-hand gutter with leader lines and a
    # minimum vertical gap, so labels never overlap each other or the points.
    front = sorted((p for p in pts if p.config_id in frontier), key=lambda p: -p.hit_rate)
    label_x, gap, prev = w - right + 16, 18.0, top - 18.0
    for p in front:
        cx, cy = sx(p.avg_retrieved_tokens), sy(p.hit_rate)
        ly = min(max(cy, prev + gap), h - bot - 2)
        prev = ly
        lbl = _short_chunker(_parse_cfg(p.config_id)["chunker"])
        el.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" class="front" />')
        el.append(f'<line x1="{cx + 7:.1f}" y1="{cy:.1f}" x2="{label_x - 5:.1f}" '
                  f'y2="{ly:.1f}" class="lead" />')
        el.append(f'<text x="{label_x:.1f}" y="{ly + 4:.1f}" class="plabel">'
                  f'{_h(lbl)} &#183; {p.hit_rate:.2f}</text>')
    # No width/height attributes — the viewBox provides the aspect ratio and CSS sizes it.
    return (f'<svg viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="Pareto frontier of hit@k versus retrieved tokens">'
            + "".join(el) + "</svg>")


def render_html(sweep: SweepResult, by: str = "hit_rate") -> str:
    """A single self-contained HTML page (inline CSS + SVG, theme-aware) for a ``SweepResult``.

    Redesigned for a reader who didn't build the tool: readable config chips (not pipe-ids),
    hit@k bars, colour-coded + legended failure stages, a Pareto scatter, and a glossary.
    """
    ranked = sweep.ranked(by)
    embeds = {_parse_cfg(m.config_id)["embed"] for m in ranked}
    show_embed = len(embeds) > 1  # only show the embedder column when it varies
    winner = ranked[0] if ranked else None

    rows = []
    for i, m in enumerate(ranked):
        cls = " class=win" if i == 0 else ""
        badge = '<span class=best>BEST</span>' if i == 0 else f"{i + 1}"
        frag = f'<span class=frag title="hits reconstructed across chunks — fragile">' \
               f'{m.fragmented_queries} fragile</span>' if m.fragmented_queries else ''
        rows.append(
            f"<tr{cls}><td class=rank>{badge}</td>"
            f"<td class=cfg>{_cfg_chips(m.config_id, show_embed)}</td>"
            f'<td class=num>{_bar(m.hit_rate)}<div class=lbl><b>{m.hit_rate:.2f}</b> '
            f'<span class=ci>{_ci(m.hit_rate_ci)}</span></div></td>'
            f'<td class=num>{_bar(m.mrr, "mrr")}<div class=lbl>{m.mrr:.2f}</div></td>'
            f"<td class=stages>{_stage_chips(m.stage_breakdown)} {frag}</td></tr>"
        )

    legend = " ".join(
        f'<span class=stg style="--c:var(--st-{s.replace("_","-")})"><i></i>'
        f'<b>{_h(_STAGE_META[s][0])}</b> — {_h(_STAGE_META[s][1])}</span>'
        for s in _STAGE_ORDER
    )

    cost_section = ""
    if sweep.cost:
        crows = []
        for m in ranked:
            c = sweep.cost.get(m.config_id)
            if not c:
                continue
            kb = f"{c.index_bytes / 1024:.0f} KB" if c.index_bytes else "—"
            crows.append(f"<tr><td class=cfg>{_cfg_chips(m.config_id, show_embed)}</td>"
                         f"<td class=num2>{c.p50_ms:.1f}</td><td class=num2>{c.p95_ms:.1f}</td>"
                         f"<td class=num2>{kb}</td></tr>")
        cost_section = (
            '<h2>Cost <span class=warn>environment-specific — measured on this machine; '
            'only quality &amp; token budgets transfer across machines</span></h2>'
            '<div class=wrap><table class=cost><tr><th>config</th><th>p50 ms</th>'
            f'<th>p95 ms</th><th>index</th></tr>{"".join(crows)}</table></div>'
        )

    validity = (
        "".join(f"<li>{_h(n)}</li>" for n in sweep.validity.notes)
        if sweep.validity.notes else "<li>OK — the sample is large enough for aggregate "
        "verdicts.</li>"
    )

    hero = ""
    if winner is not None:
        wc = _parse_cfg(winner.config_id)
        recipe = " · ".join(filter(None, [
            wc["embed"], _short_chunker(wc["chunker"]), wc["retrieval"],
            None if _short_rerank(wc.get("rerank", "none")) == "—"
            else f"rerank {_short_rerank(wc['rerank'])}",
        ]))
        hero = (f'<div class=hero><div class=herolabel>Best on your query set</div>'
                f'<div class=herofig>{winner.hit_rate:.2f}</div>'
                f'<div class=herometa><b>hit@k</b> {_ci(winner.hit_rate_ci)} · '
                f'MRR {winner.mrr:.2f}<br><span class=recipe>{_h(recipe)}</span></div></div>')

    return (
        "<!doctype html>\n<html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width, initial-scale=1">'
        "<title>Retrieval Lab report</title><style>" + _REPORT_CSS + "</style></head><body>"
        "<header><h1>Retrieval&nbsp;Lab</h1>"
        f"<p class=sub>{len(sweep.metrics)} configurations · {sweep.n_queries} queries · "
        f"{sweep.n_docs} documents. Ranked <em>on your query set</em> — never “best” in the "
        "abstract; a thin or biased query set biases the winner.</p></header>"
        + hero +
        "<details class=guide open><summary>How to read this</summary><div class=guidebody>"
        "<p><b>hit@k</b> — the fraction of queries whose correct answer appears in the top-k "
        "results (with a 95% confidence interval). <b>MRR</b> — how <em>high</em> the answer "
        "ranks (1.00 = always rank 1). <b>fragile</b> — a hit that was reconstructed across "
        "several chunks, so it may break under a small change.</p>"
        "<p class=leghead>When a query <b>misses</b>, it is attributed to the earliest stage "
        "that lost the answer:</p>"
        f"<div class=legend>{legend}</div></div></details>"
        "<h2>Ranked configurations</h2><div class=wrap><table class=main>"
        "<tr><th></th><th>configuration</th><th>hit@k</th><th>MRR</th>"
        "<th>where the misses were lost</th></tr>"
        + "".join(rows) + "</table></div>"
        "<h2>Pareto frontier <span class=note>quality vs. context size — the non-dominated "
        "configs; pick your point on the trade</span></h2>"
        f"<figure class=pareto>{_pareto_svg(sweep)}"
        "<figcaption>Filled points are on the frontier (no config beats them on both quality "
        "and tokens); hollow points are dominated.</figcaption></figure>"
        + cost_section +
        f"<h2>Validity</h2><ul class=validity>{validity}</ul>"
        "<footer>Generated by Retrieval&nbsp;Lab.</footer>"
        "</body></html>"
    )


def write_html(sweep: SweepResult, path: str | Path, by: str = "hit_rate") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(sweep, by), encoding="utf-8")
