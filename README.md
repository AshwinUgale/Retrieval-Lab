# Retrieval Lab

A corpus-specific retrieval benchmark and **deterministic, stage-level failure-attribution
engine** for RAG pipelines. On *your* corpus it answers two questions:

1. **Which configuration retrieves best**, under a fair context/cost budget?
2. **For each failing query, which stage caused the failure?** — chunking, dense/sparse
   candidate generation, fusion, reranking, the final cutoff, or budget packing.

Existing tools tell you *that* a query failed. Retrieval Lab tells you *where* it failed,
using chunking-independent source-span gold and one coverage predicate shared by the scorer
and the attribution engine, so they can never disagree.

> Design authority: `PROJECTS-TECHNICAL-SPEC.md` Part I. This README is a summary; the spec
> governs.

## Limitations (read first)

1. Results are relative to your query set; a thin or biased set yields a biased winner.
2. Recall against single-alternative gold is a **lower bound**.
3. Attribution needs a stage-decomposable pipeline; black-box retrievers get scores only.
4. Latency/cost axes are environment-specific; only quality and token-budget axes transfer.

## Quickstart

```bash
pip install -e ".[dev]"
retrieval-lab demo            # realistic keyless demo: configs diverge, writes report.html
retrieval-lab demo --dataset basic     # 4-doc sanity corpus (everything hits)
```

`demo` needs no data and no downloads. The default `realistic` dataset is a fictional
product's documentation with hard queries, so the report actually shows the differentiator —
hit@k ranging ~0.44–0.94 across configs, with failures attributed to `candidate_generation`,
`fusion`, `reranker_demotion`, and `final_cutoff`, plus fragmentation under small chunks. On
your own corpus:

```bash
retrieval-lab run --corpus docs.jsonl --queries queries.jsonl \
    --embed-models det --chunkers fixed,recursive --retrieval dense,hybrid \
    --rerank none,lexical --top-k 5 --candidate-n 50 \
    --json out.json --html report.html --fail-under 0.8    # --fail-under = CI quality gate

retrieval-lab explain  --json out.json --query-id Q17      # per-stage failure attribution
retrieval-lab pareto   --json out.json                     # quality × retrieved-tokens frontier
retrieval-lab geometry --corpus docs.jsonl                 # embedding-space risk indicators
```

Add `--measure-latency` to `run` for p50/p95 retrieval latency and index size — reported but
flagged **environment-specific** (only quality and the token budget transfer across machines).

- **`docs.jsonl`** — one `{"id", "text"}` per line.
- **`queries.jsonl`** — one query per line with **source-span gold** (character offsets +
  `quoted_text`, optionally a `source_version` hash). Offsets are verified at load and the
  tool **fails closed** on any drift.

Don't hand-compute offsets — author gold from **answer quotes** and let the tool stamp them:

```bash
# spec.jsonl:  {"id":"Q1","text":"what is CX-429?","source_id":"D1","answer":"error code CX-429"}
retrieval-lab make-gold --corpus docs.jsonl --spec spec.jsonl --out queries.jsonl
```

`answer` = one required span; `answer_all: [...]` = several required spans (conjunction);
`alternatives: [...]` = several acceptable answers (disjunction). A quote that isn't found is
rejected at authoring time.
- Exit codes for CI: `0` ok, `1` baseline-broken or `--fail-under` gate missed, `2` input /
  gold-verification error.

## Data model in one breath

Gold is defined over **source-document character spans**, never chunk indices — because
changing the chunker changes the chunks. A query is a **hit** when the union of retrieved
chunks covers a required span to ≥ 80%; the *same* predicate drives the scorer and the
attribution engine, so they can never disagree. When a query misses, attribution names the
earliest failing DAG stage: `representation`, `candidate_generation`, `fusion`,
`reranker_demotion`, `final_cutoff`, or `budget_cutoff`.

## Install & extras

Core is numpy + stdlib only — dense, BM25, hybrid, attribution, metrics, reporting, and the
CLI all run keyless. Heavy paths are opt-in extras:

| extra | adds |
|-------|------|
| `[real-embed]` | real e5 / bge embedders via sentence-transformers |
| `[rerank]` | cross-encoder reranker |
| `[ann]` | HNSW approximate dense index |
| `[dev]` | pytest + ruff |

## Test

```bash
python -m pytest        # keyless, network-free; includes the planted-failure recovery suite
ruff check src tests
```

The whole test/validation path is keyless and deterministic — a hashing-trick embedder keeps
the constructed-ground-truth recovery suite reproducible in CI. Real models and HNSW have
opt-in tests (`RLAB_REAL_EMBED=1`; install `[ann]`).

## Status

Built in phases; see `aie/roadmap/retrieval-lab/progress/STATUS.md`. Phases 0–8 complete:
foundation → dense → hybrid+attribution → rerank/budget → metrics → sweep+real models →
report/CLI → semantic/parent-child/geometry/ANN → productization.
