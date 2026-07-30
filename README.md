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

Install with a real embedding model (e5/bge run locally — no API key, no data leaves your
machine) and point it at your corpus:

```bash
pip install "retrieval-lab[real-embed]"

retrieval-lab run --corpus docs.jsonl --queries queries.jsonl \
    --embed-models e5 --chunkers fixed,recursive --retrieval dense,hybrid \
    --rerank none,lexical --top-k 5 --candidate-n 50 \
    --json out.json --html report.html --fail-under 0.8    # --fail-under = CI quality gate

retrieval-lab explain  --json out.json --query-id Q17      # per-stage failure attribution
retrieval-lab pareto   --json out.json                     # quality × retrieved-tokens frontier
retrieval-lab geometry --corpus docs.jsonl --embed-model e5  # embedding-space risk indicators
```

Swap `--embed-models e5,bge` to put two real models head-to-head on *your* data. Add
`--measure-latency` for p50/p95 latency and index size — reported but flagged
**environment-specific** (only quality and the token budget transfer across machines).

### Just want to see it work? (no download)

```bash
pip install retrieval-lab
retrieval-lab demo            # realistic corpus, keyless, writes report.html — instant
```

The `demo` runs fully offline with a **keyless deterministic embedder** — a hashing-trick
stand-in that needs no model download, so the whole thing (and the test suite) is reproducible
in CI. It's great for trying the tool and it's what the tests run on, **but it is not a
semantic embedder** — its "dense" retrieval is really fuzzy lexical matching. For real
retrieval quality, use `--embed-models e5` (or `bge`) as above. The demo still shows the
differentiator: hit@k ~0.44–0.94 across configs, with failures attributed to
`candidate_generation`, `fusion`, `reranker_demotion`, and `final_cutoff`.

### Which embedder?

| name | what it is | key? |
|------|-----------|------|
| `e5`, `bge` | real local models (sentence-transformers), `[real-embed]` extra | no |
| `det` | keyless deterministic stand-in (default; CI/offline only) | no |
| OpenAI/Cohere | API models — a small adapter reading a key from the env (not built in yet) | yes |

The tool is embedder-agnostic, so its real job is to tell you *which of these actually wins on
your corpus* — often the free local model is enough.

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

### Import real benchmarks

SQuAD's `answer_start` offsets are source-span gold already, so it imports almost 1:1 (its
multiple human answers become alternatives; `is_impossible` questions are skipped):

```bash
retrieval-lab import-squad --input dev-v2.0.json --out-dir ./squad   # downloaded by you
retrieval-lab run --corpus ./squad/docs.jsonl --queries ./squad/queries.jsonl
```

BEIR (`corpus.jsonl` + `queries.jsonl` + `qrels.tsv`) also imports — relevance is
passage-level, so gold covers the whole passage (a coarser, stricter notion than BEIR's
qrels; use a large chunk size for the closest correspondence):

```bash
retrieval-lab import-beir --corpus corpus.jsonl --queries queries.jsonl --qrels qrels.tsv \
    --out-dir ./beir
```

Every imported answer is verified against the source as it's converted; anything that doesn't
match is skipped, never trusted.
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

`0.1.0` — beta. Feature-complete against the design spec: span-gold foundation, dense / BM25 /
hybrid retrieval, RRF fusion, cross-encoder reranking, the full six-stage DAG failure
attribution, Wilson/bootstrap confidence intervals with fail-closed validity gates, a config
sweep with real (e5/bge) or keyless embedders, token-budget + Pareto fair comparison,
geometry lenses, an ANN/HNSW option, JSON/HTML reporting, a gold-authoring helper, and
SQuAD/BEIR importers. The public API (the names exported from `retrieval_lab`) follows
semantic versioning; submodule internals may change between minor versions.
