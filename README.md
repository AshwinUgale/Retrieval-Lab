# Retrieval Lab

Retrieval Lab benchmarks RAG retrieval configurations on your corpus and explains where failed
queries were lost.

It compares chunking, embedding, dense/BM25/hybrid retrieval, reranking, token budgets, and
exact or HNSW indexes. Each run produces a self-contained HTML report with ranked
configurations, confidence intervals, failure attribution, cost measurements, filters, and
quality-versus-context trade-offs.

## Why Retrieval Lab?

Aggregate retrieval scores answer *which configuration won*. They do not explain *why a query
missed*. Retrieval Lab records each stage of the retrieval pipeline and attributes every miss
to the earliest stage that lost the required evidence:

`representation` → `ann_index` → `candidate_generation` → `fusion` →
`reranker_demotion` → `final_cutoff` → `budget_cutoff`

Gold answers use source-document character spans rather than chunk IDs. This keeps evaluation
stable while chunking strategies change, and lets the scorer reconstruct answers covered by
multiple retrieved chunks.

## Quick start

Try the offline demo:

```bash
pip install retrieval-lab
retrieval-lab demo
```

The demo uses a deterministic test embedder and requires no model download or API key. For a
real benchmark with local embedding and reranking models:

```bash
pip install "retrieval-lab[real-embed,rerank]"

retrieval-lab run \
  --corpus docs.jsonl \
  --queries queries.jsonl \
  --embed-models e5,bge \
  --chunkers fixed:200,fixed:400,recursive:400,parentchild:800x200 \
  --retrieval dense,sparse,hybrid \
  --rerank none,ce \
  --candidate-n 15 \
  --top-k 5 \
  --measure-latency \
  --json results.json \
  --html report.html
```

Open `report.html` directly in a browser. It has no server or external frontend dependencies,
supports light and dark themes, and remains usable without network access.

## Input data

A run needs two JSONL files.

### `docs.jsonl`

One source document per line:

```json
{"id":"D1","text":"Cirrus returns error code CX-429 when an API key is invalid."}
{"id":"D2","text":"Requests are limited to 100 per minute.","meta":{"title":"Rate limits"}}
```

`id` and `text` are required. `meta` is optional.

### `queries.jsonl`

One labeled query per line:

```json
{"id":"Q1","text":"Which error indicates an invalid API key?","gold":{"alternatives":[{"required_spans":[{"source_id":"D1","start":15,"end":32,"quoted_text":"error code CX-429"}]}]}}
```

Each gold span points into the original document text. On load, Retrieval Lab verifies that
`quoted_text` exactly matches `[start:end]`; stale or incorrect offsets fail closed.

Do not calculate offsets manually. Create a small authoring file with answer quotes:

```json
{"id":"Q1","text":"Which error indicates an invalid API key?","source_id":"D1","answer":"error code CX-429"}
```

Then generate verified query gold:

```bash
retrieval-lab make-gold --corpus docs.jsonl --spec spec.jsonl --out queries.jsonl
```

Use `answer_all` when several spans are required together, or `alternatives` when any one of
several answers is acceptable.

## What gets compared

- **Chunking:** fixed-size, recursive, semantic, and parent-child
- **Embedding:** local E5 and BGE models, plus a deterministic offline test embedder
- **Retrieval:** exact dense, BM25 sparse, or hybrid retrieval with reciprocal rank fusion
- **Reranking:** none, lexical overlap, or a sentence-transformers cross-encoder
- **Cutoffs:** candidate count, final top-k, and optional context-token budget
- **Dense index:** exact search or HNSW approximate nearest-neighbor search

The HTML report ranks configurations by hit@k, shows MRR and confidence intervals, attributes
misses by pipeline stage, and reports retrieved tokens. With latency measurement enabled, it
also reports warm-query p50/p95 latency, index build time, and index size.

## Exact search and HNSW

Exact dense search is the default and is usually the right baseline for small and medium
corpora. For larger indexes, compare it with HNSW in the same sweep:

```bash
pip install "retrieval-lab[real-embed,ann]"

retrieval-lab run \
  --corpus docs.jsonl \
  --queries queries.jsonl \
  --embed-models e5 \
  --retrieval dense,hybrid \
  --dense-index exact,hnsw \
  --hnsw-m 16 \
  --hnsw-ef 50 \
  --ann-diagnostic-queries 100 \
  --measure-latency \
  --json results.json \
  --html report.html
```

For HNSW configurations, the report measures candidate recall against exact search and
attributes approximation-only misses to `ann_index`. Some Windows/Python combinations require
Microsoft C++ Build Tools to install `hnswlib`.

## Import benchmark datasets

Import SQuAD:

```bash
retrieval-lab import-squad --input dev-v2.0.json --out-dir ./squad
retrieval-lab run --corpus ./squad/docs.jsonl --queries ./squad/queries.jsonl
```

Import BEIR:

```bash
retrieval-lab import-beir \
  --corpus corpus.jsonl \
  --queries queries.jsonl \
  --qrels qrels.tsv \
  --out-dir ./beir
```

SQuAD answer offsets map directly to source-span gold. BEIR relevance labels apply to complete
passages, so its imported gold is coarser and stricter than answer-span evaluation.

## CI and automation

Use `--fail-under` as a quality gate:

```bash
retrieval-lab run \
  --corpus docs.jsonl \
  --queries queries.jsonl \
  --fail-under 0.80 \
  --json results.json
```

Exit codes:

- `0`: run completed and quality gates passed
- `1`: a baseline or `--fail-under` quality gate failed
- `2`: invalid input, configuration, or gold data

Additional commands:

```bash
retrieval-lab explain --json results.json --query-id Q17
retrieval-lab pareto --json results.json
retrieval-lab geometry --corpus docs.jsonl --embed-model e5
```

## Installation options

- `retrieval-lab`: lightweight core with NumPy and the deterministic embedder
- `retrieval-lab[real-embed]`: E5 and BGE through sentence-transformers
- `retrieval-lab[rerank]`: cross-encoder reranking
- `retrieval-lab[ann]`: HNSW approximate dense indexes
- `retrieval-lab[dev]`: pytest and Ruff for development

Python 3.10–3.12 is supported.

## Evaluation limits

- Results are only as representative as the labeled query set.
- Missing valid gold alternatives make measured recall a lower bound.
- Latency and index cost depend on the machine running the benchmark.
- Stage attribution requires a decomposable retrieval pipeline; black-box retrievers can only
  be scored at their observable output.

## Development

```bash
python -m pytest
ruff check src tests
```

The core test suite is deterministic and network-free. CI separately exercises the real HNSW
integration; real embedding-model tests are opt-in with `RLAB_REAL_EMBED=1`.

Retrieval Lab is currently beta. Public names exported from `retrieval_lab` follow semantic
versioning; internal submodules may change between minor releases.
