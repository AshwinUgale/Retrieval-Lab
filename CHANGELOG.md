# Changelog

All notable changes to Retrieval Lab are documented here. This project adheres to
[Semantic Versioning](https://semver.org) (pre-1.0: minor versions may introduce
additive features; the public API is not yet frozen).

## [Unreleased]

- _Nothing yet._

## [0.1.0]

- Initial release: corpus-specific retrieval benchmark with deterministic, stage-level
  failure attribution.
- Chunking (fixed / recursive / semantic / parent-child), embedding (deterministic test
  embedder + E5/BGE via `[real-embed]`), retrieval (dense / BM25 / hybrid + RRF), reranking
  (none / lexical / cross-encoder via `[rerank]`), token budgets, and exact / HNSW dense
  indexes (`[ann]`).
- Source-span gold with `quoted_text` verification (fails closed on stale offsets).
- Self-contained HTML report: ranked configs, hit@k / MRR with confidence intervals,
  per-stage miss attribution, retrieved-token cost, Pareto frontier, latency (optional).
- SQuAD and BEIR importers; `make-gold` authoring; `--fail-under` CI gate; keyless `demo`.
