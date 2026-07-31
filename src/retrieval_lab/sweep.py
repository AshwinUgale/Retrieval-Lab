"""Sweep engine: run a grid of configs over a corpus and aggregate the results (spec §I.4).

Given a corpus, a labeled query set, and a grid of components, this runs every
``(embed_model × chunker × retrieval mode × reranker × budget)`` config, scores and attributes
every query, and aggregates per-config metrics plus the cross-config validity gates.

The two expensive steps are amortized: each chunker chunks the corpus **once**, and each
``(embedder, chunker)`` builds its dense index **once** and reuses it across every mode /
reranker / budget config — so re-embedding across the sweep is free (spec §I.7). BM25 is
built once per chunker.
"""

from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from retrieval_lab.chunking.base import Chunker
from retrieval_lab.embedding.base import Embedder
from retrieval_lab.gold import DEFAULT_MIN_GOLD_COVERAGE, Query
from retrieval_lab.metrics import (
    DEFAULT_MIN_SAMPLE,
    ANNDiagnostic,
    Comparison,
    ConfigCost,
    ConfigMetrics,
    ValidityReport,
    aggregate_config,
    compare_configs,
    latency_stats,
    validity_report,
)
from retrieval_lab.models import Config, Document, QueryResult
from retrieval_lab.pipeline import RetrievalPipeline, evaluate_query, score_and_attribute
from retrieval_lab.retrieval.ann import ANNDenseRetriever, ann_vs_exact_recall
from retrieval_lab.retrieval.bm25 import BM25Retriever
from retrieval_lab.retrieval.dense import DenseRetriever
from retrieval_lab.retrieval.rerank import Reranker

_DENSE_MODES = {"dense", "hybrid"}
_SPARSE_MODES = {"sparse", "hybrid"}


@dataclass
class SweepSpec:
    """The grid to sweep. Keys become the ids used in each ``Config`` and in the report."""

    # A None value is permitted for sparse-only specs: it preserves the requested model
    # label in config ids without constructing an unused model.
    embedders: Mapping[str, Embedder | None]
    chunkers: Mapping[str, Chunker]
    retrieval_modes: Sequence[str] = ("dense", "hybrid")
    rerankers: Mapping[str | None, Reranker | None] = field(default_factory=lambda: {None: None})
    top_k: int = 5
    candidate_n: int = 50
    budgets: Sequence[int | None] = (None,)
    dense_indexes: Sequence[str] = ("exact",)
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef: int = 50
    ann_diagnostic_queries: int = 100

    def n_configs(self) -> int:
        # Sparse retrieval is embedder-independent and is therefore evaluated once, not once
        # per configured dense model.
        index_count = len(tuple(dict.fromkeys(self.dense_indexes)))
        embedding_variants = (
            len(self.embedders)
            * index_count
            * sum(m in _DENSE_MODES for m in self.retrieval_modes)
            + sum(m == "sparse" for m in self.retrieval_modes)
        )
        return (
            len(self.chunkers)
            * embedding_variants
            * len(self.rerankers)
            * len(self.budgets)
        )


@dataclass
class SweepResult:
    n_docs: int
    n_queries: int
    results_by_config: dict[str, list[QueryResult]]
    metrics: list[ConfigMetrics]
    validity: ValidityReport
    cost: dict[str, ConfigCost] | None = None  # populated only when measure_latency=True
    ann_diagnostics: dict[str, ANNDiagnostic] = field(default_factory=dict)

    def metrics_by_config(self) -> dict[str, ConfigMetrics]:
        return {m.config_id: m for m in self.metrics}

    def ranked(self, by: str = "hit_rate") -> list[ConfigMetrics]:
        """Configs ranked best-first by ``hit_rate`` or ``mrr`` (ties broken by the other)."""
        key = {
            "hit_rate": lambda m: (m.hit_rate, m.mrr),
            "mrr": lambda m: (m.mrr, m.hit_rate),
        }[by]
        return sorted(self.metrics, key=key, reverse=True)

    def best(self, by: str = "hit_rate") -> ConfigMetrics | None:
        """Return the winner only when aggregate verdicts are permitted."""
        if self.validity.verdicts_suppressed:
            return None
        ranked = self.ranked(by)
        return ranked[0] if ranked else None

    def compare(self, a_id: str, b_id: str, metric: str = "hit", seed: int = 0) -> Comparison:
        """Paired comparison of two swept configs (same queries)."""
        return compare_configs(
            self.results_by_config[a_id], self.results_by_config[b_id], metric=metric, seed=seed
        )


def run_sweep(
    documents: Mapping[str, Document],
    queries: Sequence[Query],
    spec: SweepSpec,
    min_sample: int = DEFAULT_MIN_SAMPLE,
    min_gold_coverage: float = DEFAULT_MIN_GOLD_COVERAGE,
    seed: int = 0,
    measure_latency: bool = False,
) -> SweepResult:
    """Execute the grid and return per-config results, metrics, and validity gates.

    With ``measure_latency=True`` each query's retrieval is wall-clock timed and per-config
    p50/p95 latency + index size are captured (spec §I.10). These are **environment-specific**
    and disclosed as such — off by default so the deterministic test path is unaffected.
    """
    docs = list(documents.values())
    index_names = tuple(dict.fromkeys(spec.dense_indexes))
    unknown_indexes = set(index_names) - {"exact", "hnsw"}
    if unknown_indexes:
        raise ValueError(
            f"unknown dense index(es) {sorted(unknown_indexes)!r}; use exact and/or hnsw"
        )
    if any(mode in _DENSE_MODES for mode in spec.retrieval_modes) and not index_names:
        raise ValueError("dense/hybrid retrieval needs at least one dense index")
    if spec.ann_diagnostic_queries < 0:
        raise ValueError("ann_diagnostic_queries must be non-negative")
    if "hnsw" in index_names and spec.hnsw_ef < spec.candidate_n:
        raise ValueError("hnsw_ef must be greater than or equal to candidate_n")

    results_by_config: dict[str, list[QueryResult]] = {}
    metrics: list[ConfigMetrics] = []
    cost: dict[str, ConfigCost] | None = {} if measure_latency else None
    ann_diagnostics: dict[str, ANNDiagnostic] = {}

    for ch_name, chunker in spec.chunkers.items():
        chunks = chunker.chunk_corpus(docs)
        sparse = None
        sparse_build_ms = 0.0
        if any(mode in _SPARSE_MODES for mode in spec.retrieval_modes):
            t0 = time.perf_counter()
            sparse = BM25Retriever().index(chunks)
            sparse_build_ms = (time.perf_counter() - t0) * 1000.0

        # Parent-child chunkers return parents for retrieved children; the pipeline applies
        # this to every stage set. Duck-typed so any chunker exposing `expand` participates.
        return_expander = getattr(chunker, "expand", None)

        dense_cache: dict[tuple[str, str], DenseRetriever | ANNDenseRetriever] = {}
        exact_cache: dict[str, DenseRetriever] = {}
        build_ms: dict[tuple[str, str], float] = {}
        ann_diag_by_embed: dict[str, ANNDiagnostic] = {}

        if any(mode in _DENSE_MODES for mode in spec.retrieval_modes):
            for emb_name, embedder in spec.embedders.items():
                if embedder is None:
                    raise ValueError(f"dense/hybrid retrieval needs embedder {emb_name!r}")

                # Time passage embedding once, then add it to each index's own construction
                # time so exact-vs-HNSW build costs are comparable.
                t0 = time.perf_counter()
                embedder.embed_passage([c.text for c in chunks])
                embedding_ms = (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                exact = DenseRetriever(embedder).index(chunks)
                exact_build_ms = embedding_ms + (time.perf_counter() - t0) * 1000.0
                exact_cache[emb_name] = exact
                if "exact" in index_names:
                    dense_cache[(emb_name, "exact")] = exact
                    build_ms[(emb_name, "exact")] = exact_build_ms

                if "hnsw" in index_names:
                    t0 = time.perf_counter()
                    ann = ANNDenseRetriever(
                        embedder,
                        m=spec.hnsw_m,
                        ef_construction=spec.hnsw_ef_construction,
                        ef=spec.hnsw_ef,
                    ).index(chunks)
                    ann_build_ms = embedding_ms + (time.perf_counter() - t0) * 1000.0
                    dense_cache[(emb_name, "hnsw")] = ann
                    build_ms[(emb_name, "hnsw")] = ann_build_ms

                    sample_n = min(spec.ann_diagnostic_queries, len(queries))
                    sample_ids = sorted(
                        random.Random(seed).sample(range(len(queries)), sample_n)
                    )
                    diag_queries = [queries[i].text for i in sample_ids]
                    raw_diag = ann_vs_exact_recall(
                        ann, exact, diag_queries, k=spec.candidate_n
                    )
                    per_query = raw_diag["per_query"]
                    ann_diag_by_embed[emb_name] = ANNDiagnostic(
                        k=raw_diag["k"],
                        n=raw_diag["n"],
                        mean_recall=raw_diag["mean_recall"],
                        min_recall=min(per_query, default=1.0),
                        queries_below_full_recall=sum(v < 1.0 for v in per_query),
                    )

                # Query timings are deliberately warm and comparable across index types.
                if measure_latency:
                    embedder.embed_query([q.text for q in queries])

        for mode in spec.retrieval_modes:
            if mode in _DENSE_MODES:
                variants = [
                    (emb_name, index_name, dense_cache[(emb_name, index_name)])
                    for emb_name in spec.embedders
                    for index_name in index_names
                ]
            else:
                variants = [("none", "none", None)]

            for emb_name, index_name, dense in variants:
                for rr_name, reranker in spec.rerankers.items():
                    for budget in spec.budgets:
                        config = Config(
                            embed_model=emb_name,
                            chunker=ch_name,
                            retrieval=mode,
                            rerank=rr_name,
                            top_k=spec.top_k,
                            candidate_n=spec.candidate_n,
                            budget_tokens=budget,
                            dense_index=index_name,
                            hnsw_m=spec.hnsw_m,
                            hnsw_ef_construction=spec.hnsw_ef_construction,
                            hnsw_ef=spec.hnsw_ef,
                        )
                        pipeline = RetrievalPipeline(
                            chunks,
                            config,
                            dense=dense if mode in _DENSE_MODES else None,
                            dense_reference=(
                                exact_cache[emb_name] if index_name == "hnsw" else None
                            ),
                            sparse=sparse if mode in _SPARSE_MODES else None,
                            reranker=reranker if rr_name else None,
                            return_expander=return_expander,
                        )
                        if cost is None:
                            results = [
                                evaluate_query(q, pipeline, min_gold_coverage) for q in queries
                            ]
                        else:
                            results, samples = [], []
                            for q in queries:
                                t0 = time.perf_counter()
                                outs = pipeline.run(q.text, include_reference=False)
                                samples.append((time.perf_counter() - t0) * 1000.0)
                                pipeline.attach_exact_reference(q.text, outs)
                                results.append(
                                    score_and_attribute(q, outs, config, min_gold_coverage)
                                )
                            index_bytes = (
                                dense.index_nbytes
                                if mode in _DENSE_MODES and dense is not None
                                else 0
                            )
                            config_build_ms = sparse_build_ms if mode in _SPARSE_MODES else 0.0
                            if mode in _DENSE_MODES:
                                config_build_ms += build_ms[(emb_name, index_name)]
                            cost[config.id] = latency_stats(
                                samples, index_bytes, config_build_ms
                            )
                        results_by_config[config.id] = results
                        metrics.append(
                            aggregate_config(results, config.id, min_sample, seed)
                        )
                        if index_name == "hnsw":
                            ann_diagnostics[config.id] = ann_diag_by_embed[emb_name]

    return SweepResult(
        n_docs=len(docs),
        n_queries=len(queries),
        results_by_config=results_by_config,
        metrics=metrics,
        validity=validity_report(metrics),
        cost=cost,
        ann_diagnostics=ann_diagnostics,
    )
