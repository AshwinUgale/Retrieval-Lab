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

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from retrieval_lab.chunking.base import Chunker
from retrieval_lab.embedding.base import Embedder
from retrieval_lab.gold import DEFAULT_MIN_GOLD_COVERAGE, Query
from retrieval_lab.metrics import (
    DEFAULT_MIN_SAMPLE,
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
from retrieval_lab.retrieval.bm25 import BM25Retriever
from retrieval_lab.retrieval.dense import DenseRetriever
from retrieval_lab.retrieval.rerank import Reranker

_DENSE_MODES = {"dense", "hybrid"}
_SPARSE_MODES = {"sparse", "hybrid"}


@dataclass
class SweepSpec:
    """The grid to sweep. Keys become the ids used in each ``Config`` and in the report."""

    embedders: Mapping[str, Embedder]
    chunkers: Mapping[str, Chunker]
    retrieval_modes: Sequence[str] = ("dense", "hybrid")
    rerankers: Mapping[str | None, Reranker | None] = field(default_factory=lambda: {None: None})
    top_k: int = 5
    candidate_n: int = 50
    budgets: Sequence[int | None] = (None,)

    def n_configs(self) -> int:
        return (
            len(self.embedders)
            * len(self.chunkers)
            * len(self.retrieval_modes)
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
    results_by_config: dict[str, list[QueryResult]] = {}
    metrics: list[ConfigMetrics] = []
    cost: dict[str, ConfigCost] | None = {} if measure_latency else None

    for ch_name, chunker in spec.chunkers.items():
        chunks = chunker.chunk_corpus(docs)
        sparse = BM25Retriever().index(chunks)  # once per chunker

        # Parent-child chunkers return parents for retrieved children; the pipeline applies
        # this to every stage set. Duck-typed so any chunker exposing `expand` participates.
        return_expander = getattr(chunker, "expand", None)

        for emb_name, embedder in spec.embedders.items():
            dense = DenseRetriever(embedder).index(chunks)  # once per (embedder, chunker)

            for mode in spec.retrieval_modes:
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
                        )
                        pipeline = RetrievalPipeline(
                            chunks,
                            config,
                            dense=dense if mode in _DENSE_MODES else None,
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
                                outs = pipeline.run(q.text)
                                samples.append((time.perf_counter() - t0) * 1000.0)
                                results.append(
                                    score_and_attribute(q, outs, config, min_gold_coverage)
                                )
                            index_bytes = dense.index_nbytes if mode in _DENSE_MODES else 0
                            cost[config.id] = latency_stats(samples, index_bytes)
                        results_by_config[config.id] = results
                        metrics.append(
                            aggregate_config(results, config.id, min_sample, seed)
                        )

    return SweepResult(
        n_docs=len(docs),
        n_queries=len(queries),
        results_by_config=results_by_config,
        metrics=metrics,
        validity=validity_report(metrics),
        cost=cost,
    )
