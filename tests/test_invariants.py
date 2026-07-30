"""Post-build #2 — load-bearing invariants across the whole feature matrix (spec §I.7).

The tool's central promise is that the scorer and the attribution engine share one predicate
and can *never disagree*. Concretely: a query is a hit **iff** no stage is attributed. These
tests exercise that across dense/sparse/hybrid × rerank × budget × parent-child, plus the
planted representation failure, so the guarantee is regression-locked rather than argued.
"""

from retrieval_lab.attribution import (
    STAGE_BUDGET_CUTOFF,
    STAGE_CANDIDATE_GENERATION,
    STAGE_FINAL_CUTOFF,
    STAGE_FUSION,
    STAGE_REPRESENTATION,
    STAGE_RERANKER_DEMOTION,
)
from retrieval_lab.chunking import FixedSizeChunker, ParentChildChunker
from retrieval_lab.corpora.planted import build_representation_corpus
from retrieval_lab.corpora.realistic import build_realistic_corpus
from retrieval_lab.embedding import DeterministicEmbedder, EmbeddingCache
from retrieval_lab.models import Config
from retrieval_lab.pipeline import RetrievalPipeline, evaluate_query
from retrieval_lab.retrieval import BM25Retriever, DenseRetriever, LexicalReranker
from retrieval_lab.sweep import SweepSpec, run_sweep

_KNOWN_STAGES = {
    STAGE_REPRESENTATION, STAGE_CANDIDATE_GENERATION, STAGE_FUSION,
    STAGE_RERANKER_DEMOTION, STAGE_FINAL_CUTOFF, STAGE_BUDGET_CUTOFF,
}


def _full_matrix_sweep():
    docs, queries = build_realistic_corpus()
    spec = SweepSpec(
        embedders={"det": DeterministicEmbedder(dim=2048, cache=EmbeddingCache())},
        chunkers={"large": FixedSizeChunker(chunk_size=600),
                  "small": FixedSizeChunker(chunk_size=140)},
        retrieval_modes=("dense", "sparse", "hybrid"),
        rerankers={None: None, "lexical": LexicalReranker()},
        top_k=3, candidate_n=20,
        budgets=(None, 60),  # a small budget so budget_cutoff can fire on long answers
    )
    return run_sweep(docs, queries, spec, min_sample=1)


def test_hit_iff_no_stage_attributed_across_matrix():
    sweep = _full_matrix_sweep()
    total = 0
    for results in sweep.results_by_config.values():
        for r in results:
            total += 1
            # THE invariant: hit exactly when nothing is attributed.
            assert r.hit == (r.stage_attribution is None), (r.config_id, r.query_id, r.hit,
                                                            r.stage_attribution)
            if not r.hit:
                assert r.stage_attribution in _KNOWN_STAGES
    # 1 embed × 2 chunk × 3 mode × 2 rerank × 2 budget = 24 configs × 18 queries.
    assert total == 24 * 18


def test_budget_cutoff_actually_occurs_in_the_matrix():
    sweep = _full_matrix_sweep()
    stages = {
        r.stage_attribution
        for results in sweep.results_by_config.values()
        for r in results if r.stage_attribution
    }
    assert STAGE_BUDGET_CUTOFF in stages  # the 60-token budget drops some long answers


def test_invariant_holds_under_parent_child():
    docs, queries = build_realistic_corpus()
    chunker = ParentChildChunker(parent_size=600, child_size=120)
    chunks = chunker.chunk_corpus(docs.values())
    emb = DeterministicEmbedder(dim=2048)
    dense = DenseRetriever(emb).index(chunks)
    sparse = BM25Retriever().index(chunks)
    config = Config("det", chunker.spec, "hybrid", rerank="lexical", top_k=3, candidate_n=20)
    pipe = RetrievalPipeline(chunks, config, dense=dense, sparse=sparse,
                             reranker=LexicalReranker(), return_expander=chunker.expand)
    for q in queries:
        r = evaluate_query(q, pipe)
        assert r.hit == (r.stage_attribution is None)


def test_representation_is_reachable_and_consistent():
    # Planted text loss -> representation failure, and the invariant still holds.
    docs, query, chunker = build_representation_corpus()
    chunks = chunker.chunk_corpus(docs.values())
    dense = DenseRetriever(DeterministicEmbedder(dim=2048)).index(chunks)
    config = Config("det", chunker.spec, "dense", top_k=3, candidate_n=10)
    r = evaluate_query(query, RetrievalPipeline(chunks, config, dense=dense))
    assert not r.hit and r.stage_attribution == STAGE_REPRESENTATION


def test_refused_queries_never_counted_as_hits():
    # A refused result must not read as a hit and must carry no stage verdict.
    from retrieval_lab.models import QueryResult

    r = QueryResult(query_id="q", config_id="c", hit=False, refused=True)
    assert not r.hit
    assert r.stage_attribution is None
