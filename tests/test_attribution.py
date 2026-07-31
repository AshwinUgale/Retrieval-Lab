"""Phase 2 — DAG failure attribution: unit verdicts + §I.13 recovery suite (spec §I.7, §I.13).

Two layers:
- **Unit**: hand-built ``StageOutputs`` feed the attribution engine engineered intermediate
  sets, so every stage verdict and the branch diagnostics are asserted deterministically.
- **Recovery**: real end-to-end runs on planted-failure corpora, asserting the tool
  attributes each planted defect to the correct DAG stage and raises the fragmentation
  signal where planted.
"""

import pytest

from retrieval_lab.attribution import (
    STAGE_ANN_INDEX,
    STAGE_BUDGET_CUTOFF,
    STAGE_CANDIDATE_GENERATION,
    STAGE_FINAL_CUTOFF,
    STAGE_FUSION,
    STAGE_REPRESENTATION,
    STAGE_RERANKER_DEMOTION,
    StageOutputs,
    attribute,
)
from retrieval_lab.chunking import FixedSizeChunker
from retrieval_lab.corpora.constructed import build_basic_corpus
from retrieval_lab.corpora.planted import (
    build_budget_cutoff_corpus,
    build_candidate_miss_corpus,
    build_fragmented_corpus,
    build_representation_corpus,
    build_reranker_demotion_corpus,
)
from retrieval_lab.embedding import DeterministicEmbedder
from retrieval_lab.gold import EvidenceSet, GoldAnswer, GoldSpan
from retrieval_lab.models import Chunk, Config
from retrieval_lab.pipeline import RetrievalPipeline, evaluate_query
from retrieval_lab.retrieval import BM25Retriever, DenseRetriever, LexicalReranker

# --------------------------------------------------------------------------------------
# Unit: hand-built StageOutputs -> exact stage verdicts
# --------------------------------------------------------------------------------------

GOLD = GoldAnswer((EvidenceSet((GoldSpan("D", 0, 10, quoted_text="?" * 10),)),))
GOLD_CHUNK = Chunk.make("D", 0, 10, "?" * 10, chunker_spec="t")   # fully covers the span
MISS_CHUNK = Chunk.make("DX", 0, 10, "?" * 10, chunker_spec="t")  # other doc, 0 coverage

DENSE = Config(embed_model="e", chunker="c", retrieval="dense", top_k=1, candidate_n=5)
HYBRID = Config(embed_model="e", chunker="c", retrieval="hybrid", top_k=1, candidate_n=5)
RERANK = Config(embed_model="e", chunker="c", retrieval="dense", rerank="lexical",
                top_k=1, candidate_n=5)
BUDGET = Config(embed_model="e", chunker="c", retrieval="dense", top_k=2, candidate_n=5,
                budget_tokens=100)


def test_hit_attributes_to_no_stage():
    outs = StageOutputs(
        all_chunks=[GOLD_CHUNK], candidate_union=[GOLD_CHUNK],
        pre_final=[GOLD_CHUNK], final=[GOLD_CHUNK], dense_candidates=[GOLD_CHUNK],
    )
    assert attribute(outs, GOLD, DENSE).stage is None


def test_representation_failure():
    # The chunk set never contains the answer (simulated text loss).
    outs = StageOutputs(
        all_chunks=[MISS_CHUNK], candidate_union=[MISS_CHUNK],
        pre_final=[MISS_CHUNK], final=[MISS_CHUNK], dense_candidates=[MISS_CHUNK],
    )
    assert attribute(outs, GOLD, DENSE).stage == STAGE_REPRESENTATION


def test_candidate_generation_failure_with_both_branches_missing():
    outs = StageOutputs(
        all_chunks=[GOLD_CHUNK, MISS_CHUNK],   # answer exists in the corpus
        candidate_union=[MISS_CHUNK],          # but neither branch retrieved it
        pre_final=[MISS_CHUNK], final=[MISS_CHUNK],
        dense_candidates=[MISS_CHUNK], sparse_candidates=[MISS_CHUNK],
    )
    res = attribute(outs, GOLD, HYBRID)
    assert res.stage == STAGE_CANDIDATE_GENERATION
    assert res.branch_diag == {"dense": False, "sparse": False}


def test_ann_index_failure_when_exact_counterfactual_found_gold():
    config = Config(
        embed_model="e", chunker="c", retrieval="dense",
        top_k=1, candidate_n=5, dense_index="hnsw",
    )
    outs = StageOutputs(
        all_chunks=[GOLD_CHUNK, MISS_CHUNK],
        candidate_union=[MISS_CHUNK],
        exact_candidate_union=[GOLD_CHUNK],
        pre_final=[MISS_CHUNK],
        final=[MISS_CHUNK],
        dense_candidates=[MISS_CHUNK],
    )
    assert attribute(outs, GOLD, config).stage == STAGE_ANN_INDEX


def test_fusion_failure_hybrid_only():
    outs = StageOutputs(
        all_chunks=[GOLD_CHUNK, MISS_CHUNK],
        candidate_union=[GOLD_CHUNK, MISS_CHUNK],  # union has the answer
        fused=[MISS_CHUNK],                        # fused shortlist dropped it
        pre_final=[MISS_CHUNK], final=[MISS_CHUNK],
        dense_candidates=[MISS_CHUNK], sparse_candidates=[GOLD_CHUNK],
    )
    assert attribute(outs, GOLD, HYBRID).stage == STAGE_FUSION


def test_final_cutoff_failure():
    # Ranking placed the gold chunk at rank 2, but top_k=1.
    outs = StageOutputs(
        all_chunks=[GOLD_CHUNK, MISS_CHUNK],
        candidate_union=[MISS_CHUNK, GOLD_CHUNK],
        pre_final=[MISS_CHUNK, GOLD_CHUNK],
        final=[MISS_CHUNK],
        dense_candidates=[MISS_CHUNK, GOLD_CHUNK],
    )
    assert attribute(outs, GOLD, DENSE).stage == STAGE_FINAL_CUTOFF


def test_reranker_demotion_failure():
    # The reranker's full candidate_n input has the answer, but its top_k output does not.
    outs = StageOutputs(
        all_chunks=[GOLD_CHUNK, MISS_CHUNK],
        candidate_union=[GOLD_CHUNK, MISS_CHUNK],
        reranker_input=[GOLD_CHUNK, MISS_CHUNK],  # shortlist fed to the reranker has it
        reranked=[MISS_CHUNK, GOLD_CHUNK],        # reranker demoted it
        pre_final=[MISS_CHUNK, GOLD_CHUNK],
        final=[MISS_CHUNK],                       # top_k=1 -> gold pushed out
        dense_candidates=[GOLD_CHUNK, MISS_CHUNK],
    )
    assert attribute(outs, GOLD, RERANK).stage == STAGE_RERANKER_DEMOTION


def test_final_cutoff_never_returned_under_a_reranker():
    # Hardening: even with an inconsistent hand-built input (reranker_input misses), a
    # reranker config must never be attributed to final_cutoff, and a miss must never map to
    # stage None. The returned unit came through the reranker, so it's a reranker demotion.
    outs = StageOutputs(
        all_chunks=[GOLD_CHUNK, MISS_CHUNK],
        candidate_union=[GOLD_CHUNK, MISS_CHUNK],
        reranker_input=[MISS_CHUNK],   # deliberately inconsistent: input misses
        reranked=[MISS_CHUNK],
        pre_final=[MISS_CHUNK], final=[MISS_CHUNK],
        dense_candidates=[GOLD_CHUNK, MISS_CHUNK],
    )
    res = attribute(outs, GOLD, RERANK)
    assert res.stage == STAGE_RERANKER_DEMOTION
    assert res.stage is not None  # invariant: a miss always attributes to some stage


def test_budget_cutoff_failure():
    # top_k satisfies gold, but the budget-packed subset does not.
    outs = StageOutputs(
        all_chunks=[GOLD_CHUNK, MISS_CHUNK],
        candidate_union=[GOLD_CHUNK, MISS_CHUNK],
        pre_final=[MISS_CHUNK, GOLD_CHUNK],
        final=[MISS_CHUNK, GOLD_CHUNK],   # top_k has the answer
        budget_packed=[MISS_CHUNK],       # ...but packing dropped it
        dense_candidates=[GOLD_CHUNK, MISS_CHUNK],
    )
    assert attribute(outs, GOLD, BUDGET).stage == STAGE_BUDGET_CUTOFF


def test_budget_hit_when_packed_subset_satisfies():
    outs = StageOutputs(
        all_chunks=[GOLD_CHUNK],
        candidate_union=[GOLD_CHUNK],
        pre_final=[GOLD_CHUNK], final=[GOLD_CHUNK],
        budget_packed=[GOLD_CHUNK],  # packing kept the answer -> no failure
        dense_candidates=[GOLD_CHUNK],
    )
    assert attribute(outs, GOLD, BUDGET).stage is None


def test_branch_diagnostic_is_not_a_verdict_on_a_hit():
    # Dense branch missed but sparse found it; fusion recovers -> a hit, dense-miss reported.
    outs = StageOutputs(
        all_chunks=[GOLD_CHUNK, MISS_CHUNK],
        candidate_union=[MISS_CHUNK, GOLD_CHUNK],
        fused=[GOLD_CHUNK], pre_final=[GOLD_CHUNK], final=[GOLD_CHUNK],
        dense_candidates=[MISS_CHUNK], sparse_candidates=[GOLD_CHUNK],
    )
    res = attribute(outs, GOLD, HYBRID)
    assert res.stage is None                       # dense miss is NOT promoted to a failure
    assert res.branch_diag == {"dense": False, "sparse": True}


# --------------------------------------------------------------------------------------
# Recovery suite: real pipeline recovers planted failures (§I.13)
# --------------------------------------------------------------------------------------


def _pipeline(docs, chunker, config, reranker=None):
    chunks = chunker.chunk_corpus(docs.values())
    dense = None
    sparse = None
    if config.retrieval in {"dense", "hybrid"}:
        dense = DenseRetriever(DeterministicEmbedder(dim=2048)).index(chunks)
    if config.retrieval in {"sparse", "hybrid"}:
        sparse = BM25Retriever().index(chunks)
    return RetrievalPipeline(chunks, config, dense=dense, sparse=sparse, reranker=reranker)


def test_recovery_clean_hits_on_basic_corpus():
    docs, queries = build_basic_corpus()
    config = Config("det-hash", "fixed", "hybrid", top_k=3, candidate_n=10)
    # Large chunks so each short doc is one chunk -> answers are single-chunk, unfragmented.
    pipe = _pipeline(docs, FixedSizeChunker(chunk_size=400, overlap=0), config)
    for q in queries:
        res = evaluate_query(q, pipe)
        assert res.hit
        assert res.stage_attribution is None
        assert res.fragmented_spans == []


def test_recovery_representation_failure():
    docs, query, chunker = build_representation_corpus()
    config = Config("det-hash", chunker.spec, "dense", top_k=3, candidate_n=10)
    pipe = _pipeline(docs, chunker, config)
    res = evaluate_query(query, pipe)
    assert not res.hit
    assert res.stage_attribution == STAGE_REPRESENTATION


def test_recovery_final_cutoff_from_fragmentation():
    docs, query, split = build_fragmented_corpus()
    config = Config("det-hash", "fixed", "dense", top_k=1, candidate_n=10)
    pipe = _pipeline(docs, FixedSizeChunker(chunk_size=split, overlap=0), config)
    res = evaluate_query(query, pipe)
    assert not res.hit
    assert res.stage_attribution == STAGE_FINAL_CUTOFF


def test_recovery_fragmentation_signal_on_a_hit():
    docs, query, split = build_fragmented_corpus()
    config = Config("det-hash", "fixed", "dense", top_k=5, candidate_n=10)
    pipe = _pipeline(docs, FixedSizeChunker(chunk_size=split, overlap=0), config)
    res = evaluate_query(query, pipe)
    assert res.hit                       # reconstructable across chunks
    assert res.fragmented_spans          # ...but flagged as fragmented
    assert res.stage_attribution is None


def test_recovery_candidate_generation_failure():
    docs, query = build_candidate_miss_corpus()
    config = Config("det-hash", "fixed", "hybrid", top_k=1, candidate_n=1)
    pipe = _pipeline(docs, FixedSizeChunker(chunk_size=200), config)
    res = evaluate_query(query, pipe)
    assert not res.hit
    assert res.stage_attribution == STAGE_CANDIDATE_GENERATION
    assert res.branch_diag == {"dense": False, "sparse": False}


def test_recovery_reranker_demotion():
    docs, query = build_reranker_demotion_corpus()
    config = Config("det-hash", "fixed", "dense", rerank="lexical", top_k=1, candidate_n=10)
    pipe = _pipeline(docs, FixedSizeChunker(chunk_size=200), config, reranker=LexicalReranker())
    res = evaluate_query(query, pipe)
    assert not res.hit
    assert res.stage_attribution == STAGE_RERANKER_DEMOTION


def test_recovery_budget_cutoff():
    docs, query, budget = build_budget_cutoff_corpus()
    config = Config("det-hash", "fixed", "dense", top_k=3, candidate_n=10, budget_tokens=budget)
    pipe = _pipeline(docs, FixedSizeChunker(chunk_size=400), config)
    res = evaluate_query(query, pipe)
    assert not res.hit
    assert res.stage_attribution == STAGE_BUDGET_CUTOFF


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_pipeline_requires_the_right_retrievers(mode):
    config = Config("e", "c", mode)
    with pytest.raises(ValueError):
        RetrievalPipeline([], config)  # no retrievers supplied


def test_pipeline_requires_reranker_when_config_asks_for_one():
    config = Config("e", "c", "dense", rerank="lexical")
    dense = DenseRetriever(DeterministicEmbedder(dim=64)).index([])
    with pytest.raises(ValueError, match="reranker"):
        RetrievalPipeline([], config, dense=dense)  # rerank requested, none supplied
