"""Post-build #1 — the realistic corpus must show real divergence (spec §I.13, user story).

Locks in the property that makes it a useful example: configs genuinely differ and queries
fail at *different* DAG stages (not everything at 1.00). If a future change makes the corpus
trivial again, this fails.
"""

from retrieval_lab.chunking import FixedSizeChunker
from retrieval_lab.corpora.realistic import build_realistic_corpus, dump_realistic_corpus_jsonl
from retrieval_lab.embedding import DeterministicEmbedder, EmbeddingCache
from retrieval_lab.gold import load_documents, load_queries, verify_query
from retrieval_lab.retrieval import LexicalReranker
from retrieval_lab.sweep import SweepSpec, run_sweep


def _sweep():
    docs, queries = build_realistic_corpus()
    cache = EmbeddingCache()
    spec = SweepSpec(
        embedders={"det": DeterministicEmbedder(dim=2048, cache=cache)},
        chunkers={"large": FixedSizeChunker(chunk_size=600),
                  "small": FixedSizeChunker(chunk_size=140)},
        retrieval_modes=("dense", "sparse", "hybrid"),
        rerankers={None: None, "lexical": LexicalReranker()},
        top_k=3, candidate_n=20,
    )
    return docs, queries, run_sweep(docs, queries, spec, min_sample=1)


def test_all_gold_verifies_against_source():
    docs, queries = build_realistic_corpus()
    assert len(docs) >= 20 and len(queries) >= 15
    for q in queries:
        verify_query(q, docs)  # exact offsets + quoted_text; raises on any drift


def test_configs_genuinely_diverge():
    _, _, sweep = _sweep()
    hit_rates = [m.hit_rate for m in sweep.metrics]
    assert max(hit_rates) - min(hit_rates) >= 0.2  # not everything at 1.00
    assert max(hit_rates) < 1.0 or min(hit_rates) < 0.9  # the corpus is non-trivial


def test_multiple_failure_stages_surface():
    _, _, sweep = _sweep()
    stages = {
        r.stage_attribution
        for results in sweep.results_by_config.values()
        for r in results
        if r.stage_attribution
    }
    assert len(stages) >= 3  # e.g. candidate_generation, final_cutoff, reranker_demotion


def test_small_chunks_fragment_some_answers():
    _, _, sweep = _sweep()
    frag = any(m.fragmented_queries > 0 for m in sweep.metrics if "|small|" in m.config_id)
    assert frag


def test_large_chunks_beat_small_chunks_on_average():
    _, _, sweep = _sweep()
    m = sweep.metrics_by_config()
    large = [v.hit_rate for k, v in m.items() if "|large|" in k]
    small = [v.hit_rate for k, v in m.items() if "|small|" in k]
    assert sum(large) / len(large) > sum(small) / len(small)


def test_rare_term_query_favors_sparse_over_dense_on_small_chunks():
    # Q01 asks about the exact code CX-429: BM25 should find it where the dense branch misses.
    _, _, sweep = _sweep()
    m = sweep.metrics_by_config()
    dense_id = next(k for k in m if k.startswith("det|small|dense|rerank=none"))
    sparse_id = next(k for k in m if k.startswith("none|small|sparse|rerank=none"))
    by = sweep.results_by_config

    def hit(cid, qid):
        return next(r.hit for r in by[cid] if r.query_id == qid)

    assert hit(sparse_id, "Q01") and not hit(dense_id, "Q01")


def test_dump_round_trips_through_loaders(tmp_path):
    docs_path, queries_path = dump_realistic_corpus_jsonl(tmp_path)
    documents = load_documents(docs_path)
    queries = load_queries(queries_path, documents, strict=True)  # fail-closed verification
    assert len(queries) == len(build_realistic_corpus()[1])
