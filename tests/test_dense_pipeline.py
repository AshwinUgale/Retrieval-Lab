"""Phase 1 — end-to-end dense slice: chunk → embed → dense retrieve → score (spec §I.5)."""

import pytest

from retrieval_lab.chunking import FixedSizeChunker, RecursiveChunker
from retrieval_lab.corpora.constructed import build_basic_corpus
from retrieval_lab.embedding import DeterministicEmbedder, EmbeddingCache
from retrieval_lab.models import Config
from retrieval_lab.retrieval import DenseRetriever
from retrieval_lab.scoring import score_query


@pytest.mark.parametrize(
    "chunker",
    [FixedSizeChunker(chunk_size=120, overlap=20), RecursiveChunker(chunk_size=120)],
    ids=lambda c: c.spec,
)
def test_dense_pipeline_hits_every_query(chunker):
    docs, queries = build_basic_corpus()
    chunks = chunker.chunk_corpus(docs.values())

    embedder = DeterministicEmbedder(dim=2048, cache=EmbeddingCache())
    retriever = DenseRetriever(embedder).index(chunks)
    config = Config(embed_model=embedder.name, chunker=chunker.spec, retrieval="dense",
                    top_k=3, candidate_n=10)

    for query in queries:
        ranked = retriever.retrieve(query.text, k=config.candidate_n)
        result = score_query(query, ranked, config.id, top_k=config.top_k)
        assert result.hit, f"{query.id} should hit with {chunker.spec}"
        assert result.gold_completion_rank is not None
        assert result.retrieved_tokens > 0


def test_scorer_reports_miss_and_completion_rank_beyond_top_k():
    docs, queries = build_basic_corpus()
    chunks = FixedSizeChunker(chunk_size=120, overlap=20).chunk_corpus(docs.values())
    embedder = DeterministicEmbedder(dim=2048)
    retriever = DenseRetriever(embedder).index(chunks)
    query = queries[0]

    ranked = retriever.retrieve(query.text, k=10)
    # With top_k=1 the answer may sit just below the cutoff -> miss at k=1 but a defined
    # completion rank in the deeper ranked list.
    result = score_query(query, ranked, "cfg", top_k=1)
    if not result.hit:
        assert result.gold_completion_rank is not None
        assert result.gold_completion_rank > 1


def test_retrieve_before_index_raises():
    with pytest.raises(RuntimeError):
        DenseRetriever(DeterministicEmbedder(dim=32)).retrieve("q", k=3)
