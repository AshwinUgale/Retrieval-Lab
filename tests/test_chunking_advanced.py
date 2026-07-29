"""Phase 7 — semantic + parent-child chunkers (spec §I.7)."""

import pytest

from retrieval_lab.chunking import ParentChildChunker, SemanticChunker
from retrieval_lab.corpora.constructed import build_basic_corpus
from retrieval_lab.embedding import DeterministicEmbedder
from retrieval_lab.gold import GoldSpan, coverage
from retrieval_lab.models import Config, Document
from retrieval_lab.pipeline import RetrievalPipeline, evaluate_query
from retrieval_lab.retrieval import DenseRetriever

# --------------------------------- Semantic -------------------------------------------


def _covers_whole(chunks, doc):
    whole = GoldSpan(doc.id, 0, len(doc.text), quoted_text=doc.text)
    return coverage(chunks, whole) == 1.0


def test_semantic_tiles_the_document_and_keeps_offsets():
    embedder = DeterministicEmbedder(dim=1024)
    chunker = SemanticChunker(embedder, breakpoint_percentile=75.0, max_chars=200)
    docs, _ = build_basic_corpus()
    for doc in docs.values():
        chunks = chunker.chunk(doc)
        assert chunks
        assert _covers_whole(chunks, doc)
        for c in chunks:
            assert doc.text[c.start:c.end] == c.text


def test_semantic_respects_max_chars():
    embedder = DeterministicEmbedder(dim=512)
    chunker = SemanticChunker(embedder, max_chars=120)
    long = Document(id="D", text=". ".join(f"sentence number {i} here" for i in range(40)) + ".")
    chunks = chunker.chunk(long)
    assert all(c.length <= 160 for c in chunks)  # cap + one trailing sentence
    assert _covers_whole(chunks, long)


def test_semantic_single_sentence_is_one_chunk():
    embedder = DeterministicEmbedder(dim=64)
    chunks = SemanticChunker(embedder).chunk(Document(id="D", text="just one sentence here"))
    assert len(chunks) == 1


# --------------------------------- Parent-child ---------------------------------------


def test_parent_child_indexes_children_links_parents():
    chunker = ParentChildChunker(parent_size=100, child_size=25)
    doc = Document(id="D", text="x" * 250)
    children = chunker.chunk(doc)
    # Every returned unit is a child of size <= child_size, with a parent link.
    assert all(c.length <= 25 for c in children)
    assert all(c.parent_id is not None for c in children)
    # Expanding maps children to fewer, larger parents.
    parents = chunker.expand(children)
    assert len(parents) < len(children)
    assert all(p.length <= 100 for p in parents)


def test_parent_child_children_tile_and_parents_tile():
    chunker = ParentChildChunker(parent_size=120, child_size=40)
    doc = Document(id="D", text="abcde " * 40)  # 240 chars
    children = chunker.chunk_corpus([doc])
    assert _covers_whole(children, doc)
    assert _covers_whole(chunker.expand(children), doc)


def test_parent_child_returns_parent_context_end_to_end():
    # A child matches the query, but the gold answer needs the wider parent to be covered.
    docs, _ = build_basic_corpus()
    chunker = ParentChildChunker(parent_size=400, child_size=60)
    chunks = chunker.chunk_corpus(docs.values())
    embedder = DeterministicEmbedder(dim=2048)
    dense = DenseRetriever(embedder).index(chunks)
    config = Config("det", chunker.spec, "dense", top_k=3, candidate_n=20)
    pipe = RetrievalPipeline(chunks, config, dense=dense, return_expander=chunker.expand)

    from retrieval_lab.corpora.constructed import build_basic_corpus as _b
    _, queries = _b()
    # The whole-sentence gold is recovered because the returned parent is large enough.
    res = evaluate_query(queries[0], pipe)  # "What does a 404 status code mean?"
    assert res.hit


def test_parent_child_rejects_bad_sizes():
    with pytest.raises(ValueError):
        ParentChildChunker(parent_size=100, child_size=200)
