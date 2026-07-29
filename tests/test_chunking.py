"""Phase 1 — chunkers tile the document and keep exact offsets (spec §I.7)."""

import pytest

from retrieval_lab.chunking import FixedSizeChunker, RecursiveChunker
from retrieval_lab.corpora.constructed import build_basic_corpus
from retrieval_lab.gold import GoldSpan, coverage
from retrieval_lab.models import Document

CHUNKERS = [
    FixedSizeChunker(chunk_size=80, overlap=0),
    FixedSizeChunker(chunk_size=80, overlap=20),
    RecursiveChunker(chunk_size=80),
]


@pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: c.spec)
def test_chunks_tile_the_whole_document(chunker):
    docs, _ = build_basic_corpus()
    for doc in docs.values():
        chunks = chunker.chunk(doc)
        assert chunks, "expected at least one chunk"
        # The union of all chunks must cover the entire document (representation stage can
        # then only fail on genuine text loss, never on a boundary artifact).
        whole = GoldSpan(doc.id, 0, len(doc.text), quoted_text=doc.text)
        assert coverage(chunks, whole) == 1.0


@pytest.mark.parametrize("chunker", CHUNKERS, ids=lambda c: c.spec)
def test_chunk_text_matches_source_offsets(chunker):
    docs, _ = build_basic_corpus()
    for doc in docs.values():
        for c in chunker.chunk(doc):
            assert c.source_id == doc.id
            assert doc.text[c.start : c.end] == c.text
            assert 0 <= c.start < c.end <= len(doc.text)


def test_fixed_respects_size_and_overlap_step():
    doc = Document(id="D", text="abcdefghij" * 5)  # 50 chars
    chunks = FixedSizeChunker(chunk_size=20, overlap=5).chunk(doc)
    # step = 15 -> starts 0, 15, 30; the window at 30 reaches the end, so no redundant
    # tail chunk at 45 is emitted.
    assert [c.start for c in chunks] == [0, 15, 30]
    assert chunks[0].end == 20
    assert chunks[-1].end == 50


def test_recursive_prefers_paragraph_then_line_boundaries():
    text = "Para one line.\n\nPara two is a bit longer than the first paragraph here."
    chunks = RecursiveChunker(chunk_size=30).chunk(Document(id="D", text=text))
    # Still tiles the whole document.
    whole = GoldSpan("D", 0, len(text), quoted_text=text)
    assert coverage(chunks, whole) == 1.0
    assert all(c.length <= 60 for c in chunks)


def test_invalid_chunker_params_rejected():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=0)
