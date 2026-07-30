"""Post-build #3 — gold authoring from quotes, not offsets (spec §I.8)."""

import json

import pytest

from retrieval_lab.authoring import (
    build_gold,
    build_query,
    load_authoring_spec,
    make_span,
    write_queries_jsonl,
)
from retrieval_lab.gold import load_queries, satisfies_gold, verify_query
from retrieval_lab.hashing import content_hash
from retrieval_lab.models import Chunk, Document


def _doc(text: str, doc_id: str = "D1") -> Document:
    return Document(id=doc_id, text=text)


def _chunk_covering(doc: Document) -> Chunk:
    return Chunk.make(doc.id, 0, len(doc.text), doc.text, chunker_spec="t")


def test_make_span_stamps_exact_offsets_and_version():
    doc = _doc("the quick brown fox jumps")
    span = make_span(doc, "brown fox")
    assert doc.text[span.start:span.end] == "brown fox"
    assert span.quoted_text == "brown fox"
    assert span.source_version == content_hash(doc.text)


def test_make_span_raises_on_missing_quote():
    with pytest.raises(ValueError, match="not found"):
        make_span(_doc("hello world"), "goodbye")


def test_make_span_selects_the_requested_occurrence():
    doc = _doc("go left then go right then go home")
    first = make_span(doc, "go ", occurrence=0)
    third = make_span(doc, "go ", occurrence=2)
    assert first.start < third.start
    assert doc.text[third.start:third.end] == "go "


def test_build_gold_single_answer():
    doc = _doc("Bread bakes at 220 degrees.")
    docs = {"D1": doc}
    gold = build_gold({"source_id": "D1", "answer": "220 degrees"}, docs)
    assert satisfies_gold([_chunk_covering(doc)], gold)


def test_build_gold_answer_all_is_a_conjunction():
    doc = _doc("Name is required. Dimension is required.")
    docs = {"D1": doc}
    gold = build_gold(
        {"source_id": "D1", "answer_all": ["Name is required", "Dimension is required"]}, docs
    )
    # Both spans present -> satisfied; only one -> not.
    assert satisfies_gold([_chunk_covering(doc)], gold)
    assert len(gold.alternatives) == 1 and len(gold.alternatives[0].required_spans) == 2


def test_build_gold_alternatives_is_a_disjunction():
    d1, d2 = _doc("Use an API key.", "D1"), _doc("Or use OAuth.", "D2")
    docs = {"D1": d1, "D2": d2}
    gold = build_gold({"alternatives": [
        {"source_id": "D1", "answer": "API key"},
        {"source_id": "D2", "answer": "OAuth"},
    ]}, docs)
    assert satisfies_gold([Chunk.make("D2", 0, len(d2.text), d2.text, "t")], gold)  # 2nd alt alone


def test_build_query_verifies_and_round_trips_through_loader(tmp_path):
    docs = {"D1": _doc("The activation code is ZQ-4471 exactly.")}
    q = build_query({"id": "Q1", "text": "code?", "source_id": "D1", "answer": "ZQ-4471"}, docs)
    verify_query(q, docs)  # no raise
    out = write_queries_jsonl([q], tmp_path / "queries.jsonl")
    reloaded = load_queries(out, docs, strict=True)  # fail-closed loader accepts it
    assert reloaded[0].id == "Q1"


def test_load_authoring_spec_end_to_end(tmp_path):
    docs = {"D1": _doc("Alpha beta gamma delta epsilon.")}
    spec = tmp_path / "spec.jsonl"
    spec.write_text(
        json.dumps({"id": "Q1", "text": "q", "source_id": "D1", "answer": "gamma delta"}) + "\n",
        encoding="utf-8",
    )
    queries = load_authoring_spec(spec, docs)
    assert len(queries) == 1
    assert satisfies_gold([_chunk_covering(docs["D1"])], queries[0].gold)


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="unknown source"):
        build_gold({"source_id": "MISSING", "answer": "x"}, {"D1": _doc("x")})
