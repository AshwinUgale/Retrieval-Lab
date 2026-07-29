"""A small constructed corpus with known answer spans.

Built programmatically so gold offsets and ``quoted_text`` are exact by construction (the
span is located in the source, not hand-typed). This basic corpus is enough to exercise the
Phase 1 dense slice end to end; Phase 2 adds a corpus with *planted* stage failures for the
§I.13 attribution recovery suite.
"""

from __future__ import annotations

from retrieval_lab.gold import EvidenceSet, GoldAnswer, GoldSpan, Query
from retrieval_lab.hashing import content_hash
from retrieval_lab.models import Document

_DOCS: dict[str, str] = {
    "D1": (
        "Photosynthesis in plants. Plants convert sunlight, water, and carbon dioxide "
        "into glucose and oxygen. Plants make energy from sunlight through a process "
        "called photosynthesis, which happens in the chloroplasts of the leaf."
    ),
    "D2": (
        "Cell biology basics. The nucleus stores the cell's genetic material. "
        "The mitochondria is the powerhouse of the cell, producing ATP that the cell "
        "uses for energy. Ribosomes assemble proteins from amino acids."
    ),
    "D3": (
        "HTTP status codes. A 200 status code means the request succeeded. "
        "A 404 status code means the requested resource was not found on the server. "
        "A 500 status code indicates an internal server error."
    ),
    "D4": (
        "Cooking pasta. Bring a large pot of salted water to a rolling boil. "
        "Add the pasta and stir occasionally so it does not stick. "
        "Drain the pasta once it is al dente, usually after eight to ten minutes."
    ),
}


def build_basic_corpus() -> tuple[dict[str, Document], list[Query]]:
    """Return ``(documents, queries)`` for the basic constructed corpus.

    Each query's gold is a single source span whose text lexically overlaps the query, so a
    dense retriever over the keyless embedder should retrieve the right document.
    """
    documents = {doc_id: Document(id=doc_id, text=text) for doc_id, text in _DOCS.items()}

    def q(qid: str, text: str, source_id: str, needle: str) -> Query:
        return Query(id=qid, text=text, gold=_gold_of(documents[source_id], needle))

    queries = [
        q("Q1", "What does a 404 status code mean?", "D3",
          "A 404 status code means the requested resource was not found on the server."),
        q("Q2", "What is the powerhouse of the cell?", "D2",
          "The mitochondria is the powerhouse of the cell, producing ATP that the cell "
          "uses for energy."),
        q("Q3", "How do plants make energy from sunlight?", "D1",
          "Plants make energy from sunlight through a process called photosynthesis, "
          "which happens in the chloroplasts of the leaf."),
        q("Q4", "How long do you cook pasta for?", "D4",
          "Drain the pasta once it is al dente, usually after eight to ten minutes."),
    ]
    return documents, queries


def span_in(doc: Document, needle: str) -> GoldSpan:
    """Locate ``needle`` in ``doc`` and return a version-stamped gold span for it."""
    start = doc.text.index(needle)
    return GoldSpan(
        source_id=doc.id,
        start=start,
        end=start + len(needle),
        quoted_text=needle,
        source_version=content_hash(doc.text),
    )


def _gold_of(doc: Document, needle: str) -> GoldAnswer:
    return GoldAnswer((EvidenceSet((span_in(doc, needle),)),))
