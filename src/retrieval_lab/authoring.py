"""Gold authoring helpers — write gold from answer *quotes*, not char offsets (spec §I.8).

Authoring gold is the hardest user-facing part of the tool: a ``GoldSpan`` needs exact
character offsets, the quoted text, and a ``source_version`` content hash. Computing offsets
by hand is error-prone and the tool then (correctly) fails closed on any mistake.

These helpers let you supply just the **answer quote(s)**; the offsets and version hash are
located in the source and stamped automatically, and the result is verified before it is
returned — so a produced ``queries.jsonl`` always loads cleanly. An offset-free *authoring
spec* (one JSON object per line) is converted to verified ``Query`` objects:

    {"id": "Q1", "text": "...", "source_id": "D1", "answer": "the exact answer sentence"}
    {"id": "Q2", "text": "...", "source_id": "D1", "answer_all": ["span A", "span B"]}
    {"id": "Q3", "text": "...", "alternatives": [
        {"source_id": "D1", "answer": "..."},
        {"source_id": "D2", "answer_all": ["...", "..."]}]}

``answer`` = one required span; ``answer_all`` = several required spans (conjunction, an
EvidenceSet); ``alternatives`` = several acceptable EvidenceSets (disjunction). If none is
given, ``answer``/``answer_all`` at the top level applies to ``source_id``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from retrieval_lab.gold import EvidenceSet, GoldAnswer, GoldSpan, Query, verify_query
from retrieval_lab.hashing import content_hash
from retrieval_lab.models import Document


def make_span(doc: Document, quote: str, occurrence: int = 0) -> GoldSpan:
    """Locate ``quote`` in ``doc`` (the ``occurrence``-th, 0-based) and stamp a gold span.

    Raises ``ValueError`` if the quote does not occur that many times — surfacing a typo at
    authoring time rather than as a silent mislabel.
    """
    if not quote:
        raise ValueError("answer quote must be non-empty")
    start = -1
    for _ in range(occurrence + 1):
        start = doc.text.find(quote, start + 1)
        if start == -1:
            raise ValueError(
                f"answer quote not found in {doc.id!r} (occurrence {occurrence}): {quote!r}"
            )
    return GoldSpan(
        source_id=doc.id,
        start=start,
        end=start + len(quote),
        quoted_text=quote,
        source_version=content_hash(doc.text),
    )


def _evidence_from(entry: Mapping, documents: Mapping[str, Document]) -> EvidenceSet:
    source_id = entry["source_id"]
    doc = documents.get(source_id)
    if doc is None:
        raise ValueError(f"authoring spec references unknown source {source_id!r}")
    if "answer_all" in entry:
        quotes: Sequence[str] = entry["answer_all"]
    elif "answer" in entry:
        quotes = [entry["answer"]]
    else:
        raise ValueError(f"evidence for {source_id!r} needs 'answer' or 'answer_all'")
    return EvidenceSet(tuple(make_span(doc, q) for q in quotes))


def build_gold(spec: Mapping, documents: Mapping[str, Document]) -> GoldAnswer:
    """Build a verified ``GoldAnswer`` from an offset-free authoring dict."""
    if "alternatives" in spec:
        alts = tuple(_evidence_from(alt, documents) for alt in spec["alternatives"])
        return GoldAnswer(alts)
    return GoldAnswer((_evidence_from(spec, documents),))


def build_query(spec: Mapping, documents: Mapping[str, Document]) -> Query:
    """Build a ``Query`` from an authoring dict and verify its gold (fail closed)."""
    query = Query(
        id=str(spec["id"]),
        text=spec["text"],
        gold=build_gold(spec, documents),
        meta=spec.get("meta", {}),
    )
    verify_query(query, documents)  # belt-and-suspenders; make_span already stamped exactly
    return query


def build_queries(specs: Iterable[Mapping], documents: Mapping[str, Document]) -> list[Query]:
    return [build_query(s, documents) for s in specs]


def load_authoring_spec(path: str | Path, documents: Mapping[str, Document]) -> list[Query]:
    """Read an offset-free authoring spec (JSONL) into verified ``Query`` objects."""
    specs: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                specs.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return build_queries(specs, documents)


def write_queries_jsonl(queries: Iterable[Query], path: str | Path) -> Path:
    """Write verified queries to a canonical ``queries.jsonl`` (with stamped offsets)."""
    from retrieval_lab.gold import query_to_dict

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for query in queries:
            fh.write(json.dumps(query_to_dict(query)) + "\n")
    return p
