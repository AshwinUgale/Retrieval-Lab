"""Ground truth: source-span gold + the single coverage predicate (spec §I.6–I.8).

This module is the tool's foundation. Two decisions live here and everything else depends
on them:

1. **Gold is defined over source-document character spans, not chunk indices** (spec §I.8).
   Changing the chunker changes the chunks, so "chunk #47" has no cross-config meaning. A
   hit is recomputed by *coverage* over whichever chunks a config retrieved, independently
   per chunker — which is what makes cross-config comparison valid.

2. **One predicate, reused everywhere.** ``satisfies_gold`` is the *only* hit test. The
   scorer, the DAG attribution engine, and ``gold_completion_rank`` all call it, so they can
   never disagree — two chunks each covering half of a span jointly satisfy it, and every
   consumer sees that identically (spec §I.7).

Coverage is computed over the **union** of retrieved chunks, not per-chunk, because chunkers
partition all text: almost every span overlaps *some* chunk, so an "any overlap" rule would
detect only text loss, not fragmentation (spec §I.7).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from retrieval_lab.hashing import content_hash, stable_hash
from retrieval_lab.models import Chunk, Document

DEFAULT_MIN_GOLD_COVERAGE = 0.8
"""Fraction of a required span that must be covered for it to count as satisfied (spec §I.7)."""


class OffsetVerificationError(Exception):
    """Raised when a gold span fails offset/version verification.

    Fail closed (spec §I.8, §I.11): never score against drifted offsets — refuse the query.
    """


# --------------------------------------------------------------------------------------
# Gold types (spec §I.6)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldSpan:
    """A required region of a source document.

    ``source_version`` is the content hash of the source at label time and ``quoted_text``
    is the exact ``source[start:end]`` captured then; both are verified at load so silent
    offset drift (Unicode/newline normalization, re-extraction, edits) is caught.
    """

    source_id: str
    start: int
    end: int
    quoted_text: str
    source_version: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid gold span [{self.start}, {self.end})")

    @property
    def id(self) -> str:
        """Stable identity of this span (position within its source)."""
        return stable_hash(self.source_id, self.start, self.end)

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class EvidenceSet:
    """A conjunction: ALL required spans must be satisfied to answer via this evidence."""

    required_spans: tuple[GoldSpan, ...]

    def __post_init__(self) -> None:
        if not self.required_spans:
            raise ValueError("EvidenceSet needs at least one required span")


@dataclass(frozen=True)
class GoldAnswer:
    """A disjunction: ANY alternative EvidenceSet suffices (spec §I.8, non-unique gold)."""

    alternatives: tuple[EvidenceSet, ...]

    def __post_init__(self) -> None:
        if not self.alternatives:
            raise ValueError("GoldAnswer needs at least one alternative EvidenceSet")

    def all_required_spans(self) -> list[GoldSpan]:
        """Every required span across all alternatives, de-duplicated by span id."""
        seen: dict[str, GoldSpan] = {}
        for alt in self.alternatives:
            for span in alt.required_spans:
                seen.setdefault(span.id, span)
        return list(seen.values())


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    gold: GoldAnswer
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Coverage math — the single predicate (spec §I.7)
# --------------------------------------------------------------------------------------


def _union_length(intervals: list[tuple[int, int]]) -> int:
    """Total length covered by a set of half-open ``[s, e)`` intervals after union."""
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total = 0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:  # disjoint from the current run
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:  # overlapping or adjacent — extend the run
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return total


def coverage(items: Iterable[Chunk], span: GoldSpan) -> float:
    """Fraction of ``span`` covered by the union of ``items`` (spec §I.7).

    Only chunks from the same source document contribute, and only the portion that overlaps
    the span. Returns a value in ``[0.0, 1.0]``.
    """
    intervals: list[tuple[int, int]] = []
    for it in items:
        if it.source_id != span.source_id:
            continue
        s = max(it.start, span.start)
        e = min(it.end, span.end)
        if e > s:
            intervals.append((s, e))
    return _union_length(intervals) / span.length


def span_satisfied(
    items: Iterable[Chunk],
    span: GoldSpan,
    min_gold_coverage: float = DEFAULT_MIN_GOLD_COVERAGE,
) -> bool:
    """True iff the union of ``items`` covers ``span`` to at least the threshold."""
    return coverage(items, span) >= min_gold_coverage


def evidence_set_satisfied(
    items: Sequence[Chunk],
    evidence: EvidenceSet,
    min_gold_coverage: float = DEFAULT_MIN_GOLD_COVERAGE,
) -> bool:
    """True iff every required span of ``evidence`` is satisfied by ``items``."""
    return all(span_satisfied(items, span, min_gold_coverage) for span in evidence.required_spans)


def satisfies_gold(
    items: Sequence[Chunk],
    gold: GoldAnswer,
    min_gold_coverage: float = DEFAULT_MIN_GOLD_COVERAGE,
) -> bool:
    """THE hit predicate (spec §I.7): some EvidenceSet has EVERY required span reaching
    ``min_gold_coverage`` via the union-coverage of ``items``.

    Every consumer — scorer, attribution, completion rank — must route through this so they
    cannot disagree.
    """
    return any(
        evidence_set_satisfied(items, alt, min_gold_coverage) for alt in gold.alternatives
    )


def gold_completion_rank(
    ranked: Sequence[Chunk],
    gold: GoldAnswer,
    min_gold_coverage: float = DEFAULT_MIN_GOLD_COVERAGE,
) -> int | None:
    """Smallest ``r`` such that the top-``r`` chunks satisfy an EvidenceSet (spec §I.7).

    Well-defined under multi-chunk / multi-span gold, where a per-chunk "first relevant"
    rank is not. Returns ``None`` if the full ranked list never satisfies gold.
    """
    for r in range(1, len(ranked) + 1):
        if satisfies_gold(ranked[: r], gold, min_gold_coverage):
            return r
    return None


def single_chunk_coverage_by_span(
    items: Iterable[Chunk],
    gold: GoldAnswer,
) -> dict[str, float]:
    """Max coverage any *single* chunk gives each required span, keyed by span id (spec §I.6).

    A span satisfied only across multiple chunks shows up low here — the raw signal behind
    fragmentation detection.
    """
    items = list(items)
    result: dict[str, float] = {}
    for span in gold.all_required_spans():
        result[span.id] = max((coverage([it], span) for it in items), default=0.0)
    return result


def fragmented_spans(
    items: Sequence[Chunk],
    gold: GoldAnswer,
    min_gold_coverage: float = DEFAULT_MIN_GOLD_COVERAGE,
) -> list[str]:
    """Spans whose *union* coverage satisfies gold but whose best *single* chunk falls short.

    These are only reconstructable across multiple chunks — a distinct defect from a clean
    single-chunk hit (spec §I.7). Returned as a list of span ids.
    """
    single = single_chunk_coverage_by_span(items, gold)
    frags: list[str] = []
    for span in gold.all_required_spans():
        union_ok = span_satisfied(items, span, min_gold_coverage)
        single_ok = single.get(span.id, 0.0) >= min_gold_coverage
        if union_ok and not single_ok:
            frags.append(span.id)
    return frags


# --------------------------------------------------------------------------------------
# Offset verification + loading (fail closed, spec §I.8, §I.11)
# --------------------------------------------------------------------------------------


def verify_span(span: GoldSpan, documents: Mapping[str, Document]) -> None:
    """Verify a gold span against the loaded source; raise on any drift (fail closed).

    Checks, in order: the source document exists; if ``source_version`` was recorded, the
    loaded document's content hash matches it; the span offsets are in range; and
    ``source[start:end] == quoted_text``.
    """
    doc = documents.get(span.source_id)
    if doc is None:
        raise OffsetVerificationError(
            f"gold references unknown source document {span.source_id!r}"
        )
    if span.source_version is not None:
        actual_version = content_hash(doc.text)
        if actual_version != span.source_version:
            raise OffsetVerificationError(
                f"source {span.source_id!r} changed since labeling: "
                f"version {actual_version[:12]}… != recorded {span.source_version[:12]}…"
            )
    if span.end > len(doc.text):
        raise OffsetVerificationError(
            f"gold span [{span.start}, {span.end}) out of range for source "
            f"{span.source_id!r} (len {len(doc.text)})"
        )
    actual = doc.text[span.start : span.end]
    if actual != span.quoted_text:
        raise OffsetVerificationError(
            f"quoted_text mismatch in {span.source_id!r} at [{span.start}, {span.end}): "
            f"source has {actual!r} but gold recorded {span.quoted_text!r}"
        )


def verify_query(query: Query, documents: Mapping[str, Document]) -> None:
    """Verify every gold span of a query; raise ``OffsetVerificationError`` on first failure."""
    for span in query.gold.all_required_spans():
        verify_span(span, documents)


# ------------------------------- serialization ----------------------------------------


def _gold_span_from_dict(d: dict) -> GoldSpan:
    return GoldSpan(
        source_id=d["source_id"],
        start=int(d["start"]),
        end=int(d["end"]),
        quoted_text=d["quoted_text"],
        source_version=d.get("source_version"),
    )


def _gold_from_dict(d: dict) -> GoldAnswer:
    alternatives = tuple(
        EvidenceSet(tuple(_gold_span_from_dict(s) for s in alt["required_spans"]))
        for alt in d["alternatives"]
    )
    return GoldAnswer(alternatives)


def query_from_dict(d: dict) -> Query:
    """Build a ``Query`` from a plain dict (one JSONL record)."""
    return Query(
        id=str(d["id"]),
        text=d["text"],
        gold=_gold_from_dict(d["gold"]),
        meta=d.get("meta", {}),
    )


def _gold_span_to_dict(s: GoldSpan) -> dict:
    d = {"source_id": s.source_id, "start": s.start, "end": s.end, "quoted_text": s.quoted_text}
    if s.source_version is not None:
        d["source_version"] = s.source_version
    return d


def gold_to_dict(gold: GoldAnswer) -> dict:
    return {
        "alternatives": [
            {"required_spans": [_gold_span_to_dict(s) for s in alt.required_spans]}
            for alt in gold.alternatives
        ]
    }


def query_to_dict(query: Query) -> dict:
    """Serialize a ``Query`` back to a JSONL-ready dict (inverse of ``query_from_dict``)."""
    d = {"id": query.id, "text": query.text, "gold": gold_to_dict(query.gold)}
    if query.meta:
        d["meta"] = query.meta
    return d


def load_documents(path: str | Path) -> dict[str, Document]:
    """Load a corpus from JSONL records ``{"id", "text", "meta"?}`` into an id→Document map."""
    documents: dict[str, Document] = {}
    for line in _iter_jsonl(path):
        doc = Document(id=str(line["id"]), text=line["text"], meta=line.get("meta", {}))
        documents[doc.id] = doc
    return documents


def write_documents_jsonl(documents: Iterable[Document], path: str | Path) -> Path:
    """Write documents to a ``docs.jsonl`` (``{"id","text","meta"?}`` per line)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for doc in documents:
            record = {"id": doc.id, "text": doc.text}
            if doc.meta:
                record["meta"] = doc.meta
            fh.write(json.dumps(record) + "\n")
    return p


def load_queries(
    path: str | Path,
    documents: Mapping[str, Document],
    strict: bool = True,
) -> list[Query]:
    """Load queries from JSONL and verify their gold against ``documents`` (fail closed).

    With ``strict=True`` (default) an offset/version mismatch raises immediately. With
    ``strict=False`` the failing query is skipped (a caller that wants per-query refusal
    should verify explicitly and mark ``QueryResult.refused`` instead).
    """
    queries: list[Query] = []
    for line in _iter_jsonl(path):
        query = query_from_dict(line)
        try:
            verify_query(query, documents)
        except OffsetVerificationError:
            if strict:
                raise
            continue
        queries.append(query)
    return queries


def _iter_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
