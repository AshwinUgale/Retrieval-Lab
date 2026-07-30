"""BEIR importer — passage-level qrels → whole-passage gold (spec §I.8, with a caveat).

BEIR ships three files: ``corpus.jsonl`` (``{_id, title, text}``), ``queries.jsonl``
(``{_id, text}``), and a ``qrels`` TSV (``query-id  corpus-id  score``). Relevance is
**passage-level** (a whole passage is relevant), not a character span.

We map each relevant passage to a gold span covering that passage's whole document, and a
query's several relevant passages to **alternatives** (retrieving any one satisfies — the
recall notion). Caveat, disclosed here and worth remembering: Retrieval Lab's hit is
*coverage-based* (≥ the threshold of the gold span), so with a chunker that splits a passage,
"hit" means "retrieved enough of the relevant passage," which is stricter than BEIR's
passage-level qrel and not identical to BEIR's nDCG@k. Best used for compatibility testing;
keep passages as whole chunks (a large ``fixed`` chunk size) for the closest correspondence.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from retrieval_lab.datasets.base import DatasetImport
from retrieval_lab.gold import EvidenceSet, GoldAnswer, GoldSpan, Query
from retrieval_lab.hashing import content_hash
from retrieval_lab.models import Document


def _passage_text(record: dict) -> str:
    title = (record.get("title") or "").strip()
    text = (record.get("text") or "").strip()
    return f"{title}\n\n{text}".strip() if title else text


def load_beir(
    corpus_path: str | Path,
    queries_path: str | Path,
    qrels_path: str | Path,
    min_score: int = 1,
    max_queries: int | None = None,
) -> DatasetImport:
    """Import a BEIR-format dataset (corpus + queries + qrels) into a ``DatasetImport``."""
    # corpus: id -> Document (only passages referenced by kept qrels are emitted below)
    corpus: dict[str, str] = {}
    for line in _iter_jsonl(corpus_path):
        corpus[str(line["_id"])] = _passage_text(line)

    query_text: dict[str, str] = {str(q["_id"]): q["text"] for q in _iter_jsonl(queries_path)}

    # qrels: query-id -> [relevant corpus-id]
    relevant: dict[str, list[str]] = defaultdict(list)
    for qid, cid, score in _iter_qrels(qrels_path):
        if score >= min_score and cid in corpus:
            relevant[qid].append(cid)

    documents: dict[str, Document] = {}
    queries: list[Query] = []
    skipped = {"no_relevant": 0, "unknown_query": 0}

    for qid, cids in relevant.items():
        if qid not in query_text:
            skipped["unknown_query"] += 1
            continue
        alts: list[EvidenceSet] = []
        for cid in dict.fromkeys(cids):  # dedup, keep order
            text = corpus[cid]
            if not text:
                continue
            if cid not in documents:
                documents[cid] = Document(id=cid, text=text)
            span = GoldSpan(cid, 0, len(text), quoted_text=text,
                            source_version=content_hash(text))
            alts.append(EvidenceSet((span,)))
        if not alts:
            skipped["no_relevant"] += 1
            continue
        queries.append(Query(id=qid, text=query_text[qid], gold=GoldAnswer(tuple(alts)),
                             meta={"dataset": "beir"}))
        if max_queries and len(queries) >= max_queries:
            break

    return DatasetImport(documents, queries, skipped)


def _iter_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                yield json.loads(raw)


def _iter_qrels(path: str | Path):
    """Yield ``(query_id, corpus_id, score)`` from a BEIR qrels TSV, skipping the header."""
    with Path(path).open("r", encoding="utf-8") as fh:
        for raw in fh:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, cid, score = parts[0], parts[1], parts[2]
            try:
                score_i = int(score)
            except ValueError:
                continue  # header row ("query-id\tcorpus-id\tscore")
            yield qid, cid, score_i
