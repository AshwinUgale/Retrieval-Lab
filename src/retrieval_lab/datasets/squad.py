"""SQuAD importer — the real dataset that fits source-span gold almost exactly (spec §I.8).

SQuAD gives, per question, a ``context`` paragraph and one or more answers as
``(text, answer_start)`` — a character offset into that context. That is precisely a
``GoldSpan``. A question's several human answers become **alternatives** (any acceptable), and
SQuAD 2.0 ``is_impossible`` questions (no answer) are skipped.

The file is a download the user provides (no automatic fetch — network/licensing stay out of
the core). Identical contexts are de-duplicated into one ``Document``. Every answer is
verified against the context as it is imported; answers that don't match (rare, from
re-processed copies) are skipped rather than trusted, and a question with no verifying answer
is dropped — the tool never scores against an offset it couldn't confirm.

    from retrieval_lab.datasets import load_squad
    imp = load_squad("dev-v2.0.json", max_queries=500)
    imp.write_jsonl("./squad")        # -> docs.jsonl + queries.jsonl, ready for `run`
"""

from __future__ import annotations

import json
from pathlib import Path

from retrieval_lab.datasets.base import DatasetImport
from retrieval_lab.gold import EvidenceSet, GoldAnswer, GoldSpan, Query
from retrieval_lab.hashing import content_hash, stable_hash
from retrieval_lab.models import Document


def load_squad(
    path: str | Path,
    max_queries: int | None = None,
    max_answers_per_query: int | None = None,
) -> DatasetImport:
    """Import a SQuAD v1.1 / v2.0 JSON file into a ``DatasetImport``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    articles = data["data"] if isinstance(data, dict) else data

    documents: dict[str, Document] = {}
    queries: list[Query] = []
    skipped = {"impossible": 0, "unverified": 0, "no_answer": 0}

    for article in articles:
        title = article.get("title")
        for para in article.get("paragraphs", []):
            context = para["context"]
            doc_id = "sq-" + stable_hash(context)[:16]  # identical contexts share a doc
            if doc_id not in documents:
                documents[doc_id] = Document(id=doc_id, text=context, meta={"title": title})
            version = content_hash(context)

            for qa in para.get("qas", []):
                if qa.get("is_impossible"):
                    skipped["impossible"] += 1
                    continue
                answers = qa.get("answers") or []
                if not answers:
                    skipped["no_answer"] += 1
                    continue

                seen: set[tuple[int, str]] = set()
                alts: list[EvidenceSet] = []
                for a in answers:
                    text = a.get("text")
                    start = a.get("answer_start")
                    if not text or start is None:
                        continue
                    end = start + len(text)
                    if context[start:end] != text:  # verify the offset (fail closed per-answer)
                        continue
                    key = (start, text)
                    if key in seen:
                        continue
                    seen.add(key)
                    alts.append(EvidenceSet((GoldSpan(doc_id, start, end, text, version),)))
                    if max_answers_per_query and len(alts) >= max_answers_per_query:
                        break

                if not alts:
                    skipped["unverified"] += 1
                    continue
                queries.append(Query(id=str(qa["id"]), text=qa["question"],
                                     gold=GoldAnswer(tuple(alts)), meta={"dataset": "squad"}))
                if max_queries and len(queries) >= max_queries:
                    return DatasetImport(documents, queries, skipped)

    return DatasetImport(documents, queries, skipped)
