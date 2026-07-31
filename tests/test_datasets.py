"""Later — real-dataset importers, tested offline with tiny inline fixtures (spec §I.8)."""

import json

from retrieval_lab.datasets import load_beir, load_squad
from retrieval_lab.gold import load_documents, load_queries, satisfies_gold, verify_query
from retrieval_lab.models import Chunk


def _whole_chunk(doc):
    return Chunk.make(doc.id, 0, len(doc.text), doc.text, chunker_spec="t")


# --------------------------------- SQuAD ----------------------------------------------


def _write_squad(tmp_path):
    ctx1 = "The Amazon rainforest is the largest tropical rainforest on Earth."
    ctx2 = "Water boils at 100 degrees Celsius at sea level."
    def ans(ctx, text):
        return {"text": text, "answer_start": ctx.index(text)}

    data = {"data": [{"title": "T", "paragraphs": [
        {"context": ctx1, "qas": [
            {"id": "q1", "question": "Largest rainforest?",
             "answers": [ans(ctx1, "The Amazon rainforest"), ans(ctx1, "Amazon rainforest")]},
            {"id": "q2", "question": "unanswerable", "is_impossible": True, "answers": []},
        ]},
        {"context": ctx2, "qas": [
            {"id": "q3", "question": "Boiling point?",
             "answers": [ans(ctx2, "100 degrees Celsius")]},
            {"id": "q4", "question": "bad offset", "answers": [
                {"text": "NONMATCH", "answer_start": 0}]},  # won't verify -> skipped
        ]},
    ]}]}
    p = tmp_path / "squad.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_squad_import_maps_answers_and_skips_bad_records(tmp_path):
    imp = load_squad(_write_squad(tmp_path))
    assert len(imp.documents) == 2                      # two distinct contexts
    assert {q.id for q in imp.queries} == {"q1", "q3"}  # q2 impossible, q4 unverified
    assert imp.skipped["impossible"] == 1
    assert imp.skipped["unverified"] == 1


def test_squad_multiple_answers_become_alternatives(tmp_path):
    imp = load_squad(_write_squad(tmp_path))
    q1 = next(q for q in imp.queries if q.id == "q1")
    assert len(q1.gold.alternatives) == 2  # two distinct human answers
    docs = imp.documents
    verify_query(q1, docs)                  # gold offsets verify against the context
    src = next(d for d in docs.values() if "Amazon" in d.text)
    assert satisfies_gold([_whole_chunk(src)], q1.gold)


def test_squad_write_jsonl_round_trips_through_fail_closed_loader(tmp_path):
    imp = load_squad(_write_squad(tmp_path))
    out = tmp_path / "out"
    docs_path, queries_path = imp.write_jsonl(out)
    documents = load_documents(docs_path)
    queries = load_queries(queries_path, documents, strict=True)  # would raise on drift
    assert len(queries) == len(imp.queries)


def test_squad_max_queries_caps(tmp_path):
    imp = load_squad(_write_squad(tmp_path), max_queries=1)
    assert len(imp.queries) == 1


# --------------------------------- BEIR -----------------------------------------------


def _write_beir(tmp_path):
    corpus = [
        {"_id": "d1", "title": "Amazon", "text": "The Amazon is the largest tropical rainforest."},
        {"_id": "d4", "title": "", "text": "Rainforests are dense; the Amazon is the biggest."},
        {"_id": "d2", "title": "", "text": "Water boils at 100 degrees Celsius."},
        {"_id": "d3", "title": "", "text": "An unrelated passage about bicycles."},
    ]
    queries = [{"_id": "q1", "text": "largest rainforest"},
               {"_id": "q2", "text": "boiling point of water"}]
    qrels = "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td4\t1\nq2\td2\t2\nq2\td3\t0\n"
    cp = tmp_path / "corpus.jsonl"
    cp.write_text("\n".join(json.dumps(c) for c in corpus), encoding="utf-8")
    qp = tmp_path / "queries.jsonl"
    qp.write_text("\n".join(json.dumps(q) for q in queries), encoding="utf-8")
    rp = tmp_path / "qrels.tsv"
    rp.write_text(qrels, encoding="utf-8")
    return cp, qp, rp


def test_beir_import_builds_passage_gold_and_alternatives(tmp_path):
    cp, qp, rp = _write_beir(tmp_path)
    imp = load_beir(cp, qp, rp)
    assert {q.id for q in imp.queries} == {"q1", "q2"}
    q1 = next(q for q in imp.queries if q.id == "q1")
    assert len(q1.gold.alternatives) == 2  # d1 and d4 are both relevant -> alternatives
    # The complete retrieval corpus is retained. Non-relevant d3 is a necessary distractor,
    # even though it does not appear in the query's gold alternatives.
    assert set(imp.documents) == {"d1", "d2", "d3", "d4"}
    assert all(
        span.source_id != "d3"
        for alt in q1.gold.alternatives
        for span in alt.required_spans
    )
    # A chunk covering a relevant passage satisfies the whole-passage gold.
    d1 = imp.documents["d1"]
    assert satisfies_gold([_whole_chunk(d1)], q1.gold)


def test_beir_write_jsonl_round_trips(tmp_path):
    cp, qp, rp = _write_beir(tmp_path)
    imp = load_beir(cp, qp, rp)
    docs_path, queries_path = imp.write_jsonl(tmp_path / "out")
    documents = load_documents(docs_path)
    queries = load_queries(queries_path, documents, strict=True)
    assert len(queries) == 2


# --------------------------------- CLI ------------------------------------------------


def test_cli_import_squad_then_run(tmp_path):
    from retrieval_lab.cli import EXIT_OK, main

    squad = _write_squad(tmp_path)
    out = tmp_path / "sq"
    assert main(["import-squad", "--input", str(squad), "--out-dir", str(out)]) == EXIT_OK
    assert main(["run", "--corpus", str(out / "docs.jsonl"),
                 "--queries", str(out / "queries.jsonl"), "--min-sample", "1"]) == EXIT_OK


def test_cli_import_beir_then_run(tmp_path):
    from retrieval_lab.cli import EXIT_OK, main

    cp, qp, rp = _write_beir(tmp_path)
    out = tmp_path / "beir"
    assert main(["import-beir", "--corpus", str(cp), "--queries", str(qp),
                 "--qrels", str(rp), "--out-dir", str(out)]) == EXIT_OK
    assert main(["run", "--corpus", str(out / "docs.jsonl"),
                 "--queries", str(out / "queries.jsonl"), "--min-sample", "1"]) == EXIT_OK
