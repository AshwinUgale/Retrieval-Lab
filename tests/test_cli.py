"""Phase 6 — the CLI end to end (spec §I.12)."""

import json

import pytest

from retrieval_lab.cli import EXIT_INPUT_ERROR, EXIT_OK, EXIT_QUALITY_GATE, main
from retrieval_lab.corpora.constructed import dump_basic_corpus_jsonl


@pytest.fixture
def demo(tmp_path):
    docs, queries = dump_basic_corpus_jsonl(tmp_path)
    return docs, queries, tmp_path


def test_run_writes_json_and_exits_ok(demo, capsys):
    docs, queries, tmp = demo
    out = tmp / "out.json"
    code = main([
        "run", "--corpus", str(docs), "--queries", str(queries),
        "--embed-models", "det", "--chunkers", "fixed,recursive",
        "--retrieval", "dense,hybrid", "--top-k", "3", "--candidate-n", "10",
        "--min-sample", "1", "--json", str(out),
    ])
    assert code == EXIT_OK
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n_queries"] == 4
    assert len(data["metrics"]) == 4  # 1 embed x 2 chunk x 2 mode
    report = capsys.readouterr().out
    assert "Retrieval Lab" in report


def test_explain_and_pareto_read_the_run_json(demo, capsys):
    docs, queries, tmp = demo
    out = tmp / "out.json"
    main(["run", "--corpus", str(docs), "--queries", str(queries),
          "--min-sample", "1", "--json", str(out)])
    capsys.readouterr()

    assert main(["explain", "--json", str(out), "--query-id", "Q1"]) == EXIT_OK
    assert "Q1" in capsys.readouterr().out

    assert main(["pareto", "--json", str(out)]) == EXIT_OK
    assert "Pareto" in capsys.readouterr().out


def test_fail_under_gate_exits_nonzero(demo, capsys):
    docs, queries, tmp = demo
    # An impossible quality bar -> CI gate trips.
    code = main(["run", "--corpus", str(docs), "--queries", str(queries),
                 "--min-sample", "1", "--fail-under", "1.01"])
    assert code == EXIT_QUALITY_GATE


def test_fail_under_fails_closed_when_verdict_is_suppressed(demo, capsys):
    docs, queries, _tmp = demo
    code = main([
        "run", "--corpus", str(docs), "--queries", str(queries),
        "--min-sample", "100", "--fail-under", "0",
    ])
    assert code == EXIT_QUALITY_GATE
    assert "cannot evaluate --fail-under" in capsys.readouterr().err


def test_sparse_only_does_not_load_optional_embedding_model(demo):
    docs, queries, _tmp = demo
    # This succeeds without sentence-transformers because sparse retrieval does not use e5.
    code = main([
        "run", "--corpus", str(docs), "--queries", str(queries),
        "--retrieval", "sparse", "--embed-models", "e5", "--min-sample", "1",
    ])
    assert code == EXIT_OK


def test_hnsw_missing_extra_is_a_clean_input_error(demo, capsys, monkeypatch):
    import retrieval_lab.sweep as sweep_module

    class MissingHNSW:
        def __init__(self, *_args, **_kwargs):
            pass

        def index(self, _chunks):
            raise ImportError("install retrieval-lab[ann]")

    monkeypatch.setattr(sweep_module, "ANNDenseRetriever", MissingHNSW)
    docs, queries, _tmp = demo
    code = main([
        "run", "--corpus", str(docs), "--queries", str(queries),
        "--dense-index", "hnsw", "--min-sample", "1",
    ])
    assert code == EXIT_INPUT_ERROR
    assert "retrieval-lab[ann]" in capsys.readouterr().err


def test_offset_drift_fails_closed(demo, tmp_path):
    docs, _queries, _tmp = demo
    # A queries file whose quoted_text does not match the source -> fail closed, exit 2.
    bad = tmp_path / "bad_queries.jsonl"
    bad.write_text(json.dumps({
        "id": "QX", "text": "q",
        "gold": {"alternatives": [{"required_spans": [
            {"source_id": "D3", "start": 0, "end": 5, "quoted_text": "WRONG"}
        ]}]},
    }) + "\n", encoding="utf-8")
    code = main(["run", "--corpus", str(docs), "--queries", str(bad), "--min-sample", "1"])
    assert code == EXIT_INPUT_ERROR


def test_unknown_embed_model_is_an_error(demo, capsys):
    docs, queries, _tmp = demo
    code = main(["run", "--corpus", str(docs), "--queries", str(queries),
                 "--embed-models", "nonexistent", "--min-sample", "1"])
    assert code == EXIT_INPUT_ERROR
    assert "unknown embed model" in capsys.readouterr().err


def test_run_writes_html(demo, tmp_path):
    docs, queries, _tmp = demo
    html = tmp_path / "report.html"
    code = main(["run", "--corpus", str(docs), "--queries", str(queries),
                 "--min-sample", "1", "--html", str(html)])
    assert code == EXIT_OK
    assert html.exists()
    assert html.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_demo_command_runs_end_to_end(tmp_path):
    out = tmp_path / "demo"
    code = main(["demo", "--dataset", "basic", "--out-dir", str(out)])
    assert code == EXIT_OK
    assert (out / "docs.jsonl").exists()
    assert (out / "queries.jsonl").exists()
    assert (out / "out.json").exists()
    assert (out / "report.html").exists()


def test_make_gold_converts_authoring_spec_to_verified_queries(tmp_path):
    docs = tmp_path / "docs.jsonl"
    docs.write_text(json.dumps({"id": "D1", "text": "Bread bakes at 220 degrees Celsius."})
                    + "\n", encoding="utf-8")
    spec = tmp_path / "spec.jsonl"
    spec.write_text(json.dumps({"id": "Q1", "text": "temp?", "source_id": "D1",
                                "answer": "220 degrees Celsius"}) + "\n", encoding="utf-8")
    out = tmp_path / "queries.jsonl"
    code = main(["make-gold", "--corpus", str(docs), "--spec", str(spec), "--out", str(out)])
    assert code == EXIT_OK
    # The produced queries then run cleanly through `run` (fail-closed gold verification).
    assert main(["run", "--corpus", str(docs), "--queries", str(out),
                 "--min-sample", "1"]) == EXIT_OK


def test_make_gold_bad_quote_fails_closed(tmp_path):
    docs = tmp_path / "docs.jsonl"
    docs.write_text(json.dumps({"id": "D1", "text": "hello world"}) + "\n", encoding="utf-8")
    spec = tmp_path / "spec.jsonl"
    spec.write_text(json.dumps({"id": "Q1", "text": "q", "source_id": "D1",
                                "answer": "not present"}) + "\n", encoding="utf-8")
    out = tmp_path / "queries.jsonl"
    code = main(["make-gold", "--corpus", str(docs), "--spec", str(spec), "--out", str(out)])
    assert code == EXIT_INPUT_ERROR  # the typo is caught at authoring time
