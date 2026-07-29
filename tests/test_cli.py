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


def test_unknown_embed_model_is_an_error(demo):
    docs, queries, _tmp = demo
    with pytest.raises(ValueError):
        main(["run", "--corpus", str(docs), "--queries", str(queries),
              "--embed-models", "nonexistent", "--min-sample", "1"])
