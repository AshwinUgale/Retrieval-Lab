"""Constructed corpora with known answer spans, for validation (spec §I.13)."""

from retrieval_lab.corpora.constructed import (
    build_basic_corpus,
    dump_basic_corpus_jsonl,
    span_in,
)

__all__ = ["build_basic_corpus", "dump_basic_corpus_jsonl", "span_in"]
