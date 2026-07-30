"""Constructed corpora with known answer spans, for validation (spec §I.13)."""

from retrieval_lab.corpora.constructed import (
    build_basic_corpus,
    dump_basic_corpus_jsonl,
    span_in,
)
from retrieval_lab.corpora.realistic import (
    build_realistic_corpus,
    dump_realistic_corpus_jsonl,
)

__all__ = [
    "build_basic_corpus",
    "dump_basic_corpus_jsonl",
    "span_in",
    "build_realistic_corpus",
    "dump_realistic_corpus_jsonl",
]
