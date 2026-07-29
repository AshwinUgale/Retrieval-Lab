"""Deterministic text utilities shared by retrieval, embedding, and budgeting.

Kept tiny and dependency-free so the whole core stays reproducible: the same string always
tokenizes and counts the same way, on any machine, with no model download.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")
# A token proxy for budgeting: words and standalone punctuation each count as one token.
# This is a stable stand-in for a real tokenizer — good enough to compare configs under a
# retrieved-token budget without pulling in a model-specific vocabulary.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def word_tokens(text: str) -> list[str]:
    """Lowercased alphanumeric word tokens — the feature unit for BM25 and hashed embeddings."""
    return _WORD_RE.findall(text.lower())


def count_tokens(text: str) -> int:
    """Deterministic token count used for retrieved-token budgets (spec §I.10)."""
    return len(_TOKEN_RE.findall(text))


def char_ngrams(text: str, sizes: tuple[int, ...]) -> list[str]:
    """Character n-grams over the lowercased, whitespace-collapsed text.

    Char n-grams let a keyless embedder capture morphological / subword overlap
    (``configure`` ~ ``configuration``) that exact-token BM25 misses — useful for
    constructing dense-vs-sparse divergence in the validation corpus.
    """
    s = re.sub(r"\s+", " ", text.lower()).strip()
    grams: list[str] = []
    for n in sizes:
        if n <= 0 or len(s) < n:
            continue
        for i in range(len(s) - n + 1):
            grams.append(s[i : i + n])
    return grams
