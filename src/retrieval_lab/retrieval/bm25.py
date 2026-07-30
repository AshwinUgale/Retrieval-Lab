"""Sparse retrieval: Okapi BM25 over exact terms (spec §I.7).

Implemented natively (numpy/stdlib) rather than pulling a dependency, because hybrid
retrieval is a *core* capability and BM25 is small, and a from-scratch implementation stays
fully deterministic and keyless. BM25 weights rare terms (via IDF) and saturates on
repetition (via ``k1``), normalizing for document length (via ``b``). It captures exact
terms a dense embedder can paraphrase past — and misses the paraphrases dense catches, which
is why fusing the two helps.

Score of a query against a document::

    score = Σ_t idf(t) · f(t,d)·(k1+1) / ( f(t,d) + k1·(1 - b + b·|d|/avgdl) )

with ``idf(t) = ln( 1 + (N - df(t) + 0.5) / (df(t) + 0.5) )`` (the non-negative Lucene form).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

from retrieval_lab.models import Chunk
from retrieval_lab.text import word_tokens


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._tf: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._idf: dict[str, float] = {}
        self._doc_len: list[int] = []
        self._avgdl: float = 0.0

    def index(self, chunks: Iterable[Chunk]) -> BM25Retriever:
        """Tokenize the corpus and precompute term statistics. Returns self for chaining."""
        self._chunks = list(chunks)
        self._tokens = [word_tokens(c.text) for c in self._chunks]
        self._tf = [Counter(toks) for toks in self._tokens]
        self._doc_len = [len(toks) for toks in self._tokens]
        n = len(self._chunks)
        self._avgdl = (sum(self._doc_len) / n) if n else 0.0

        self._df = Counter()
        for tf in self._tf:
            self._df.update(tf.keys())
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in self._df.items()
        }
        return self

    def _score_doc(self, i: int, q_terms: list[str]) -> float:
        tf = self._tf[i]
        dl = self._doc_len[i]
        denom_len = self.k1 * (1 - self.b + self.b * (dl / self._avgdl if self._avgdl else 0.0))
        score = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            score += self._idf.get(term, 0.0) * (f * (self.k1 + 1)) / (f + denom_len)
        return score

    def retrieve_scored(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Top-``k`` chunks by BM25 score. Chunks scoring 0 (no query term) are excluded.

        Ties and zero-signal queries break by corpus order, so results are deterministic.
        """
        if not self._chunks or k <= 0:
            return []
        # Score over DISTINCT query terms — accidental repetition in the query must not
        # double-weight a term (document-side term frequency is handled by the k1 saturation).
        q_terms = list(dict.fromkeys(word_tokens(query)))
        scored = [(i, self._score_doc(i, q_terms)) for i in range(len(self._chunks))]
        scored = [(i, s) for i, s in scored if s > 0.0]
        # Stable sort by descending score; ties keep ascending corpus index.
        scored.sort(key=lambda t: (-t[1], t[0]))
        return [(self._chunks[i], s) for i, s in scored[:k]]

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        return [c for c, _ in self.retrieve_scored(query, k)]
