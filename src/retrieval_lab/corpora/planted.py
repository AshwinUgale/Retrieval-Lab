"""Planted-failure scenarios for the §I.13 attribution recovery suite.

Each builder constructs a corpus + query where a *specific* DAG stage is engineered to fail
for a structural reason (not ranking luck), so the recovery test can assert the tool
attributes it to exactly that stage. The stages that need engineered *intermediate* sets
(candidate/fusion internals) are additionally covered by hand-built ``StageOutputs`` unit
tests; here we plant the ones a real end-to-end run can guarantee.
"""

from __future__ import annotations

from retrieval_lab.chunking.base import Chunker
from retrieval_lab.gold import EvidenceSet, GoldAnswer, GoldSpan, Query
from retrieval_lab.hashing import content_hash
from retrieval_lab.models import Chunk, Document


def whole_doc_gold(doc: Document) -> GoldAnswer:
    """Gold that requires (nearly) the whole document — used for fragmentation planting."""
    span = GoldSpan(
        source_id=doc.id,
        start=0,
        end=len(doc.text),
        quoted_text=doc.text,
        source_version=content_hash(doc.text),
    )
    return GoldAnswer((EvidenceSet((span,)),))


def substring_gold(doc: Document, needle: str) -> GoldAnswer:
    start = doc.text.index(needle)
    span = GoldSpan(doc.id, start, start + len(needle), needle, content_hash(doc.text))
    return GoldAnswer((EvidenceSet((span,)),))


# --------------------------------------------------------------------------------------
# Representation failure — a chunker that drops a source region (simulated cleaning loss).
# --------------------------------------------------------------------------------------


class LossyChunker(Chunker):
    """Fixed-window chunker that omits a ``[drop_start, drop_end)`` region entirely.

    Stands in for an ingestion/cleaning step that silently removes content (boilerplate
    stripping gone wrong, a truncation cap). Any gold landing in the dropped region can never
    be covered by *any* chunk, so attribution reports a representation failure.
    """

    def __init__(self, drop_start: int, drop_end: int, chunk_size: int = 100) -> None:
        self.drop_start = drop_start
        self.drop_end = drop_end
        self.chunk_size = chunk_size

    @property
    def spec(self) -> str:
        return f"lossy:drop={self.drop_start}-{self.drop_end}:size={self.chunk_size}"

    def chunk(self, doc: Document) -> list[Chunk]:
        n = len(doc.text)
        spans: list[tuple[int, int]] = []
        for lo, hi in ((0, min(self.drop_start, n)), (min(self.drop_end, n), n)):
            s = lo
            while s < hi:
                spans.append((s, min(s + self.chunk_size, hi)))
                s += self.chunk_size
        return self._emit(doc, spans)


def build_representation_corpus() -> tuple[dict[str, Document], Query, Chunker]:
    """A single doc whose answer sits in a region the ``LossyChunker`` drops."""
    prefix = "Intro filler that is not the answer at all. "
    answer = "The activation code is ZQ-4471 and must be entered within ten minutes."
    suffix = " Trailing filler that is also not the answer whatsoever."
    text = prefix + answer + suffix
    doc = Document(id="DREP", text=text)
    a_start = text.index(answer)
    chunker = LossyChunker(drop_start=a_start, drop_end=a_start + len(answer), chunk_size=40)
    query = Query(id="QREP", text="what is the activation code", gold=substring_gold(doc, answer))
    return {"DREP": doc}, query, chunker


# --------------------------------------------------------------------------------------
# Fragmentation — the answer spans two chunks; no single chunk covers it to threshold.
# --------------------------------------------------------------------------------------


def build_fragmented_corpus() -> tuple[dict[str, Document], Query, int]:
    """Answer doc + unrelated distractors; returns the ``chunk_size`` that splits the answer.

    Used two ways: with ``top_k=1`` it plants a **final-cutoff** failure (the answer needs
    two chunks, one slot can't hold it); with a larger ``top_k`` it plants a **fragmentation
    signal** (a hit reconstructed only across chunks).
    """
    answer = (
        "The Zorblax reactor aligns its plasma coils, and the aligned plasma coils "
        "then sustain the Zorblax reaction for several hours of runtime."
    )
    docs = {
        "DFRAG": Document(id="DFRAG", text=answer),
        "DX1": Document(id="DX1", text="An unrelated passage about baking sourdough bread."),
        "DX2": Document(id="DX2", text="Another unrelated note on quarterly tax deadlines."),
    }
    query = Query(
        id="QFRAG",
        text="how do the plasma coils sustain the Zorblax reaction",
        gold=whole_doc_gold(docs["DFRAG"]),
    )
    split_size = len(answer) // 2  # two chunks, each ~0.5 coverage of the whole-doc span
    return docs, query, split_size


# --------------------------------------------------------------------------------------
# Candidate-generation failure — a distractor both retrievers rank above the answer.
# --------------------------------------------------------------------------------------


def build_candidate_miss_corpus() -> tuple[dict[str, Document], Query]:
    """A distractor contains the query verbatim; the real answer is a paraphrase elsewhere.

    Both dense and sparse rank the distractor first, so with ``candidate_n=1`` the raw
    candidate union never contains the gold — a candidate-generation failure with both
    branch diagnostics reading "missed".
    """
    docs = {
        "DDIST": Document(
            id="DDIST",
            text="What temperature to bake bread is a question many home cooks ask online.",
        ),
        "DANS": Document(
            id="DANS",
            text="Bread should be baked at 220 degrees Celsius until the crust turns golden.",
        ),
        "DX": Document(id="DX", text="A short unrelated note about bicycle maintenance."),
    }
    answer = "Bread should be baked at 220 degrees Celsius until the crust turns golden."
    query = Query(
        id="QCAND",
        text="what temperature to bake bread",
        gold=substring_gold(docs["DANS"], answer),
    )
    return docs, query


# --------------------------------------------------------------------------------------
# Reranker demotion — a lexical reranker pushes the (morphologically-matched) answer out.
# --------------------------------------------------------------------------------------


def build_reranker_demotion_corpus() -> tuple[dict[str, Document], Query]:
    """Dense retrieves the answer via char-n-gram overlap, but a lexical reranker demotes it.

    The gold doc shares *inflected* forms with the query (``resetting``/``passwords``) that
    the dense embedder's char n-grams match, but **no exact tokens** — so a shared-exact-term
    reranker scores it zero while a distractor that repeats the query verbatim scores high.
    With ``top_k=1`` the reranker pushes gold past the cutoff: a reranker demotion.
    """
    docs = {
        "DANS": Document(id="DANS", text="Resetting passwords is quick and simple."),
        "DDIST": Document(id="DDIST", text="reset the password reset the password reset."),
        "DX": Document(id="DX", text="An unrelated note about bicycle tire pressure."),
    }
    query = Query(
        id="QRR",
        text="reset the password",
        gold=whole_doc_gold(docs["DANS"]),
    )
    return docs, query


# --------------------------------------------------------------------------------------
# Budget cutoff — the answer is retrieved into top_k but too large for the token budget.
# --------------------------------------------------------------------------------------


def build_budget_cutoff_corpus() -> tuple[dict[str, Document], Query, int]:
    """Answer doc comfortably retrieved into top_k, but larger than a tiny token budget.

    Returns ``(docs, query, budget_tokens)``. Every document exceeds ``budget_tokens``, so the
    whole-chunk packing policy admits nothing that satisfies gold — a loss attributable to the
    budget policy alone, not to retrieval.
    """
    docs = {
        "DBIG": Document(
            id="DBIG",
            text=(
                "The quarterly revenue figure for the northern division was reported as "
                "exactly four point two million dollars in the audited statement."
            ),
        ),
        "DX": Document(id="DX", text="A short unrelated passage about garden composting."),
    }
    query = Query(
        id="QBUD",
        text="what was the quarterly revenue for the northern division",
        gold=whole_doc_gold(docs["DBIG"]),
    )
    return docs, query, 4  # 4-token budget; the answer chunk is far larger
