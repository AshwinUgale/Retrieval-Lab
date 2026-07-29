"""Parent-child chunker: index small children, return their large parents (spec §I.7).

Small child chunks give retrieval *precision* (a tight unit matches the query sharply); the
larger parent gives the reader *context* (and covers more of a gold span). So children are
what gets indexed and ranked, but when a child is retrieved its **parent** is what is
returned — and the token budget counts the parent text (spec §I.10).

``chunk`` returns the child chunks (the indexed units); ``expand`` maps a ranked list of
children to their de-duplicated parents, in first-seen order. The pipeline applies ``expand``
to every stage set when this chunker is used, so coverage, attribution, and budgeting all
operate on the returned unit (the parent) consistently.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from retrieval_lab.chunking.base import Chunker
from retrieval_lab.models import Chunk, Document


class ParentChildChunker(Chunker):
    def __init__(self, parent_size: int = 1200, child_size: int = 300) -> None:
        if child_size <= 0 or parent_size <= 0:
            raise ValueError("sizes must be positive")
        if child_size > parent_size:
            raise ValueError("child_size must be <= parent_size")
        self.parent_size = parent_size
        self.child_size = child_size
        self._parents: dict[str, Chunk] = {}  # child id -> parent chunk

    @property
    def spec(self) -> str:
        return f"parentchild:parent={self.parent_size}:child={self.child_size}"

    def chunk(self, doc: Document) -> list[Chunk]:
        n = len(doc.text)
        if n == 0:
            return []
        children: list[Chunk] = []
        for p_start in range(0, n, self.parent_size):
            p_end = min(p_start + self.parent_size, n)
            parent = Chunk.make(
                doc.id, p_start, p_end, doc.text[p_start:p_end],
                chunker_spec=self.spec + ":parent",
            )
            for c_start in range(p_start, p_end, self.child_size):
                c_end = min(c_start + self.child_size, p_end)
                child = Chunk.make(
                    doc.id, c_start, c_end, doc.text[c_start:c_end],
                    chunker_spec=self.spec, parent_id=parent.id,
                )
                self._parents[child.id] = parent
                children.append(child)
        return children

    def chunk_corpus(self, docs: Iterable[Document]) -> list[Chunk]:
        self._parents = {}  # rebuild the child->parent map for this corpus
        return super().chunk_corpus(docs)

    def expand(self, children: Sequence[Chunk]) -> list[Chunk]:
        """Map ranked children to their de-duplicated parents, preserving first-seen order.

        A child with no known parent (e.g. constructed directly) maps to itself, so this is
        safe to apply to any chunk list.
        """
        seen: set[str] = set()
        out: list[Chunk] = []
        for child in children:
            target = self._parents.get(child.id, child)
            if target.id not in seen:
                seen.add(target.id)
                out.append(target)
        return out
