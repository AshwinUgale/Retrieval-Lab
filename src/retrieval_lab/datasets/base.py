"""Shared result type + writer for dataset importers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from retrieval_lab.gold import Query, write_documents_jsonl
from retrieval_lab.models import Document


@dataclass
class DatasetImport:
    """The outcome of importing a dataset: documents, verified queries, and skip counts."""

    documents: dict[str, Document]
    queries: list[Query]
    skipped: dict[str, int] = field(default_factory=dict)

    def write_jsonl(self, directory: str | Path) -> tuple[Path, Path]:
        """Write ``docs.jsonl`` + ``queries.jsonl`` (gold offsets already verified)."""
        from retrieval_lab.authoring import write_queries_jsonl

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        docs_path = write_documents_jsonl(self.documents.values(), directory / "docs.jsonl")
        queries_path = write_queries_jsonl(self.queries, directory / "queries.jsonl")
        return docs_path, queries_path

    def summary(self) -> str:
        skips = ", ".join(f"{k}={v}" for k, v in self.skipped.items()) or "none"
        return (f"{len(self.documents)} docs, {len(self.queries)} queries "
                f"(skipped: {skips})")
