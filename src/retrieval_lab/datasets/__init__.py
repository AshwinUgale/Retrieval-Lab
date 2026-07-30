"""Importers that convert real labeled retrieval datasets into Retrieval Lab's data model.

These let you benchmark on real data instead of constructed corpora. Each importer verifies
gold as it builds it and **skips** (rather than crashes on) records that don't verify, so a
produced ``queries.jsonl`` always loads cleanly under the fail-closed loader.
"""

from retrieval_lab.datasets.base import DatasetImport
from retrieval_lab.datasets.beir import load_beir
from retrieval_lab.datasets.squad import load_squad

__all__ = ["DatasetImport", "load_squad", "load_beir"]
