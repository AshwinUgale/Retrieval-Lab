"""Chunkers — turn documents into retrievable units that remember their source span."""

from retrieval_lab.chunking.base import Chunker
from retrieval_lab.chunking.fixed import FixedSizeChunker
from retrieval_lab.chunking.recursive import RecursiveChunker

__all__ = ["Chunker", "FixedSizeChunker", "RecursiveChunker"]
