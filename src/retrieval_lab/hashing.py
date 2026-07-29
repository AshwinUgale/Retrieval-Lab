"""Stable, process-independent hashing.

Python's built-in ``hash()`` is salted per-process for strings, so it must never be used
for chunk identity or the content-addressed embedding cache — those keys have to be stable
across runs and machines. Everything that needs a durable identity goes through here.
"""

from __future__ import annotations

import hashlib


def stable_hash(*parts: object) -> str:
    """Return a stable hex digest of the given parts.

    Parts are joined with a NUL separator (which cannot appear inside the string forms we
    use) so that ``stable_hash("a", "bc")`` and ``stable_hash("ab", "c")`` differ.
    """
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def content_hash(text: str) -> str:
    """Content hash of a document's text — used as a ``source_version`` fingerprint."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
