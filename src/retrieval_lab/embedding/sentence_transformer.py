"""Real embedding models via sentence-transformers (spec §I.7, §I.14).

Behind the ``[real-embed]`` extra so the core never depends on it. e5 and bge are
**asymmetric** models — they expect a role prefix on queries vs passages — so the roles are
applied here and folded into the cache key (the prefixed text is what gets embedded), which
keeps the content-addressed cache correct across roles.

Factories ``e5_embedder`` / ``bge_embedder`` set the right prefixes; the deterministic
embedder remains the default and the whole test path stays keyless.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from retrieval_lab.embedding.base import Embedder, EmbeddingCache, l2_normalize


class SentenceTransformerEmbedder(Embedder):
    """Wraps a sentence-transformers model, L2-normalizing output and applying role prefixes."""

    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        passage_prefix: str = "",
        cache: EmbeddingCache | None = None,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "SentenceTransformerEmbedder needs the '[real-embed]' extra: "
                "pip install 'retrieval-lab[real-embed]'"
            ) from exc
        self._model = SentenceTransformer(model_name)
        dim = int(self._model.get_sentence_embedding_dimension())
        # Name includes the prefixes so two prefix configs of one model don't share a cache.
        name = f"{model_name}|q={query_prefix}|p={passage_prefix}"
        super().__init__(name=name, dim=dim, cache=cache)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size

    def _embed_raw(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - needs a model
        emb = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return l2_normalize(np.asarray(emb, dtype=np.float32))

    def embed_query(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed([self.query_prefix + t for t in texts])

    def embed_passage(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed([self.passage_prefix + t for t in texts])


def e5_embedder(
    model_name: str = "intfloat/e5-small-v2",
    cache: EmbeddingCache | None = None,
) -> SentenceTransformerEmbedder:
    """An e5 embedder with its required ``query:`` / ``passage:`` prefixes."""
    return SentenceTransformerEmbedder(
        model_name, query_prefix="query: ", passage_prefix="passage: ", cache=cache
    )


def bge_embedder(
    model_name: str = "BAAI/bge-small-en-v1.5",
    cache: EmbeddingCache | None = None,
) -> SentenceTransformerEmbedder:
    """A bge embedder with the retrieval query instruction (passages unprefixed)."""
    return SentenceTransformerEmbedder(
        model_name,
        query_prefix="Represent this sentence for searching relevant passages: ",
        passage_prefix="",
        cache=cache,
    )
