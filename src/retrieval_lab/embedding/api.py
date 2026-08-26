"""Hosted embedding providers behind the ``[api-embed]`` extra.

The OpenAI client is imported lazily so the default install and test suite stay keyless.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from retrieval_lab.embedding.base import Embedder, EmbeddingCache, l2_normalize


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings adapter.

    The API key is resolved by the OpenAI SDK from the usual environment variables, so
    tests can inject a fake client and normal runs can use ``OPENAI_API_KEY``.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        cache: EmbeddingCache | None = None,
        *,
        client=None,
        dim: int = 1536,
        dimensions: int | None = None,
    ) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - only without the extra
                raise ImportError(
                    "OpenAIEmbedder needs the '[api-embed]' extra: "
                    "pip install 'retrieval-lab[api-embed]'"
                ) from exc
            client = OpenAI()

        name = f"api:{model_name}"
        if dimensions is not None:
            name = f"{name}:dim={dimensions}"
            dim = dimensions
        super().__init__(name=name, dim=dim, cache=cache)
        self.model_name = model_name
        self.dimensions = dimensions
        self._client = client

    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        kwargs = {"model": self.model_name, "input": texts}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        response = self._client.embeddings.create(**kwargs)
        vectors = [_embedding_vector(item) for item in response.data]
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed(texts)

    def embed_passage(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed(texts)


def _embedding_vector(item) -> Sequence[float]:
    if isinstance(item, dict):
        return item["embedding"]
    return item.embedding


def openai_embedder(
    model_name: str = "text-embedding-3-small",
    cache: EmbeddingCache | None = None,
    *,
    dim: int = 1536,
    dimensions: int | None = None,
) -> OpenAIEmbedder:
    return OpenAIEmbedder(model_name, cache=cache, dim=dim, dimensions=dimensions)
