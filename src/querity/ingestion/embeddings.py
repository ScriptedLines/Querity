from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from tenacity import (
    Retrying,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from querity.common.cache import LocalCache, build_local_cache, hash_key
from querity.common.logging import get_logger
from querity.ingestion.settings import IngestionSettings

logger = get_logger(__name__)


def embedding_model_id(settings: IngestionSettings) -> str:
    return (
        f"{settings.embedding_provider}:{settings.embedding_model}:{settings.embedding_dimensions}"
    )


class EmbeddingValidationError(ValueError):
    pass


def _build_embedding_client(settings: IngestionSettings) -> Any:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    kwargs: dict[str, Any] = {
        "model": settings.embedding_model,
    }
    if settings.nvidia_api_key is not None:
        kwargs["api_key"] = settings.nvidia_api_key.get_secret_value().strip()
    return NVIDIAEmbeddings(**kwargs)


class EmbeddingClient:
    def __init__(
        self,
        settings: IngestionSettings,
        *,
        cache: LocalCache | None = None,
    ) -> None:
        self._settings = settings
        self._client = _build_embedding_client(settings)
        self._cache = cache if cache is not None else build_local_cache(settings)
        self._retryer = Retrying(
            stop=stop_after_attempt(4),
            wait=wait_exponential_jitter(initial=1, max=30),
            retry=retry_if_not_exception_type((ValueError, TypeError)),
            reraise=True,
        )

    @property
    def model_id(self) -> str:
        return embedding_model_id(self._settings)

    def _validate_vectors(self, texts: Sequence[str], vectors: Any) -> list[list[float]]:
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingValidationError(
                "Embedding provider returned an unexpected number of vectors"
            )
        validated: list[list[float]] = []
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != self._settings.embedding_dimensions:
                actual = len(vector) if isinstance(vector, list) else "unknown"
                raise EmbeddingValidationError(
                    "Embedding dimension mismatch: expected "
                    f"{self._settings.embedding_dimensions}, got {actual}. "
                    "Check EMBEDDING_MODEL/EMBEDDING_DIMENSIONS and whether the target "
                    "pgvector column uses the same dimension."
                )
            floats = [float(item) for item in vector]
            if not all(math.isfinite(item) for item in floats):
                raise EmbeddingValidationError(
                    "Embedding provider returned a non-finite vector component"
                )
            validated.append(floats)
        return validated

    def _embed_once(
        self,
        texts: Sequence[str],
        client: Any | None = None,
        *,
        query: bool = False,
    ) -> list[list[float]]:
        embedder = client or self._client
        if query:
            if len(texts) != 1:
                raise ValueError("Query embedding accepts exactly one text")
            vectors = [embedder.embed_query(texts[0])]
        else:
            vectors = embedder.embed_documents(list(texts))
        return self._validate_vectors(texts, vectors)

    def _embed_items(
        self,
        texts: Sequence[str],
        client: Any | None = None,
        *,
        query: bool = False,
    ) -> tuple[list[list[float] | None], list[str]]:
        try:
            vectors = self._retryer(self._embed_once, texts, client, query=query)
            return list(vectors), [""] * len(texts)
        except Exception as exc:
            if isinstance(exc, EmbeddingValidationError):
                return [None] * len(texts), [str(exc)] * len(texts)
            if len(texts) == 1:
                logger.warning("Embedding failed for one document: %s", exc)
                return [None], [str(exc)]
            midpoint = len(texts) // 2
            left_vectors, left_errors = self._embed_items(texts[:midpoint], client, query=query)
            right_vectors, right_errors = self._embed_items(texts[midpoint:], client, query=query)
            return left_vectors + right_vectors, left_errors + right_errors

    def _cache_key(self, text: str, *, query: bool) -> str:
        task = "query" if query else "document"
        return hash_key("embedding", self.model_id, task, text)

    def _cached_vector(self, text: str, *, query: bool) -> list[float] | None:
        if self._cache is None:
            return None
        vector = self._cache.get_vector(self._cache_key(text, query=query))
        if vector is None or len(vector) != self._settings.embedding_dimensions:
            return None
        return vector

    def _store_vector(self, text: str, vector: list[float], *, query: bool) -> None:
        if self._cache is None:
            return
        self._cache.set_vector(self._cache_key(text, query=query), vector)

    def _cache_lookup_documents(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float] | None], dict[str, list[int]]]:
        vectors: list[list[float] | None] = [None] * len(texts)
        missing: dict[str, list[int]] = {}
        for index, text in enumerate(texts):
            key = self._cache_key(text, query=False)
            vector = self._cache.get_vector(key) if self._cache is not None else None
            if vector is not None and len(vector) != self._settings.embedding_dimensions:
                vector = None
            if vector is None:
                missing.setdefault(key, []).append(index)
            else:
                vectors[index] = vector
        return vectors, missing

    def embed_query(self, text: str) -> list[float]:
        cached = self._cached_vector(text, query=True)
        if cached is not None:
            return cached
        vectors, errors = self._embed_items([text], query=True)
        vector = vectors[0]
        if vector is None:
            raise RuntimeError(f"Query embedding failed: {errors[0] or 'unknown error'}")
        self._store_vector(text, vector, query=True)
        return vector

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        results, missing = self._cache_lookup_documents(texts)
        keys = list(missing)
        pending_texts = [texts[missing[key][0]] for key in keys]
        batch_size = self._settings.embedding_batch_size
        for start in range(0, len(pending_texts), batch_size):
            batch = pending_texts[start : start + batch_size]
            batch_keys = keys[start : start + batch_size]
            vectors, errors = self._embed_items(batch)
            if any(vector is None for vector in vectors):
                first_error = next(error for error in errors if error)
                raise RuntimeError(f"Embedding batch failed: {first_error}")
            for key, vector in zip(batch_keys, vectors, strict=True):
                assert vector is not None
                for index in missing[key]:
                    results[index] = vector
                if self._cache is not None:
                    self._cache.set_vector(key, vector)
        return [vector for vector in results if vector is not None]

    def embed_documents_tolerant(
        self, texts: Sequence[str]
    ) -> tuple[list[list[float] | None], list[str]]:
        if not texts:
            return [], []
        results, missing = self._cache_lookup_documents(texts)
        errors: list[str] = [""] * len(texts)
        keys = list(missing)
        pending_texts = [texts[missing[key][0]] for key in keys]
        batch_size = self._settings.embedding_batch_size
        for start in range(0, len(pending_texts), batch_size):
            batch = pending_texts[start : start + batch_size]
            batch_keys = keys[start : start + batch_size]
            vectors, batch_errors = self._embed_items(batch)
            for key, vector, error in zip(batch_keys, vectors, batch_errors, strict=True):
                for index in missing[key]:
                    results[index] = vector
                    errors[index] = error
                if vector is not None and self._cache is not None:
                    self._cache.set_vector(key, vector)
        return results, errors
