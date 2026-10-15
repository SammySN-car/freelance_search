import math
from collections.abc import Sequence
from typing import Any

import httpx

from .models import EmbeddingBatchResult, Listing

EMBED_PATH = "/api/embed"


class EmbeddingError(Exception):
    """Raised when Ollama cannot produce valid embeddings."""


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        raise ValueError("embedding vectors must have the same non-zero length")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("embedding vectors must not be zero vectors")
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def listing_text(listing: Listing) -> str:
    return f"{listing.title}\n{listing.description}".strip()


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 180.0,
        batch_size: int = 16,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = httpx.Timeout(timeout_seconds)
        self.batch_size = batch_size
        self._profile_vector: list[float] | None = None

    async def _embed(self, client: httpx.AsyncClient, inputs: list[str]) -> list[list[float]]:
        try:
            response = await client.post(
                f"{self.base_url}{EMBED_PATH}",
                json={"model": self.model, "input": inputs},
            )
            response.raise_for_status()
            payload: Any = response.json()
            embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
            if (
                not isinstance(embeddings, list)
                or len(embeddings) != len(inputs)
                or any(
                    not isinstance(vector, list)
                    or not vector
                    or any(
                        isinstance(value, bool) or not isinstance(value, (int, float))
                        for value in vector
                    )
                    for vector in embeddings
                )
            ):
                raise EmbeddingError("Ollama returned an invalid embeddings payload")
            return [[float(value) for value in vector] for vector in embeddings]
        except EmbeddingError:
            raise
        except httpx.TimeoutException as exc:
            raise EmbeddingError("Ollama embedding request timed out") from exc
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise EmbeddingError(f"Ollama embedding request failed: {exc}") from exc

    async def embed_listings(
        self,
        listings: list[Listing],
        profile: str,
    ) -> EmbeddingBatchResult:
        embeddings: dict[str, list[float]] = {}
        errors: dict[str, str] = {}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if self._profile_vector is None:
                try:
                    self._profile_vector = (await self._embed(client, [profile]))[0]
                except EmbeddingError as exc:
                    message = f"profile: {exc}"
                    return EmbeddingBatchResult(
                        {},
                        {listing.listing_id: message for listing in listings},
                    )
            for start in range(0, len(listings), self.batch_size):
                batch = listings[start : start + self.batch_size]
                try:
                    vectors = await self._embed(
                        client, [listing_text(listing) for listing in batch]
                    )
                except EmbeddingError as exc:
                    for listing in batch:
                        errors[listing.listing_id] = str(exc)
                    continue
                for listing, vector in zip(batch, vectors, strict=True):
                    embeddings[listing.listing_id] = vector
        return EmbeddingBatchResult(embeddings, errors)

    def similarities(self, embeddings: dict[str, list[float]]) -> dict[str, float]:
        if self._profile_vector is None:
            raise EmbeddingError("profile has not been embedded")
        return {
            listing_id: cosine_similarity(self._profile_vector, vector)
            for listing_id, vector in embeddings.items()
        }
