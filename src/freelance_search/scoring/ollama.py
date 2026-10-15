import json
import re
from typing import Any

import httpx

from ..models import Listing, OllamaHealth, ScoreBatchResult, ScoreResult

GENERATE_PATH = "/api/generate"
TAGS_PATH = "/api/tags"
MAX_REASON_LENGTH = 120


class ScoringError(Exception):
    """Raised when Ollama cannot produce a valid score for a listing."""


def build_prompt(listing: Listing, profile: str) -> str:
    return (
        "Score this freelance listing against the profile below. Return exactly one JSON object "
        'with this shape: {"score": <number 0-10>, "reason": "<one line>"}. '
        "The score must be a JSON number, not a string. Do not include markdown or extra keys.\n\n"
        f"PROFILE:\n{profile}\n\n"
        "LISTING:\n"
        f"Title: {listing.title}\n"
        f"Company: {listing.company}\n"
        f"Pay: {listing.pay or 'Not listed'}\n"
        f"Location: {listing.location or 'Not listed'}\n"
        f"Description: {listing.description}\n"
        f"URL: {listing.url}"
    )


def _validate_score(payload: Any, listing_id: str) -> ScoreResult:
    if not isinstance(payload, dict):
        raise ScoringError(f"{listing_id}: response must be a JSON object")
    score = payload.get("score")
    reason = payload.get("reason")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ScoringError(f"{listing_id}: score must be a JSON number")
    if not 0 <= score <= 10:
        raise ScoringError(f"{listing_id}: score must be between 0 and 10")
    if not isinstance(reason, str) or not reason.strip():
        raise ScoringError(f"{listing_id}: reason must be a non-empty string")
    flattened = re.sub(r"\s+", " ", reason).strip()
    return ScoreResult(score=float(score), reason=flattened[:MAX_REASON_LENGTH])


class OllamaScorer:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = httpx.Timeout(timeout_seconds)

    async def score_listing(
        self,
        client: httpx.AsyncClient,
        listing: Listing,
        profile: str,
    ) -> ScoreResult:
        try:
            response = await client.post(
                f"{self.base_url}{GENERATE_PATH}",
                json={
                    "model": self.model,
                    "prompt": build_prompt(listing, profile),
                    "format": "json",
                    "stream": False,
                },
            )
            response.raise_for_status()
            response_data = response.json()
            raw_result = response_data.get("response")
            if not isinstance(raw_result, str):
                raise ScoringError(f"{listing.listing_id}: missing Ollama response text")
            try:
                parsed_result = json.loads(raw_result)
            except json.JSONDecodeError as exc:
                raise ScoringError(f"{listing.listing_id}: response was not valid JSON") from exc
            return _validate_score(parsed_result, listing.listing_id)
        except ScoringError:
            raise
        except httpx.TimeoutException as exc:
            raise ScoringError(f"{listing.listing_id}: Ollama request timed out") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ScoringError(f"{listing.listing_id}: Ollama request failed: {exc}") from exc

    async def score_many(self, listings: list[Listing], profile: str) -> ScoreBatchResult:
        scores: dict[str, ScoreResult] = {}
        errors: dict[str, str] = {}
        timeout = self.timeout
        async with httpx.AsyncClient(timeout=timeout) as client:
            for listing in listings:
                try:
                    scores[listing.listing_id] = await self.score_listing(client, listing, profile)
                except ScoringError as exc:
                    errors[listing.listing_id] = str(exc)
        return ScoreBatchResult(scores=scores, errors=errors)

    async def health_check(self) -> OllamaHealth:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}{TAGS_PATH}")
                response.raise_for_status()
                models = response.json().get("models", [])
            names = {model.get("name") for model in models if isinstance(model, dict)}
            model_present = self.model in names
            return OllamaHealth(reachable=True, model_present=model_present)
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            return OllamaHealth(reachable=False, model_present=False, error=str(exc))
