import json

import httpx
import pytest
import respx

from freelance_search.models import Listing
from freelance_search.scoring.ollama import GENERATE_PATH, TAGS_PATH, OllamaScorer

LISTING = Listing(
    listing_id="test:1",
    title="RAG Engineer",
    company="Acme",
    url="https://jobs.test/1",
    source="test",
    description="Build retrieval augmented generation systems.",
)
PROFILE = "AI/ML engineer focused on RAG and local LLMs."


def scorer() -> OllamaScorer:
    return OllamaScorer("http://ollama.test", "gemma3:4b", timeout_seconds=1)


@pytest.mark.asyncio
@respx.mock
async def test_valid_ollama_response_is_strictly_parsed():
    respx.post(f"http://ollama.test{GENERATE_PATH}").mock(
        return_value=httpx.Response(
            200, json={"response": '{"score": 8.5, "reason": "Strong RAG match."}'}
        )
    )

    result = await scorer().score_many([LISTING], PROFILE)

    assert result.scores["test:1"].score == 8.5
    assert result.errors == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_response",
    [
        "not json",
        '{"score": 11, "reason": "Too high"}',
        '{"score": "9/10", "reason": "String score"}',
    ],
)
@respx.mock
async def test_invalid_ollama_payload_is_an_error(raw_response):
    respx.post(f"http://ollama.test{GENERATE_PATH}").mock(
        return_value=httpx.Response(200, json={"response": raw_response})
    )

    result = await scorer().score_many([LISTING], PROFILE)

    assert result.scores == {}
    assert "test:1" in result.errors


@pytest.mark.asyncio
@respx.mock
async def test_reason_is_flattened_and_capped():
    reason = "This is a multiline\nreason with " + ("extra " * 40)
    respx.post(f"http://ollama.test{GENERATE_PATH}").mock(
        return_value=httpx.Response(
            200, json={"response": json.dumps({"score": 7, "reason": reason})}
        )
    )

    result = await scorer().score_many([LISTING], PROFILE)

    normalized_reason = result.scores["test:1"].reason
    assert "\n" not in normalized_reason
    assert len(normalized_reason) == 120


@pytest.mark.asyncio
@respx.mock
async def test_one_listing_failure_does_not_abort_batch():
    respx.post(f"http://ollama.test{GENERATE_PATH}").mock(
        side_effect=[
            httpx.Response(200, json={"response": "bad"}),
            httpx.Response(200, json={"response": '{"score": 6, "reason": "Good match"}'}),
        ]
    )
    second = Listing("test:2", "Python Engineer", "Beta", "https://jobs.test/2", "test")

    result = await scorer().score_many([LISTING, second], PROFILE)

    assert set(result.scores) == {"test:2"}
    assert set(result.errors) == {"test:1"}


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [httpx.ConnectError("down"), httpx.ReadTimeout("slow")])
@respx.mock
async def test_ollama_transport_failures_are_errors(exception):
    respx.post(f"http://ollama.test{GENERATE_PATH}").mock(side_effect=exception)

    result = await scorer().score_many([LISTING], PROFILE)

    assert result.scores == {}
    assert "test:1" in result.errors


@pytest.mark.asyncio
@respx.mock
async def test_health_check_reports_reachability_and_model():
    respx.get(f"http://ollama.test{TAGS_PATH}").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "gemma3:4b"}]})
    )

    health = await scorer().health_check()

    assert health.reachable is True
    assert health.model_present is True


@pytest.mark.asyncio
@respx.mock
async def test_health_check_requires_exact_model_tag():
    respx.get(f"http://ollama.test{TAGS_PATH}").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "gemma3:12b"}]})
    )

    health = await scorer().health_check()

    assert health.reachable is True
    assert health.model_present is False


@pytest.mark.asyncio
@pytest.mark.live
async def test_live_ollama_scoring():
    live_scorer = OllamaScorer("http://127.0.0.1:11434", "gemma3:4b", timeout_seconds=180)

    result = await live_scorer.score_many([LISTING], PROFILE)

    assert result.errors == {}
    assert 0 <= result.scores["test:1"].score <= 10
