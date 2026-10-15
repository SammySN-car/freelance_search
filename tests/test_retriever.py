import httpx
import pytest
import respx

from freelance_search.models import Listing
from freelance_search.retriever import EMBED_PATH, OllamaEmbedder, cosine_similarity


def job(identifier: str) -> Listing:
    return Listing(identifier, f"Job {identifier}", "Company", f"https://jobs/{identifier}", "test")


@pytest.mark.asyncio
@respx.mock
async def test_embedder_caches_profile_and_batches_listing_inputs():
    requests: list[list[str]] = []

    def reply(request: httpx.Request) -> httpx.Response:
        inputs = request.content
        import json

        values = json.loads(inputs)["input"]
        requests.append(values)
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0] for _ in values]})

    respx.post(f"http://ollama.test{EMBED_PATH}").mock(side_effect=reply)
    embedder = OllamaEmbedder("http://ollama.test", "embed", batch_size=2)

    result = await embedder.embed_listings([job("1"), job("2"), job("3")], "profile")

    assert len(result.embeddings) == 3
    assert requests == [["profile"], ["Job 1", "Job 2"], ["Job 3"]]


@pytest.mark.asyncio
@respx.mock
async def test_malformed_embedding_response_is_explicit_and_retryable():
    route = respx.post(f"http://ollama.test{EMBED_PATH}").mock(
        side_effect=[
            httpx.Response(200, json={"embeddings": [[1.0, 0.0]]}),
            httpx.Response(200, json={"embeddings": [[1.0, 0.0]]}),
            httpx.Response(200, json={"wrong": []}),
            httpx.Response(200, json={"embeddings": [[1.0, 0.0]]}),
        ]
    )
    embedder = OllamaEmbedder("http://ollama.test", "embed", batch_size=1)

    first = await embedder.embed_listings([job("1"), job("2")], "profile")
    second = await embedder.embed_listings([job("2")], "profile")

    assert first.embeddings == {"1": [1.0, 0.0]}
    assert "2" in first.errors
    assert second.embeddings == {"2": [1.0, 0.0]}
    assert route.call_count == 4


def test_cosine_similarity_distinguishes_related_vectors():
    profile = [1.0, 0.0]

    assert cosine_similarity(profile, [0.99, 0.01]) > cosine_similarity(profile, [0.0, 1.0])


@pytest.mark.asyncio
@respx.mock
async def test_embedding_http_failure_is_explicit():
    respx.post(f"http://ollama.test{EMBED_PATH}").mock(
        side_effect=[
            httpx.Response(200, json={"embeddings": [[1.0, 0.0]]}),
            httpx.ConnectError("down"),
        ]
    )
    embedder = OllamaEmbedder("http://ollama.test", "embed")

    result = await embedder.embed_listings([job("1")], "profile")

    assert result.embeddings == {}
    assert "1" in result.errors
    with pytest.raises(ValueError):
        cosine_similarity([0.0], [0.0])
