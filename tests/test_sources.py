import httpx
import pytest
import respx

from freelance_search.sources.arbeitnow import ARBEITNOW_URL, ArbeitnowSource
from freelance_search.sources.jobicy import JOBICY_URL, JobicySource
from freelance_search.sources.remoteok import REMOTEOK_URL, RemoteOKSource
from freelance_search.sources.serpapi import SERPAPI_URL, SerpApiSource
from freelance_search.sources.weworkremotely import WWR_RSS_URL, WeWorkRemotelySource


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["", "   "])
@respx.mock
async def test_serpapi_without_location_sends_no_location_param(location):
    respx.get(SERPAPI_URL).mock(return_value=httpx.Response(200, json={"jobs_results": []}))

    async with httpx.AsyncClient() as client:
        await SerpApiSource("key", ["one", "two"], location).fetch(client)

    for call in respx.calls:
        assert "location" not in httpx.URL(str(call.request.url)).params


@pytest.mark.asyncio
@respx.mock
async def test_serpapi_normalizes_jobs():
    respx.get(SERPAPI_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "jobs_results": [
                        {
                            "job_id": "abc",
                            "title": "RAG Engineer",
                            "company_name": "Acme",
                            "share_link": "https://jobs.test/abc",
                            "description": "Build retrieval systems.",
                            "detected_extensions": {"salary": "$50/hr"},
                            "location": "Remote",
                        }
                    ]
                },
            ),
            httpx.Response(200, json={"jobs_results": []}),
            httpx.Response(200, json={"jobs_results": []}),
        ]
    )

    async with httpx.AsyncClient() as client:
        listings = await SerpApiSource("key", ["one", "two", "three"]).fetch(client)

    assert listings[0].listing_id == "serpapi:abc"
    assert listings[0].title == "RAG Engineer"
    assert listings[0].pay == "$50/hr"


@pytest.mark.asyncio
@respx.mock
async def test_serpapi_passes_location_to_every_query():
    respx.get(SERPAPI_URL).mock(return_value=httpx.Response(200, json={"jobs_results": []}))

    async with httpx.AsyncClient() as client:
        await SerpApiSource("key", ["one", "two"], "Berlin, Germany").fetch(client)

    assert len(respx.calls) == 2
    for call in respx.calls:
        assert httpx.URL(str(call.request.url)).params.get("location") == "Berlin, Germany"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "url", "payload", "expected_id"),
    [
        (
            RemoteOKSource(),
            REMOTEOK_URL,
            [{"id": 7, "position": "ML Engineer", "company": "Acme", "url": "https://job.test/7"}],
            "remoteok:7",
        ),
        (
            JobicySource(),
            JOBICY_URL,
            {
                "jobs": [
                    {
                        "id": 8,
                        "jobTitle": "Python Engineer",
                        "companyName": "Beta",
                        "url": "https://job.test/8",
                    }
                ]
            },
            "jobicy:8",
        ),
        (
            ArbeitnowSource(),
            ARBEITNOW_URL,
            {
                "data": [
                    {
                        "slug": "python-engineer",
                        "title": "Python Engineer",
                        "company_name": "Gamma",
                        "url": "https://job.test/9",
                    }
                ]
            },
            "arbeitnow:python-engineer",
        ),
    ],
)
@pytest.mark.asyncio
@respx.mock
async def test_json_sources_normalize_listings(source, url, payload, expected_id):
    respx.get(url).mock(return_value=httpx.Response(200, json=payload))

    async with httpx.AsyncClient() as client:
        listings = await source.fetch(client)

    assert listings[0].listing_id == expected_id
    assert listings[0].url.startswith("https://job.test/")


@pytest.mark.asyncio
@respx.mock
async def test_we_work_remotely_normalizes_rss():
    respx.get(WWR_RSS_URL).mock(
        return_value=httpx.Response(
            200,
            content=(
                b'<?xml version="1.0"?><rss><channel><item>'
                b"<title>ML Engineer at Delta</title>"
                b"<link>https://job.test/rss</link><author>Delta</author>"
                b"<description>Build ML systems.</description>"
                b"</item></channel></rss>"
            ),
        )
    )

    async with httpx.AsyncClient() as client:
        listings = await WeWorkRemotelySource().fetch(client)

    assert listings[0].listing_id == "weworkremotely:https://job.test/rss"
    assert listings[0].company == "Delta"

@pytest.mark.asyncio
@respx.mock
async def test_serpapi_prefers_stable_source_link_over_share_link():
    respx.get(SERPAPI_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "jobs_results": [
                        {
                            "job_id": "abc",
                            "title": "RAG Engineer",
                            "company_name": "Acme",
                            "source_link": "https://origin.test/posting/1",
                            "share_link": "https://www.google.com/search?ibp=htl;jobs&htidocid=zzz",
                        }
                    ]
                },
            ),
            httpx.Response(200, json={"jobs_results": []}),
            httpx.Response(200, json={"jobs_results": []}),
        ]
    )

    async with httpx.AsyncClient() as client:
        listings = await SerpApiSource("key", ["one", "two", "three"]).fetch(client)

    assert listings[0].url == "https://origin.test/posting/1"


@pytest.mark.asyncio
@respx.mock
async def test_serpapi_falls_back_to_share_link_without_source_link():
    respx.get(SERPAPI_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "jobs_results": [
                        {
                            "job_id": "abc",
                            "title": "RAG Engineer",
                            "company_name": "Acme",
                            "share_link": "https://www.google.com/search?ibp=htl;jobs&htidocid=zzz",
                        }
                    ]
                },
            ),
            httpx.Response(200, json={"jobs_results": []}),
            httpx.Response(200, json={"jobs_results": []}),
        ]
    )

    async with httpx.AsyncClient() as client:
        listings = await SerpApiSource("key", ["one", "two", "three"]).fetch(client)

    assert "htidocid=zzz" in listings[0].url
