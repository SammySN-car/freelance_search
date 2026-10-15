from datetime import date

import pytest

from freelance_search.models import EmbeddingBatchResult, ScoreBatchResult, ScoreResult
from freelance_search.pipeline import run_pipeline


class FakeSource:
    def __init__(self, name, listings=None, error=None):
        self.name = name
        self.listings = listings or []
        self.error = error

    async def fetch(self, client):
        if self.error:
            raise self.error
        return self.listings


class FakeScorer:
    def __init__(self, scores, errors=None):
        self.scores = scores
        self.errors = errors or {}
        self.calls = []

    async def score_many(self, listings, profile):
        self.calls.append([listing.listing_id for listing in listings])
        return ScoreBatchResult(
            scores={
                listing.listing_id: self.scores[listing.listing_id]
                for listing in listings
                if listing.listing_id in self.scores
            },
            errors=self.errors,
        )


class FakeEmbedder:
    def __init__(self, similarities):
        self.similarity_values = similarities
        self.calls = []

    async def embed_listings(self, listings, profile):
        self.calls.append([listing.listing_id for listing in listings])
        return EmbeddingBatchResult(
            {
                listing.listing_id: [self.similarity_values[listing.listing_id], 1.0]
                for listing in listings
            },
            {},
        )

    def similarities(self, embeddings):
        return {
            listing_id: self.similarity_values[listing_id]
            for listing_id in embeddings
        }


def listing(identifier, title="RAG Engineer", company="Acme", length=20):
    from freelance_search.models import Listing

    return Listing(
        listing_id=identifier,
        title=title,
        company=company,
        url=f"https://jobs.test/{identifier}",
        source="test",
        description="x" * length,
    )


@pytest.mark.asyncio
async def test_pipeline_deduplicates_filters_sorts_caps_and_records_run(database):
    jobs = [
        listing("1", "Beta", "Zeta"),
        listing("2", "Alpha", "Zeta"),
        listing("2", "Duplicate", "Zeta"),
        listing("3", "Low", "Acme"),
        listing("4", "Gamma", "Acme"),
        listing("5", "Delta", "Acme"),
        listing("6", "Epsilon", "Acme"),
        listing("7", "Zeta", "Acme"),
    ]
    scores = {
        "1": ScoreResult(8, "Good"),
        "2": ScoreResult(8, "Good"),
        "4": ScoreResult(7, "Good"),
        "5": ScoreResult(6, "Good"),
        "6": ScoreResult(5, "Good"),
        "7": ScoreResult(9, "Best"),
    }

    result = await run_pipeline(
        db=database,
        sources=[FakeSource("primary", jobs), FakeSource("broken", error=RuntimeError("down"))],
        scorer=FakeScorer(scores, {"3": "model error"}),
        profile="profile",
        min_score=5,
        digest_date=date(2026, 10, 15),
    )

    assert result.empty is False
    assert result.shown_listing_ids == ("7", "1", "2", "4", "5")
    with database.connect() as connection:
        duplicate = connection.execute(
            "SELECT title FROM listings WHERE listing_id = '2'"
        ).fetchone()
    assert duplicate["title"] == "Duplicate"
    assert "source errors: 1" in result.message
    assert "scoring errors: 1" in result.message
    with database.connect() as connection:
        run = connection.execute("SELECT * FROM runs").fetchone()
    assert run["status"] == "completed"
    assert run["fetched_count"] == 8
    assert run["new_count"] == 7
    assert run["deduped_count"] == 1
    assert run["scored_count"] == 6
    assert run["scoring_errors"] == '{"3": "model error"}'
    assert run["source_stats"] == (
        '{"broken": {"failed": 1, "fetched": 0}, '
        '"primary": {"failed": 0, "fetched": 8}}'
    )


@pytest.mark.asyncio
async def test_already_shown_jobs_are_not_scored_or_returned(database):
    job = listing("shown")
    database.save_listings([job])
    database.mark_shown(["shown"])

    result = await run_pipeline(
        db=database,
        sources=[FakeSource("test", [job])],
        scorer=FakeScorer({}),
        profile="profile",
        min_score=0,
        digest_date=date(2026, 10, 15),
    )

    assert result.empty is True
    assert result.shown_listing_ids == ()
    assert "No new matching jobs." in result.message


@pytest.mark.asyncio
async def test_long_entries_drop_lowest_without_truncating_urls(database):
    jobs = [listing(str(index), f"Job {index} " + ("x" * 600)) for index in range(1, 6)]
    scores = {str(index): ScoreResult(10 - index / 10, "Reason") for index in range(1, 6)}

    result = await run_pipeline(
        db=database,
        sources=[FakeSource("test", jobs)],
        scorer=FakeScorer(scores),
        profile="profile",
        min_score=0,
        digest_date=date(2026, 10, 15),
    )

    assert len(result.message) <= 2000
    assert result.shown_listing_ids == ("1", "2")
    assert "https://jobs.test/1" in result.message
    assert "https://jobs.test/3" not in result.message


@pytest.mark.asyncio
async def test_failed_delivery_does_not_mark_listings_shown(database):
    job = listing("delivery")
    result = await run_pipeline(
        db=database,
        sources=[FakeSource("test", [job])],
        scorer=FakeScorer({"delivery": ScoreResult(9, "Good")}),
        profile="profile",
        min_score=0,
    )

    assert result.shown_listing_ids == ("delivery",)
    assert database.unshown_listing_ids(["delivery"]) == {"delivery"}


@pytest.mark.asyncio
async def test_deterministic_tie_break_uses_title_then_company(database):
    jobs = [
        listing("a", "Zeta", "Beta"),
        listing("z", "Alpha", "Acme"),
    ]

    result = await run_pipeline(
        db=database,
        sources=[FakeSource("test", jobs)],
        scorer=FakeScorer({"a": ScoreResult(8, "A"), "z": ScoreResult(8, "B")}),
        profile="profile",
        min_score=0,
    )

    assert result.shown_listing_ids == ("z", "a")


@pytest.mark.asyncio
async def test_retrieval_ranks_judging_and_judge_once(database):
    jobs = [listing(str(index), f"Job {index}") for index in range(1, 32)]
    similarity = {str(index): 1.0 - index / 100 for index in range(1, 32)}
    scorer = FakeScorer(
        {str(index): ScoreResult(8, f"reason {index}") for index in range(1, 32)}
    )
    embedder = FakeEmbedder(similarity)

    first = await run_pipeline(
        db=database,
        sources=[FakeSource("test", jobs)],
        scorer=scorer,
        embedder=embedder,
        max_to_score=30,
        profile="profile",
        min_score=0,
    )
    second = await run_pipeline(
        db=database,
        sources=[FakeSource("test", jobs)],
        scorer=scorer,
        embedder=embedder,
        max_to_score=30,
        profile="profile",
        min_score=0,
    )

    assert "https://jobs.test/31" not in first.message
    assert len(embedder.calls[0]) == 31
    assert len(embedder.calls) == 1
    assert "31" not in scorer.calls[0]
    assert scorer.calls[1] == ["31"]
    assert second.stats["judged"] == 1


@pytest.mark.asyncio
async def test_embedding_failure_keeps_listing_for_next_run(database):
    jobs = [listing("good"), listing("retry")]
    embedder = FakeEmbedder({"good": 0.9, "retry": 0.8})
    original = embedder.embed_listings
    calls = 0

    async def fail_once(listings, profile):
        nonlocal calls
        calls += 1
        if calls == 1:
            return EmbeddingBatchResult({"good": [0.9, 1.0]}, {"retry": "batch failed"})
        return await original(listings, profile)

    embedder.embed_listings = fail_once
    scorer = FakeScorer({"good": ScoreResult(9, "good"), "retry": ScoreResult(8, "retry")})

    await run_pipeline(
        db=database,
        sources=[FakeSource("test", jobs)],
        scorer=scorer,
        embedder=embedder,
        profile="profile",
        min_score=0,
    )
    result = await run_pipeline(
        db=database,
        sources=[FakeSource("test", jobs)],
        scorer=scorer,
        embedder=embedder,
        profile="profile",
        min_score=0,
    )

    assert result.stats["embedded"] == 1
    assert "retry" in result.message
