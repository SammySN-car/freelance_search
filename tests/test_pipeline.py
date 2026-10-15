from datetime import date

import pytest

from freelance_search.models import ScoreBatchResult, ScoreResult
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

    async def score_many(self, listings, profile):
        return ScoreBatchResult(
            scores={
                listing.listing_id: self.scores[listing.listing_id]
                for listing in listings
                if listing.listing_id in self.scores
            },
            errors=self.errors,
        )


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
        listing("b", "Same", "Beta"),
        listing("a", "Same", "Acme"),
    ]

    result = await run_pipeline(
        db=database,
        sources=[FakeSource("test", jobs)],
        scorer=FakeScorer({"a": ScoreResult(8, "A"), "b": ScoreResult(8, "B")}),
        profile="profile",
        min_score=0,
    )

    assert result.shown_listing_ids == ("a", "b")
