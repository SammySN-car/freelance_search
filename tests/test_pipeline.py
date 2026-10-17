from datetime import date

import pytest

from freelance_search.models import EmbeddingBatchResult, ScoreBatchResult, ScoreResult
from freelance_search.pipeline import clean_url, format_digest, run_pipeline


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


def listing(identifier, title="RAG Engineer", company="Acme", length=20, url=None):
    from freelance_search.models import Listing

    return Listing(
        listing_id=identifier,
        title=title,
        company=company,
        url=url or f"https://jobs.test/{identifier}",
        source="test",
        description="x" * length,
    )


def all_descriptions(embeds):
    return "\n".join(embed.description for embed in embeds)


GOOGLE_URL = (
    "https://www.google.com/search?ibp=htl;jobs&q=RAG+engineer+freelance"
    "&hl=en-US&shem=epsd1,esd2e&shndl=37&shmd=abc&shmds=v1&source=sh/x/job"
    "&htidocid=q42CP6AGGoWV12uWAAAAAA%3D%3D#fpstate=tldetail&htiq=RAG"
)


def test_clean_url_strips_google_tracking_but_keeps_job_identity():
    cleaned = clean_url(GOOGLE_URL)

    assert "htidocid=q42CP6AGGoWV12uWAAAAAA%3D%3D" in cleaned
    assert "ibp=htl;jobs" in cleaned
    assert cleaned.startswith("https://www.google.com/search?")
    for junk in ("shem=", "shndl=", "shmd=", "shmds=", "source=", "#fpstate", "hl="):
        assert junk not in cleaned


def test_clean_url_strips_generic_tracking_and_keeps_everything_else():
    url = "https://jobs.test/7?utm_source=x&utm_medium=y&id=42&ref=board#apply"

    cleaned = clean_url(url)

    assert cleaned == "https://jobs.test/7?id=42#apply"


def test_clean_url_leaves_plain_urls_untouched():
    assert clean_url("https://jobs.test/plain") == "https://jobs.test/plain"


def test_digest_lists_numbered_masked_links_without_reasons():
    jobs = [
        (listing(str(index), f"Job {index}", "Acme"), ScoreResult(9 - index / 10, "secret reason"))
        for index in range(5)
    ]

    message, embeds, shown_ids = format_digest(
        jobs,
        digest_date=date(2026, 10, 16),
        footer="sources 2 · fetched 5 · new 5 · judged 5",
    )

    assert message == ""
    assert len(embeds) == 1
    assert shown_ids == tuple(str(index) for index in range(5))
    description = embeds[0].description
    assert description.count("\n") == 4
    for index in range(5):
        line = f"{index + 1}. [Job {index} · Acme](https://jobs.test/{index})"
        assert line in description
        assert "secret reason" not in description
        assert "9/10" not in description
    assert embeds[0].title.startswith("Opportunity shortlist - 2026-10-16")
    assert embeds[0].footer.text.endswith(" · scored by local gemma3:4b")


def test_digest_respects_digest_size_cap():
    jobs = [
        (listing(str(index), f"Job {index}"), ScoreResult(10 - index / 10, "r"))
        for index in range(30)
    ]

    _, embeds, shown_ids = format_digest(
        jobs,
        digest_date=date(2026, 10, 16),
        footer="f",
        digest_size=7,
    )

    assert len(shown_ids) == 7
    assert shown_ids == tuple(str(index) for index in range(7))
    assert len(embeds[0].description.splitlines()) == 7


def test_digest_uses_model_name_in_footer():
    jobs = [(listing("1"), ScoreResult(9, "r"))]

    _, embeds, _ = format_digest(
        jobs,
        digest_date=date(2026, 10, 16),
        footer="f",
        score_model="qwen3:4b",
    )

    assert embeds[0].footer.text.endswith(" · scored by local qwen3:4b")


def test_digest_splits_long_lists_across_embeds():
    jobs = [
        (listing(str(index), "A" * 70, "B" * 30), ScoreResult(9, "r"))
        for index in range(40)
    ]

    _, embeds, shown_ids = format_digest(
        jobs,
        digest_date=date(2026, 10, 16),
        footer="f",
        digest_size=40,
    )

    assert len(embeds) > 1
    assert all(len(embed.description) <= 3900 for embed in embeds)
    assert "(1/" in embeds[0].title and f"({len(embeds)}/{len(embeds)})" in embeds[-1].title
    assert len(shown_ids) == 40


def test_digest_drops_oversized_entries_without_truncating_urls():
    long_url = "https://jobs.test/deep/" + ("x" * 5700)
    jobs = [(listing("only", url=long_url), ScoreResult(9, "r"))]

    _, embeds, shown_ids = format_digest(
        jobs,
        digest_date=date(2026, 10, 16),
        footer="f",
    )

    assert embeds == []
    assert shown_ids == ()


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
    assert result.shown_listing_ids == ("7", "1", "2", "4", "5", "6")
    with database.connect() as connection:
        duplicate = connection.execute(
            "SELECT title FROM listings WHERE listing_id = '2'"
        ).fetchone()
    assert duplicate["title"] == "Duplicate"
    footer = result.embeds[-1].footer.text
    assert "source errors: 1" in footer
    assert "scoring errors: 1" in footer
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
async def test_long_urls_dropped_from_budget_never_truncated(database):
    jobs = []
    for index in range(1, 7):
        jobs.append(
            listing(
                str(index),
                f"Job {index}",
                url=f"https://jobs.test/{index}/" + ("x" * 1100),
            )
        )
    scores = {str(index): ScoreResult(10 - index / 10, "Reason") for index in range(1, 7)}

    result = await run_pipeline(
        db=database,
        sources=[FakeSource("test", jobs)],
        scorer=FakeScorer(scores),
        profile="profile",
        min_score=0,
        digest_date=date(2026, 10, 15),
    )

    assert 0 < len(result.shown_listing_ids) < 6
    description = all_descriptions(result.embeds)
    for listing_id in result.shown_listing_ids:
        url = f"https://jobs.test/{listing_id}/"
        line = next(line for line in description.splitlines() if f"]({url}" in line)
        assert line.endswith(f"]({url}{'x' * 1100})")


@pytest.mark.asyncio
async def test_budget_overflow_returns_text_message_not_empty_payload(database):
    job = listing("deep", url="https://jobs.test/deep/" + ("x" * 5700))

    result = await run_pipeline(
        db=database,
        sources=[FakeSource("test", [job])],
        scorer=FakeScorer({"deep": ScoreResult(9, "r")}),
        profile="profile",
        min_score=0,
    )

    assert result.empty is True
    assert result.embeds == []
    assert "No links fit the size budget" in result.message
    assert result.shown_listing_ids == ()


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

    assert "https://jobs.test/31" not in all_descriptions(first.embeds)
    assert len(embedder.calls[0]) == 31
    assert len(embedder.calls) == 1
    assert "31" not in scorer.calls[0]
    assert scorer.calls[1] == ["31"]
    assert second.stats["judged"] == 1


@pytest.mark.asyncio
async def test_digest_size_limit_applies_to_pipeline(database):
    jobs = [listing(str(index), f"Job {index}") for index in range(1, 11)]
    scores = {str(index): ScoreResult(10 - index / 10, "r") for index in range(1, 11)}

    result = await run_pipeline(
        db=database,
        sources=[FakeSource("test", jobs)],
        scorer=FakeScorer(scores),
        profile="profile",
        min_score=0,
        digest_size=3,
    )

    assert result.shown_listing_ids == ("1", "2", "3")


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
    assert "retry" in all_descriptions(result.embeds)

def test_dedup_collapses_same_title_and_company_across_urls():
    from freelance_search.pipeline import _identity_deduplicate

    first = listing("a", "Senior AI Engineer - Freelance", "Mirakl - Labs", url="https://x/1")
    second = listing(
        "b", "senior  ai engineer - freelance", "mirakl - labs", url="https://y/2"
    )

    kept = _identity_deduplicate([first, second])

    # later arrival wins, as with id/url duplicates
    assert [item.listing_id for item in kept] == ["b"]


@pytest.mark.asyncio
async def test_pipeline_hard_drops_stated_mismatched_locations(database):
    jobs = [
        listing("1", "Berlin Python Dev", "LocalCo"),
        listing("2", "Amsterdam Python Dev", "FarCo"),
        listing("3", "Remote Python Dev", "AnywhereCo"),
        listing("4", "Untagged Python Dev", "QuietCo"),
    ]
    from dataclasses import replace

    jobs[0] = replace(jobs[0], location="Berlin, Germany")
    jobs[1] = replace(jobs[1], location="Amsterdam")
    jobs[2] = replace(jobs[2], location="Remote")
    jobs[3] = replace(jobs[3], location=None)
    scorer = FakeScorer({job.listing_id: ScoreResult(9, "r") for job in jobs})

    result = await run_pipeline(
        db=database,
        sources=[FakeSource("fake", jobs)],
        scorer=scorer,
        profile="Python dev",
        min_score=1,
        location="Berlin",
    )

    judged = scorer.calls[0]
    assert "1" in judged
    assert "2" not in judged
    assert "3" in judged
    assert "4" in judged
    assert result.stats["location_filtered"] == 1


@pytest.mark.asyncio
async def test_pipeline_without_location_keeps_everything(database):
    jobs = [
        listing("1", "Job A", "Co", url="https://a/1"),
        listing("2", "Job B", "Co", url="https://b/2"),
    ]
    from dataclasses import replace

    jobs[0] = replace(jobs[0], location="Amsterdam")
    jobs[1] = replace(jobs[1], location="Paris")
    scorer = FakeScorer({job.listing_id: ScoreResult(9, "r") for job in jobs})

    result = await run_pipeline(
        db=database,
        sources=[FakeSource("fake", jobs)],
        scorer=scorer,
        profile="Python dev",
        min_score=1,
        location="",
    )

    assert len(scorer.calls[0]) == 2
    assert result.stats["location_filtered"] == 0


def test_seniority_filter_drops_senior_titles_only_for_junior_levels():
    from freelance_search.pipeline import filter_by_seniority

    jobs = [
        listing("1", "Python Developer"),
        listing("2", "Senior Python Developer"),
        listing("3", "Staff Software Engineer"),
        listing("4", "Lead Solutions Architect"),
        listing("5", "Principal Data Scientist"),
        listing("6", "Senior Director, Consulting"),
        listing("7", "Head of Growth"),
        listing("8", "Engineer - senior systems"),  # mid-word senior still matches \b
    ]

    intermediate = filter_by_seniority(jobs, "intermediate")
    beginner = filter_by_seniority(jobs, "beginner")
    senior = filter_by_seniority(jobs, "senior")
    unconfigured = filter_by_seniority(jobs, "")

    assert [item.listing_id for item in intermediate] == ["1"]
    assert [item.listing_id for item in beginner] == ["1"]
    assert len(senior) == len(jobs)
    assert len(unconfigured) == len(jobs)


@pytest.mark.asyncio
async def test_pipeline_drops_senior_titles_and_counts_them(database):
    from dataclasses import replace

    jobs = [
        listing("1", "Mid Python Dev", "OkCo"),
        listing("2", "Senior Python Dev", "ProCo"),
        listing("3", "Staff Engineer", "ProCo2"),
    ]
    jobs = [replace(job, company=f"Co{n}") for n, job in enumerate(jobs, 1)]
    scorer = FakeScorer({job.listing_id: ScoreResult(9, "r") for job in jobs})

    result = await run_pipeline(
        db=database,
        sources=[FakeSource("fake", jobs)],
        scorer=scorer,
        profile="Python dev",
        min_score=1,
        experience_level="intermediate",
    )

    assert scorer.calls[0] == ["1"]
    assert result.stats["seniority_filtered"] == 2


@pytest.mark.asyncio
async def test_pipeline_keeps_senior_titles_at_senior_level(database):
    jobs = [
        listing("1", "Mid Python Dev", "OkCo"),
        listing("2", "Senior Python Dev", "ProCo"),
    ]
    scorer = FakeScorer({job.listing_id: ScoreResult(9, "r") for job in jobs})

    result = await run_pipeline(
        db=database,
        sources=[FakeSource("fake", jobs)],
        scorer=scorer,
        profile="Python dev",
        min_score=1,
        experience_level="senior",
    )

    assert sorted(scorer.calls[0]) == ["1", "2"]
    assert result.stats["seniority_filtered"] == 0


@pytest.mark.asyncio
async def test_source_errors_include_exception_class(database):
    class EmptyMessage(RuntimeError):
        def __str__(self):
            return ""

    result = await run_pipeline(
        db=database,
        sources=[FakeSource("broken", error=EmptyMessage())],
        scorer=FakeScorer({}, {}),
        profile="p",
        min_score=1,
    )

    assert result.source_errors["broken"] == "EmptyMessage: no detail"
