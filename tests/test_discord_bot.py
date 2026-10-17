import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from freelance_search.config import Settings
from freelance_search.discord_bot import (
    BotService,
    create_bot,
    inspect_discord_channel,
    is_owner,
    next_digest_at,
    refresh_ack,
)
from freelance_search.models import Listing, OllamaHealth, ScoreBatchResult
from freelance_search.pipeline import PreparedDigest


def make_settings(**overrides):
    base = {
        "discord_token": "token",
        "discord_owner_id": 123,
        "discord_channel_id": 456,
        "serpapi_api_key": "key",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


class Source:
    name = "test"

    async def fetch(self, client):
        return []


class Scorer:
    def __init__(self):
        self.last_health = OllamaHealth(True, True)

    async def score_many(self, listings, profile):
        return ScoreBatchResult({}, {})

    async def health_check(self):
        return self.last_health


def prepared(run_id=1, ids=("job-1",), empty=False, embeds=None):
    return PreparedDigest(
        "Opportunity shortlist - 2026-10-15\n\nmessage",
        ids,
        empty=empty,
        stats={"sources": 1, "fetched": 1, "new": 1, "deduped": 0, "scored": 1},
        source_errors={},
        scoring_errors={},
        run_id=run_id,
        embeds=embeds or [],
    )


@pytest.mark.asyncio
async def test_successful_delivery_marks_only_after_send(database):
    service = BotService(make_settings(), database, [Source()], Scorer())
    database.initialize()
    result = prepared()
    channel = AsyncMock()
    database.save_listings(
        [
            Listing(
                "job-1",
                "Job",
                "Company",
                "https://jobs.test/1",
                "test",
            )
        ]
    )

    await channel.send(result.message or None, embeds=result.embeds or None)
    service.deliver(result)

    assert database.unshown_listing_ids(["job-1"]) == set()


def test_non_owner_is_rejected_by_owner_guard():
    assert is_owner(999, 123) is False
    assert is_owner(123, 123) is True


def test_startup_channel_check_finds_configured_channel():
    channel = type("Channel", (), {"id": 456, "name": "jobs"})()
    bot = type(
        "Bot",
        (),
        {"get_channel": lambda self, channel_id: channel, "guilds": []},
    )()

    assert inspect_discord_channel(bot, 456) is channel


def test_bot_enables_only_guild_intent(database):
    bot = create_bot(make_settings(), database)

    assert bot.intents.guilds is True
    assert bot.intents.message_content is False
    assert bot.intents.members is False


@pytest.mark.asyncio
async def test_refresh_success_sends_public_digest_and_short_ack(database):
    bot = create_bot(make_settings(), database)
    result = prepared(ids=(), empty=True)
    result.stats.update({"judged": 30, "new": 12})
    bot.service.run = AsyncMock(return_value=result)
    channel = AsyncMock()
    bot.get_channel = lambda channel_id: channel
    command = bot.tree.get_command("refresh")

    class Interaction:
        user = type("User", (), {"id": 123})()
        response = AsyncMock()
        followup = AsyncMock()

    await command.callback(Interaction())

    channel.send.assert_awaited_once_with(result.message or None, embeds=result.embeds or None)
    Interaction.followup.send.assert_awaited_once_with(
        "Digest posted · links 0 · judged 30 · new 12",
        ephemeral=True,
    )


@pytest.mark.asyncio
async def test_refresh_posts_multiple_embeds_then_marks_shown(database):
    import discord

    bot = create_bot(make_settings(), database)
    database.save_listings(
        [
            Listing("a", "Job A", "Acme", "https://jobs.test/a", "test"),
            Listing("b", "Job B", "Beta", "https://jobs.test/b", "test"),
        ]
    )
    embeds = [
        discord.Embed(
            title="Opportunity shortlist - 2026-10-16",
            description="1. [Job A \u00b7 Acme](https://jobs.test/a)",
        ),
        discord.Embed(
            title="Opportunity shortlist - 2026-10-16 (2/2)",
            description="2. [Job B \u00b7 Beta](https://jobs.test/b)",
        ),
    ]
    result = prepared(ids=("a", "b"), embeds=embeds)
    result.message = ""  # real link digests carry content in embeds only
    bot.service.run = AsyncMock(return_value=result)
    channel = AsyncMock()
    bot.get_channel = lambda channel_id: channel
    command = bot.tree.get_command("refresh")

    class Interaction:
        user = type("User", (), {"id": 123})()
        response = AsyncMock()
        followup = AsyncMock()

    await command.callback(Interaction())

    channel.send.assert_awaited_once_with(None, embeds=embeds)
    assert database.unshown_listing_ids(["a", "b"]) == set()


def test_startup_channel_check_reports_missing_channel():
    text_channel = type("Channel", (), {"id": 789, "name": "general"})()
    guild = type("Guild", (), {"id": 1, "name": "My Server", "channels": [text_channel]})()
    bot = type(
        "Bot",
        (),
        {"get_channel": lambda self, channel_id: None, "guilds": [guild]},
    )()

    assert inspect_discord_channel(bot, 456) is None


@pytest.mark.asyncio
async def test_refresh_command_rejects_non_owner(database):
    bot = create_bot(make_settings(), database)
    command = bot.tree.get_command("refresh")

    class Interaction:
        user = type("User", (), {"id": 999})()
        response = AsyncMock()

    await command.callback(Interaction())

    Interaction.response.send_message.assert_awaited_once_with("Owner only.", ephemeral=True)


def test_refresh_ack_contains_links_judged_and_new():
    result = prepared()
    result.stats.update({"judged": 30, "new": 12})

    assert refresh_ack(result) == "Digest posted · links 1 · judged 30 · new 12"


@pytest.mark.asyncio
async def test_failed_delivery_does_not_mark(database):
    service = BotService(make_settings(), database, [Source()], Scorer())
    database.initialize()
    database.save_listings(
        [Listing("job-1", "Job", "Company", "https://jobs.test/1", "test")]
    )
    result = prepared(run_id=database.start_run())

    service.delivery_failed(result, RuntimeError("Discord unavailable"))

    assert database.unshown_listing_ids(["job-1"]) == {"job-1"}
    assert database.latest_run()["status"] == "failed"
    assert "Discord delivery failed" in database.latest_run()["error"]


@pytest.mark.asyncio
async def test_overlap_lock_returns_busy_without_second_pipeline(database):
    service = BotService(make_settings(), database, [Source()], Scorer())
    database.initialize()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked_run(*args, **kwargs):
        entered.set()
        await release.wait()
        return prepared()

    import freelance_search.discord_bot as discord_bot

    original = discord_bot.run_pipeline
    discord_bot.run_pipeline = blocked_run
    try:
        first = asyncio.create_task(service.run())
        await entered.wait()
        assert await service.run() is None
        release.set()
        assert await first
    finally:
        discord_bot.run_pipeline = original


def test_next_digest_time_uses_configured_timezone():
    now = datetime.fromisoformat("2026-10-15T21:46:00+05:30")

    result = next_digest_at(now, "08:00", "Asia/Kolkata")

    assert result.isoformat() == "2026-10-16T08:00:00+05:30"


@pytest.mark.asyncio
async def test_status_contains_run_counts_query_count_and_ollama_health(database):
    service = BotService(make_settings(), database, [Source()], Scorer())
    database.initialize()
    run_id = database.start_run()
    database.finish_run(
        run_id,
        status="completed",
        fetched_count=4,
        new_count=2,
        deduped_count=2,
        scored_count=2,
        source_stats={"test": {"fetched": 4, "failed": 0}},
        scoring_errors={},
        serpapi_queries_used=3,
    )
    await service.scorer.health_check()

    message = service.status_message("running")

    assert "fetched 4 · new 2 · deduped 2 · scored 2 · shown 0" in message
    assert "SerpApi queries: 3" in message
    assert "reachable=True" in message
    assert "present=True" in message
    assert "Discord channel: NOT FOUND" in message
    assert "Scheduler: running" in message

    service.set_discord_channel(type("Channel", (), {"name": "jobs"})())
    assert "Discord channel: OK (#jobs)" in service.status_message("running")


def test_status_shows_customization_line(database):
    service = BotService(
        make_settings(location="India", experience_level="beginner", digest_size=30),
        database,
        [Source()],
        Scorer(),
    )
    database.initialize()

    message = service.status_message("stopped")

    expected = "Customization: level beginner · location India · 30 links/run · top 50 judged"
    assert expected in message


def test_status_marks_unset_customization(database):
    service = BotService(make_settings(), database, [Source()], Scorer())
    database.initialize()

    message = service.status_message("stopped")

    assert "Customization: level unset · location unset · 25 links/run · top 50 judged" in message


@pytest.mark.asyncio
async def test_bot_service_passes_composed_profile_and_digest_size(database):
    service = BotService(
        make_settings(
            profile="Python dev.",
            location="India",
            experience_level="beginner",
            digest_size=12,
        ),
        database,
        [Source()],
        Scorer(),
    )
    database.initialize()
    captured = {}

    async def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return prepared()

    import freelance_search.discord_bot as discord_bot

    original = discord_bot.run_pipeline
    discord_bot.run_pipeline = fake_pipeline
    try:
        await service.run()
    finally:
        discord_bot.run_pipeline = original

    assert captured["profile"].startswith("Python dev.")
    assert "Candidate location: India" in captured["profile"]
    assert "Candidate experience level: beginner" in captured["profile"]
    assert captured["digest_size"] == 12