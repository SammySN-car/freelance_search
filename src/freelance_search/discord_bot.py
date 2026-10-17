import asyncio
import json
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import discord
import httpx
from discord.ext import commands, tasks

from .config import Settings, compose_profile
from .db import Database
from .github_analysis import GithubAnalyzer, GithubError
from .models import GithubVerdict
from .pipeline import PreparedDigest, run_pipeline
from .retriever import OllamaEmbedder
from .scoring.ollama import OllamaScorer
from .sources import (
    ArbeitnowSource,
    JobicySource,
    RemoteOKSource,
    SerpApiSource,
    WeWorkRemotelySource,
)

LOGGER = logging.getLogger(__name__)
CHANNEL_HINT = (
    "DISCORD_CHANNEL_ID must be a channel ID copied via Developer Mode from a "
    "channel this bot can see — a server ID will not work."
)


def inspect_discord_channel(bot: commands.Bot, channel_id: int) -> object | None:
    channel = bot.get_channel(channel_id)
    LOGGER.info("Configured Discord channel ID: %s", channel_id)
    if channel is not None:
        LOGGER.info(
            "Configured Discord channel found: #%s (%s)",
            getattr(channel, "name", "unknown"),
            channel_id,
        )
        return channel

    LOGGER.error("Configured Discord channel ID %s was not found.", channel_id)
    for guild in getattr(bot, "guilds", []):
        LOGGER.error("Guild: %s (%s)", getattr(guild, "name", "unknown"), guild.id)
        for candidate in getattr(guild, "channels", []):
            LOGGER.error(
                "  Text channel: #%s (%s)",
                getattr(candidate, "name", "unknown"),
                candidate.id,
            )
    LOGGER.error(CHANNEL_HINT)
    return None


def refresh_ack(result: PreparedDigest) -> str:
    return (
        f"Digest posted · links {len(result.shown_listing_ids)} · "
        f"judged {result.stats.get('judged', 0)} · "
        f"new {result.stats.get('new', 0)}"
    )


def next_digest_at(now: datetime, digest_time: str, timezone: str) -> datetime:
    local_now = now.astimezone(ZoneInfo(timezone))
    hour, minute = (int(part) for part in digest_time.split(":"))
    candidate = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= local_now:
        candidate += timedelta(days=1)
    return candidate


def is_owner(user_id: int, owner_id: int) -> bool:
    return user_id == owner_id


class BotService:
    def __init__(
        self,
        settings: Settings,
        database: Database,
        sources: list,
        scorer: OllamaScorer,
        embedder: OllamaEmbedder | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.sources = sources
        self.scorer = scorer
        self.embedder = embedder
        self.run_lock = asyncio.Lock()
        self.last_error: str | None = None
        self.discord_channel_name: str | None = None
        self.github_verdict: GithubVerdict | None = None
        self.github_source: str = "unset"

    def set_discord_channel(self, channel: object | None) -> None:
        self.discord_channel_name = getattr(channel, "name", None) if channel else None

    def _github_is_fresh(self, verdict: GithubVerdict) -> bool:
        try:
            analyzed = datetime.fromisoformat(verdict.analyzed_at)
        except ValueError:
            return False
        age_days = (datetime.now() - analyzed).total_seconds() / 86400
        return age_days < self.settings.github_refresh_days

    async def resolve_github_verdict(self) -> GithubVerdict | None:
        """GitHub verdict with cache; refresh at most once per github_refresh_days."""
        url = self.settings.github_url.strip()
        if not url:
            self.github_source = "unset"
            self.github_verdict = None
            return None
        cached = self.database.get_github_verdict()
        if cached is not None and self._github_is_fresh(cached):
            self.github_source = "cache"
            self.github_verdict = cached
            return cached
        analyzer = GithubAnalyzer(
            url,
            self.settings.ollama_base_url,
            self.settings.ollama_model,
            self.settings.ollama_timeout_seconds,
        )
        try:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
                verdict = await analyzer.analyze(client)
        except (GithubError, httpx.HTTPError, TimeoutError, OSError) as exc:
            LOGGER.warning("GitHub analysis failed, falling back: %s", exc)
            if cached is not None:
                self.github_source = "cache (refresh failed)"
                self.github_verdict = cached
                return cached
            self.github_source = "failed"
            self.github_verdict = None
            return None
        self.database.save_github_verdict(verdict)
        self.github_source = "fresh"
        self.github_verdict = verdict
        return verdict

    async def run(self) -> PreparedDigest | None:
        if self.run_lock.locked():
            return None
        async with self.run_lock:
            try:
                github = await self.resolve_github_verdict()
                active_level = (
                    github.level if github is not None else self.settings.experience_level
                )
                result = await run_pipeline(
                    db=self.database,
                    sources=self.sources,
                    scorer=self.scorer,
                    profile=compose_profile(self.settings, github),
                    experience_level=active_level,
                    min_score=self.settings.min_score,
                    serpapi_queries_used=len(self.settings.serpapi_queries),
                    embedder=self.embedder,
                    location=self.settings.location,
                    max_to_score=self.settings.max_to_score,
                    digest_size=self.settings.digest_size,
                    score_model=self.settings.ollama_model,
                )
                self.last_error = None
                return result
            except Exception as exc:
                self.last_error = str(exc)
                run = self.database.latest_run()
                if run is not None and run["status"] == "running":
                    self.database.record_run_error(run["run_id"], str(exc))
                raise

    def deliver(self, result: PreparedDigest) -> None:
        self.database.mark_shown(result.shown_listing_ids)
        self.database.record_delivery(result.run_id, len(result.shown_listing_ids))

    def delivery_failed(self, result: PreparedDigest, error: Exception) -> None:
        self.last_error = f"Discord delivery failed: {error}"
        self.database.record_run_error(result.run_id, self.last_error)

    def status_message(self, scheduler_state: str) -> str:
        run = self.database.latest_run()
        health = self.scorer.last_health
        if run is None:
            run_summary = "Last run: never"
        else:
            source_stats = json.loads(run["source_stats"])
            source_failures = sum(stats["failed"] for stats in source_stats.values())
            run_summary = (
                f"Last run: {run['completed_at'] or run['started_at']} · {run['status']}\n"
                f"Counts: fetched {run['fetched_count']} · new {run['new_count']} · "
                f"deduped {run['deduped_count']} · scored {run['scored_count']} · "
                f"shown {run['shown_count']} · embedded {run['embedded_count']} · "
                f"judged {run['judged_count']}\n"
                f"Source failures: {source_failures} · SerpApi queries: "
                f"{run['serpapi_queries_used']}"
            )
        if health is None:
            health_summary = "Ollama: not checked"
        else:
            health_summary = (
                f"Ollama: reachable={health.reachable} · "
                f"model {self.settings.ollama_model} present={health.model_present}"
            )
        if self.discord_channel_name:
            channel_summary = f"Discord channel: OK (#{self.discord_channel_name})"
        else:
            channel_summary = (
                "Discord channel: NOT FOUND — fix DISCORD_CHANNEL_ID "
                "(see startup log)"
            )
        if self.github_source == "unset":
            level = self.settings.experience_level or "unset"
            github_line = f"GitHub: not configured · level {level} (manual)"
        elif self.github_verdict is not None:
            github_line = (
                f"GitHub: {self.github_verdict.level} from "
                f"{self.github_verdict.repos_total} repos ({self.github_source}) · "
                f"{self.github_verdict.summary}"
            )
        else:
            github_line = "GitHub: analysis failed · using manual level"
        customization = (
            f"Customization: level {self.settings.experience_level or 'unset'} · "
            f"location {self.settings.location or 'unset'} · "
            f"{self.settings.digest_size} links/run · top "
            f"{self.settings.max_to_score} judged"
        )
        return (
            f"{run_summary}\n{health_summary}\n{channel_summary}\n"
            f"{github_line}\n{customization}\nScheduler: {scheduler_state}"
        )


def build_sources(settings: Settings) -> list:
    return [
        SerpApiSource(settings.serpapi_api_key, settings.serpapi_queries, settings.location),
        RemoteOKSource(),
        JobicySource(),
        ArbeitnowSource(),
        WeWorkRemotelySource(),
    ]


def create_bot(settings: Settings, database: Database | None = None) -> commands.Bot:
    database = database or Database(settings.database_path)
    database.initialize()
    scorer = OllamaScorer(
        settings.ollama_base_url,
        settings.ollama_model,
        settings.ollama_timeout_seconds,
    )
    embedder = OllamaEmbedder(
        settings.ollama_base_url,
        settings.embedding_model,
        settings.ollama_timeout_seconds,
    )
    service = BotService(settings, database, build_sources(settings), scorer, embedder)
    intents = discord.Intents.none()
    intents.guilds = True
    bot = commands.Bot(command_prefix="!", intents=intents)

    def owner_only(interaction: discord.Interaction) -> bool:
        return is_owner(interaction.user.id, settings.discord_owner_id)

    async def post_digest(channel: object, result: PreparedDigest) -> None:
        await channel.send(result.message or None, embeds=result.embeds or None)

    @bot.tree.command(name="refresh", description="Run the opportunity search now")
    async def refresh(interaction: discord.Interaction) -> None:
        if not owner_only(interaction):
            await interaction.response.send_message("Owner only.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        try:
            result = await service.run()
        except Exception as exc:
            await interaction.followup.send(
                f"Refresh failed: {exc}",
                ephemeral=True,
            )
            return
        if result is None:
            await interaction.followup.send("Run already in progress.", ephemeral=True)
            return
        try:
            channel = inspect_discord_channel(bot, settings.discord_channel_id)
            if channel is None:
                raise RuntimeError("Configured Discord channel was not found")
            await post_digest(channel, result)
        except Exception as exc:
            service.delivery_failed(result, exc)
            await interaction.followup.send(
                f"Delivery failed: {exc}. Listings were not marked shown.",
                ephemeral=True,
            )
            return
        service.deliver(result)
        await interaction.followup.send(refresh_ack(result), ephemeral=True)

    @bot.tree.command(name="status", description="Show opportunity search status")
    async def status(interaction: discord.Interaction) -> None:
        if not owner_only(interaction):
            await interaction.response.send_message("Owner only.", ephemeral=True)
            return
        await service.scorer.health_check()
        await interaction.response.send_message(
            service.status_message("running" if scheduled_run.is_running() else "stopped"),
            ephemeral=True,
        )

    @tasks.loop(hours=24)
    async def scheduled_run() -> None:
        try:
            result = await service.run()
            if result is None or result.empty:
                return
            channel = inspect_discord_channel(bot, settings.discord_channel_id)
            if channel is None:
                raise RuntimeError("Configured Discord channel was not found")
            await post_digest(channel, result)
            service.deliver(result)
        except Exception as exc:
            service.last_error = f"Scheduled run failed: {exc}"
            run = service.database.latest_run()
            if run is not None and run["status"] == "completed":
                service.database.record_run_error(run["run_id"], service.last_error)

    @scheduled_run.before_loop
    async def wait_for_digest_time() -> None:
        await bot.wait_until_ready()
        now = datetime.now(ZoneInfo(settings.digest_timezone))
        target = next_digest_at(now, settings.digest_time, settings.digest_timezone)
        await asyncio.sleep((target - now).total_seconds())

    @bot.event
    async def on_ready() -> None:
        service.set_discord_channel(
            inspect_discord_channel(bot, settings.discord_channel_id)
        )
        if not scheduled_run.is_running():
            scheduled_run.start()
        await bot.tree.sync()

    bot.service = service
    bot.scheduled_run = scheduled_run
    return bot