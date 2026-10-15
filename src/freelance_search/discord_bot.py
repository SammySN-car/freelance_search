import asyncio
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import discord
from discord.ext import commands, tasks

from .config import Settings
from .db import Database
from .pipeline import PreparedDigest, run_pipeline
from .scoring.ollama import OllamaScorer
from .sources import (
    ArbeitnowSource,
    JobicySource,
    RemoteOKSource,
    SerpApiSource,
    WeWorkRemotelySource,
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
    ) -> None:
        self.settings = settings
        self.database = database
        self.sources = sources
        self.scorer = scorer
        self.run_lock = asyncio.Lock()
        self.last_error: str | None = None

    async def run(self) -> PreparedDigest | None:
        if self.run_lock.locked():
            return None
        async with self.run_lock:
            try:
                result = await run_pipeline(
                    db=self.database,
                    sources=self.sources,
                    scorer=self.scorer,
                    profile=self.settings.profile,
                    min_score=self.settings.min_score,
                    serpapi_queries_used=len(self.settings.serpapi_queries),
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
                f"shown {run['shown_count']}\n"
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
        return f"{run_summary}\n{health_summary}\nScheduler: {scheduler_state}"


def build_sources(settings: Settings) -> list:
    return [
        SerpApiSource(settings.serpapi_api_key, settings.serpapi_queries),
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
    service = BotService(settings, database, build_sources(settings), scorer)
    intents = discord.Intents.none()
    bot = commands.Bot(command_prefix="!", intents=intents)

    def owner_only(interaction: discord.Interaction) -> bool:
        return is_owner(interaction.user.id, settings.discord_owner_id)

    @bot.tree.command(name="refresh", description="Run the freelance job search now")
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
            channel = bot.get_channel(settings.discord_channel_id)
            if channel is None:
                raise RuntimeError("Configured Discord channel was not found")
            await channel.send(result.message)
        except Exception as exc:
            service.delivery_failed(result, exc)
            await interaction.followup.send(
                f"{result.message}\n\nDelivery failed; listings were not marked shown.",
                ephemeral=True,
            )
            return
        service.deliver(result)
        await interaction.followup.send(result.message, ephemeral=True)

    @bot.tree.command(name="status", description="Show freelance search status")
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
            channel = bot.get_channel(settings.discord_channel_id)
            if channel is None:
                raise RuntimeError("Configured Discord channel was not found")
            await channel.send(result.message)
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
        if not scheduled_run.is_running():
            scheduled_run.start()
        await bot.tree.sync()

    bot.service = service
    bot.scheduled_run = scheduled_run
    return bot
