from collections.abc import Iterable, Sequence
from datetime import date, datetime

import httpx

from .db import Database
from .models import Listing, ScoreBatchResult, ScoreResult

DISCORD_MESSAGE_LIMIT = 2000
MAX_DIGEST_ENTRIES = 5


class PipelineSource:
    name: str

    async def fetch(self, client: httpx.AsyncClient) -> list[Listing]:
        raise NotImplementedError


class PipelineScorer:
    async def score_many(self, listings: list[Listing], profile: str) -> ScoreBatchResult:
        raise NotImplementedError


class PreparedDigest:
    def __init__(
        self,
        message: str,
        shown_listing_ids: tuple[str, ...],
        *,
        empty: bool,
        stats: dict[str, int],
        source_errors: dict[str, str],
        scoring_errors: dict[str, str],
        run_id: int,
    ) -> None:
        self.message = message
        self.shown_listing_ids = shown_listing_ids
        self.empty = empty
        self.stats = stats
        self.source_errors = source_errors
        self.scoring_errors = scoring_errors
        self.run_id = run_id


def _sort_key(item: tuple[Listing, ScoreResult]) -> tuple[float, str, str, str]:
    listing, score = item
    return (-score.score, listing.title.casefold(), listing.company.casefold(), listing.listing_id)


def _entry(number: int, listing: Listing, score: ScoreResult) -> str:
    pay = f" · {listing.pay}" if listing.pay else ""
    return (
        f"{number}. {listing.title} · {listing.company}{pay} · {score.score:g}/10 · "
        f"{score.reason} · {listing.url}"
    )


def format_digest(
    ranked: Sequence[tuple[Listing, ScoreResult]],
    *,
    digest_date: date,
    footer: str,
) -> tuple[str, tuple[str, ...]]:
    header = f"Freelance shortlist - {digest_date.isoformat()}"
    selected = list(ranked[:MAX_DIGEST_ENTRIES])
    while selected:
        entries = "\n".join(
            _entry(index, listing, score) for index, (listing, score) in enumerate(selected, 1)
        )
        message = f"{header}\n\n{entries}\n\n{footer}"
        if len(message) <= DISCORD_MESSAGE_LIMIT:
            return message, tuple(listing.listing_id for listing, _ in selected)
        selected.pop()
    return f"{header}\n\n{footer}", ()


async def run_pipeline(
    *,
    db: Database,
    sources: Iterable[PipelineSource],
    scorer: PipelineScorer,
    profile: str,
    min_score: float,
    digest_date: date | None = None,
    client: httpx.AsyncClient | None = None,
    serpapi_queries_used: int = 0,
) -> PreparedDigest:
    run_id = db.start_run()
    source_stats: dict[str, dict[str, int]] = {}
    source_errors: dict[str, str] = {}
    fetched: list[Listing] = []
    own_client = client is None
    request_client = client or httpx.AsyncClient()
    try:
        for source in sources:
            try:
                listings = await source.fetch(request_client)
                source_stats[source.name] = {"fetched": len(listings), "failed": 0}
                fetched.extend(listings)
            except (httpx.HTTPError, TimeoutError, OSError, ValueError, RuntimeError) as exc:
                source_stats[source.name] = {"fetched": 0, "failed": 1}
                source_errors[source.name] = str(exc)

        unique_listings = list({listing.listing_id: listing for listing in fetched}.values())
        db.save_listings(unique_listings)
        unshown_ids = db.unshown_listing_ids(listing.listing_id for listing in unique_listings)
        new_listings = [listing for listing in unique_listings if listing.listing_id in unshown_ids]
        deduped_count = len(fetched) - len(new_listings)
        batch = await scorer.score_many(new_listings, profile)
        ranked = sorted(
            (
                (listing, batch.scores[listing.listing_id])
                for listing in new_listings
                if listing.listing_id in batch.scores
                and batch.scores[listing.listing_id].score >= min_score
            ),
            key=_sort_key,
        )
        date_value = digest_date or datetime.now().date()
        footer = (
            f"Sources: {len(source_stats)} · fetched: {len(fetched)} · new: {len(new_listings)} · "
            f"scored: {len(batch.scores)}"
        )
        if source_errors:
            footer += f" · source errors: {len(source_errors)}"
        if batch.errors:
            footer += f" · scoring errors: {len(batch.errors)}"
        if ranked:
            message, shown_ids = format_digest(ranked, digest_date=date_value, footer=footer)
            empty = False
        else:
            message = (
                f"Freelance shortlist - {date_value.isoformat()}\n\n"
                f"No new matching jobs.\n\n{footer}"
            )
            shown_ids = ()
            empty = True
        db.finish_run(
            run_id,
            status="completed",
            fetched_count=len(fetched),
            new_count=len(new_listings),
            deduped_count=deduped_count,
            scored_count=len(batch.scores),
            source_stats=source_stats,
            scoring_errors=batch.errors,
            serpapi_queries_used=serpapi_queries_used,
            error="; ".join(f"{name}: {error}" for name, error in source_errors.items()) or None,
        )
        return PreparedDigest(
            message,
            shown_ids,
            empty=empty,
            stats={
                "sources": len(source_stats),
                "fetched": len(fetched),
                "new": len(new_listings),
                "deduped": deduped_count,
                "scored": len(batch.scores),
            },
            source_errors=source_errors,
            scoring_errors=batch.errors,
            run_id=run_id,
        )
    finally:
        if own_client:
            await request_client.aclose()
