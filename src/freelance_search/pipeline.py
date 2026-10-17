from collections.abc import Iterable, Sequence
from datetime import date, datetime
from urllib.parse import urlsplit, urlunsplit

import discord
import httpx

from .db import Database
from .models import EmbeddingBatchResult, Listing, ScoreBatchResult, ScoreResult

EMBED_DESCRIPTION_LIMIT = 3900
EMBED_TOTAL_BUDGET = 5600
TITLE_LENGTH_LIMIT = 80

GOOGLE_JOBS_PARAMS = {"ibp", "q", "htidocid"}
TRACKING_PARAMS = {
    "gclid",
    "fbclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_",
    "se",
    "source",
    "spm",
}


class PipelineSource:
    name: str

    async def fetch(self, client: httpx.AsyncClient) -> list[Listing]:
        raise NotImplementedError


class PipelineScorer:
    async def score_many(self, listings: list[Listing], profile: str) -> ScoreBatchResult:
        raise NotImplementedError


class PipelineEmbedder:
    async def embed_listings(
        self, listings: list[Listing], profile: str
    ) -> EmbeddingBatchResult:
        raise NotImplementedError

    def similarities(self, embeddings: dict[str, list[float]]) -> dict[str, float]:
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
        embedding_errors: dict[str, str] | None = None,
        embeds: Sequence[discord.Embed] = (),
    ) -> None:
        self.message = message
        self.shown_listing_ids = shown_listing_ids
        self.empty = empty
        self.stats = stats
        self.source_errors = source_errors
        self.scoring_errors = scoring_errors
        self.embedding_errors = embedding_errors or {}
        self.embeds = list(embeds)
        self.run_id = run_id


def _sort_key(item: tuple[Listing, ScoreResult]) -> tuple[float, str, str, str]:
    listing, score = item
    return (-score.score, listing.title.casefold(), listing.company.casefold(), listing.listing_id)


def _identity_deduplicate(listings: Iterable[Listing]) -> list[Listing]:
    selected: dict[str, Listing] = {}
    id_by_url: dict[str, str] = {}
    for listing in listings:
        previous_by_id = selected.pop(listing.listing_id, None)
        if previous_by_id is not None:
            id_by_url.pop(previous_by_id.url, None)
        previous_id = id_by_url.pop(listing.url, None)
        if previous_id is not None:
            selected.pop(previous_id, None)
        selected[listing.listing_id] = listing
        id_by_url[listing.url] = listing.listing_id
    return list(selected.values())


def clean_url(url: str) -> str:
    """Strip tracking parameters and fragments so links stay short and stable."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.query:
        return url
    host = parts.netloc.lower()
    is_google_jobs = (
        host.endswith("google.com") and parts.path == "/search" and "htidocid" in parts.query
    )
    kept = []
    for pair in parts.query.split("&"):
        key = pair.partition("=")[0].lower()
        if is_google_jobs:
            if key in GOOGLE_JOBS_PARAMS:
                kept.append(pair)
        elif key in TRACKING_PARAMS or key.startswith("utm_"):
            continue
        else:
            kept.append(pair)
    fragment = "" if is_google_jobs or parts.fragment.startswith("fpstate") else parts.fragment
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "&".join(kept), fragment))


def _digest_line(rank: int, listing: Listing) -> str:
    title = " ".join(listing.title.split())[:TITLE_LENGTH_LIMIT]
    title = title.replace("[", "(").replace("]", ")")
    company = " ".join(listing.company.split()).replace("[", "(").replace("]", ")")
    return f"{rank}. [{title} · {company}]({clean_url(listing.url)})"


def format_digest(
    ranked: Sequence[tuple[Listing, ScoreResult]],
    *,
    digest_date: date,
    footer: str,
    digest_size: int = 25,
    score_model: str = "gemma3:4b",
) -> tuple[str, list[discord.Embed], tuple[str, ...]]:
    """Build a links-only digest: numbered masked links, best-scoring first."""
    lines: list[str] = []
    shown_ids: list[str] = []
    used = 0
    for rank, (listing, _score) in enumerate(ranked, start=1):
        if len(lines) >= digest_size:
            break
        line = _digest_line(rank, listing)
        if len(line) > EMBED_DESCRIPTION_LIMIT or used + len(line) + 1 > EMBED_TOTAL_BUDGET:
            break
        lines.append(line)
        shown_ids.append(listing.listing_id)
        used += len(line) + 1

    chunks: list[list[str]] = []
    current: list[str] = []
    current_length = 0
    for line in lines:
        if current and current_length + len(line) + 1 > EMBED_DESCRIPTION_LIMIT:
            chunks.append(current)
            current = []
            current_length = 0
        current.append(line)
        current_length += len(line) + 1
    if current:
        chunks.append(current)

    embeds: list[discord.Embed] = []
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        title = f"Opportunity shortlist - {digest_date.isoformat()}"
        if total > 1:
            title += f" ({index}/{total})"
        embed = discord.Embed(title=title, description="\n".join(chunk))
        if index == total:
            embed.set_footer(text=f"{footer} · scored by local {score_model}")
        embeds.append(embed)
    return "", embeds, tuple(shown_ids)


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
    embedder: PipelineEmbedder | None = None,
    max_to_score: int = 50,
    digest_size: int = 25,
    score_model: str = "gemma3:4b",
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

        unique_listings = _identity_deduplicate(fetched)
        db.save_listings(unique_listings)
        unshown_ids = db.unshown_listing_ids(listing.listing_id for listing in unique_listings)
        new_listings = [listing for listing in unique_listings if listing.listing_id in unshown_ids]
        deduped_count = len(fetched) - len(new_listings)
        records = db.listing_records(listing.listing_id for listing in new_listings)
        embedding_errors: dict[str, str] = {}
        embedded_count = 0
        if embedder is not None:
            never_embedded = [
                listing
                for listing in new_listings
                if records[listing.listing_id]["embedded_at"] is None
            ]
            if never_embedded:
                embedding_batch = await embedder.embed_listings(never_embedded, profile)
                embedding_errors = embedding_batch.errors
                similarities = embedder.similarities(embedding_batch.embeddings)
                db.update_embeddings(similarities)
                embedded_count = len(similarities)
                records = db.listing_records(listing.listing_id for listing in new_listings)
            candidates = [
                listing
                for listing in new_listings
                if records[listing.listing_id]["score"] is None
                and records[listing.listing_id]["similarity"] is not None
            ]
            candidates.sort(
                key=lambda listing: (
                    -records[listing.listing_id]["similarity"],
                    listing.title.casefold(),
                    listing.company.casefold(),
                    listing.listing_id,
                )
            )
            judge_batch = await scorer.score_many(candidates[:max_to_score], profile)
            db.update_scores(judge_batch.scores)
            judged_count = len(judge_batch.scores)
            records = db.listing_records(listing.listing_id for listing in new_listings)
            batch = ScoreBatchResult(
                scores={
                    listing.listing_id: ScoreResult(
                        records[listing.listing_id]["score"],
                        records[listing.listing_id]["score_reason"],
                    )
                    for listing in new_listings
                    if records[listing.listing_id]["score"] is not None
                },
                errors=judge_batch.errors,
            )
        else:
            batch = await scorer.score_many(new_listings, profile)
            db.update_scores(batch.scores)
            judged_count = len(batch.scores)
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
            f"sources {len(source_stats)} · fetched {len(fetched)} · new {len(new_listings)} · "
            f"judged {judged_count}"
        )
        if source_errors:
            footer += f" · source errors: {len(source_errors)}"
        if batch.errors:
            footer += f" · scoring errors: {len(batch.errors)}"
        if embedding_errors:
            footer += f" · embedding errors: {len(embedding_errors)}"
        if ranked:
            message, embeds, shown_ids = format_digest(
                ranked,
                digest_date=date_value,
                footer=footer,
                digest_size=digest_size,
                score_model=score_model,
            )
            if shown_ids:
                empty = False
            else:
                message = (
                    f"Opportunity shortlist - {date_value.isoformat()}\n\n"
                    f"No links fit the size budget this run.\n\n{footer}"
                )
                embeds = []
                empty = True
        else:
            message = (
                f"Opportunity shortlist - {date_value.isoformat()}\n\n"
                f"No new matching jobs.\n\n{footer}"
            )
            shown_ids = ()
            embeds = []
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
            embedding_errors=embedding_errors,
            embedded_count=embedded_count,
            judged_count=judged_count,
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
                "embedded": embedded_count,
                "judged": judged_count,
            },
            source_errors=source_errors,
            scoring_errors=batch.errors,
            run_id=run_id,
            embedding_errors=embedding_errors,
            embeds=embeds,
        )
    finally:
        if own_client:
            await request_client.aclose()