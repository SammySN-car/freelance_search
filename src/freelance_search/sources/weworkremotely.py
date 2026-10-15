import feedparser
import httpx

from ..models import Listing
from .base import require_success, source_id

WWR_RSS_URL = "https://weworkremotely.com/remote-jobs.rss"


class WeWorkRemotelySource:
    name = "weworkremotely"

    async def fetch(self, client: httpx.AsyncClient) -> list[Listing]:
        response = await client.get(WWR_RSS_URL)
        require_success(response)
        feed = feedparser.parse(response.content)
        listings = []
        for entry in feed.entries:
            url = entry.get("link")
            if not url:
                continue
            listings.append(
                Listing(
                    listing_id=source_id(self.name, url),
                    title=entry.get("title", "Untitled job"),
                    company=entry.get("author", "Unknown company"),
                    url=url,
                    source=self.name,
                    description=entry.get("summary", ""),
                )
            )
        return listings
