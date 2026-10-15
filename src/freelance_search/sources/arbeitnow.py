import httpx

from ..models import Listing
from .base import require_success, source_id

ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"


class ArbeitnowSource:
    name = "arbeitnow"

    async def fetch(self, client: httpx.AsyncClient) -> list[Listing]:
        response = await client.get(ARBEITNOW_URL)
        require_success(response)
        listings = []
        for job in response.json().get("data", []):
            url = job.get("url")
            if not url:
                continue
            job_id = job.get("slug") or url
            listings.append(
                Listing(
                    listing_id=source_id(self.name, str(job_id)),
                    title=job.get("title", "Untitled job"),
                    company=job.get("company_name", "Unknown company"),
                    url=url,
                    source=self.name,
                    description=job.get("description", ""),
                    location=job.get("location") or None,
                )
            )
        return listings
