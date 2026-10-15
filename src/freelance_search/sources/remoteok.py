import httpx

from ..models import Listing
from .base import require_success, source_id

REMOTEOK_URL = "https://remoteok.com/api"


class RemoteOKSource:
    name = "remoteok"

    async def fetch(self, client: httpx.AsyncClient) -> list[Listing]:
        response = await client.get(REMOTEOK_URL)
        require_success(response)
        listings = []
        for job in response.json():
            if not isinstance(job, dict) or not job.get("url"):
                continue
            job_id = job.get("id") or job["url"]
            listings.append(
                Listing(
                    listing_id=source_id(self.name, str(job_id)),
                    title=job.get("position", "Untitled job"),
                    company=job.get("company", "Unknown company"),
                    url=job["url"],
                    source=self.name,
                    description=job.get("description", ""),
                    pay=job.get("salary") or None,
                    location=job.get("location") or None,
                )
            )
        return listings
