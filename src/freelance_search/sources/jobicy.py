import httpx

from ..models import Listing
from .base import require_success, source_id

JOBICY_URL = "https://jobicy.com/api/v2/remote-jobs"


class JobicySource:
    name = "jobicy"

    async def fetch(self, client: httpx.AsyncClient) -> list[Listing]:
        response = await client.get(JOBICY_URL)
        require_success(response)
        payload = response.json()
        listings = []
        for job in payload.get("jobs", []):
            url = job.get("url") or job.get("jobUrl")
            if not url:
                continue
            job_id = job.get("id") or url
            listings.append(
                Listing(
                    listing_id=source_id(self.name, str(job_id)),
                    title=job.get("jobTitle", "Untitled job"),
                    company=job.get("companyName", "Unknown company"),
                    url=url,
                    source=self.name,
                    description=job.get("jobDescription", ""),
                    pay=job.get("salary") or None,
                    location=job.get("jobGeo") or None,
                )
            )
        return listings
