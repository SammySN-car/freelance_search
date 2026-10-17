import httpx

from ..models import Listing
from .base import require_success, source_id

SERPAPI_URL = "https://serpapi.com/search.json"


class SerpApiSource:
    name = "serpapi"

    def __init__(self, api_key: str, queries: list[str], location: str = "") -> None:
        self.api_key = api_key
        self.queries = queries
        self.location = location.strip()

    async def fetch(self, client: httpx.AsyncClient) -> list[Listing]:
        listings: list[Listing] = []
        for query in self.queries:
            params = {"engine": "google_jobs", "q": query, "api_key": self.api_key}
            if self.location:
                params["location"] = self.location
            response = await client.get(SERPAPI_URL, params=params)
            require_success(response)
            payload = response.json()
            for job in payload.get("jobs_results", []):
                # source_link is the origin posting and stays valid;
                # share_link is Google's session URL (htidocid) and rots in days
                url = job.get("source_link") or job.get("share_link")
                if not url:
                    continue
                job_id = job.get("job_id") or url
                listings.append(
                    Listing(
                        listing_id=source_id(self.name, str(job_id)),
                        title=job.get("title", "Untitled job"),
                        company=job.get("company_name", "Unknown company"),
                        url=url,
                        source=self.name,
                        description=job.get("description", ""),
                        pay=job.get("detected_extensions", {}).get("salary"),
                        location=job.get("location"),
                    )
                )
        return listings
