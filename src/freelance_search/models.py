from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Listing:
    listing_id: str
    title: str
    company: str
    url: str
    source: str
    description: str = ""
    pay: str | None = None
    location: str | None = None
    published_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ScoreResult:
    score: float
    reason: str
