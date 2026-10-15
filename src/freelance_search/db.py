import sqlite3
from collections.abc import Iterable
from pathlib import Path

from .models import Listing

SCHEMA = """
CREATE TABLE IF NOT EXISTS listings (
    listing_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    company TEXT NOT NULL,
    url TEXT NOT NULL,
    source TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    pay TEXT,
    location TEXT,
    published_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS shown_listings (
    listing_id TEXT PRIMARY KEY REFERENCES listings(listing_id),
    shown_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    status TEXT NOT NULL,
    fetched_count INTEGER NOT NULL DEFAULT 0,
    new_count INTEGER NOT NULL DEFAULT 0,
    scored_count INTEGER NOT NULL DEFAULT 0,
    shown_count INTEGER NOT NULL DEFAULT 0,
    serpapi_queries_used INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def save_listings(self, listings: Iterable[Listing]) -> int:
        rows = [
            (
                listing.listing_id,
                listing.title,
                listing.company,
                listing.url,
                listing.source,
                listing.description,
                listing.pay,
                listing.location,
                listing.published_at.isoformat() if listing.published_at else None,
            )
            for listing in listings
        ]
        with self.connect() as connection:
            before = connection.total_changes
            connection.executemany(
                """
                INSERT OR IGNORE INTO listings
                (listing_id, title, company, url, source, description, pay, location, published_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            return connection.total_changes - before

    def unshown_listing_ids(self, listing_ids: Iterable[str]) -> set[str]:
        ids = list(listing_ids)
        if not ids:
            return set()
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT listing_id FROM listings
                WHERE listing_id IN ({placeholders})
                AND listing_id NOT IN (SELECT listing_id FROM shown_listings)
                """,
                ids,
            )
            return {row["listing_id"] for row in rows}

    def mark_shown(self, listing_ids: Iterable[str]) -> None:
        with self.connect() as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO shown_listings (listing_id) VALUES (?)",
                ((listing_id,) for listing_id in listing_ids),
            )
