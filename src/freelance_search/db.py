import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime
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
    deduped_count INTEGER NOT NULL DEFAULT 0,
    scored_count INTEGER NOT NULL DEFAULT 0,
    shown_count INTEGER NOT NULL DEFAULT 0,
    serpapi_queries_used INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    source_stats TEXT NOT NULL DEFAULT '{}',
    scoring_errors TEXT NOT NULL DEFAULT '{}'
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
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(runs)")
            }
            migrations = {
                "deduped_count": (
                    "ALTER TABLE runs ADD COLUMN deduped_count INTEGER NOT NULL DEFAULT 0"
                ),
                "source_stats": (
                    "ALTER TABLE runs ADD COLUMN source_stats TEXT NOT NULL DEFAULT '{}'"
                ),
                "scoring_errors": (
                    "ALTER TABLE runs ADD COLUMN scoring_errors TEXT NOT NULL DEFAULT '{}'"
                ),
            }
            for column, statement in migrations.items():
                if column not in columns:
                    connection.execute(statement)

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
                AND NOT EXISTS (
                    SELECT 1
                    FROM shown_listings AS shown
                    JOIN listings AS shown_listing
                      ON shown_listing.listing_id = shown.listing_id
                    WHERE shown.listing_id = listings.listing_id
                       OR shown_listing.url = listings.url
                )
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

    def start_run(self) -> int:
        with self.connect() as connection:
            cursor = connection.execute("INSERT INTO runs (status) VALUES (?)", ("running",))
            return int(cursor.lastrowid)

    def finish_run(
        self,
        run_id: int,
        *,
        status: str,
        fetched_count: int,
        new_count: int,
        deduped_count: int,
        scored_count: int,
        source_stats: dict[str, dict[str, int]],
        scoring_errors: dict[str, str],
        serpapi_queries_used: int = 0,
        error: str | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET completed_at = ?, status = ?, fetched_count = ?, new_count = ?,
                    deduped_count = ?, scored_count = ?, source_stats = ?,
                    scoring_errors = ?, serpapi_queries_used = ?, error = ?
                WHERE run_id = ?
                """,
                (
                    datetime.now().isoformat(),
                    status,
                    fetched_count,
                    new_count,
                    deduped_count,
                    scored_count,
                    json.dumps(source_stats, sort_keys=True),
                    json.dumps(scoring_errors, sort_keys=True),
                    serpapi_queries_used,
                    error,
                    run_id,
                )
            )

    def record_delivery(self, run_id: int, shown_count: int) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE runs SET shown_count = ? WHERE run_id = ?",
                (shown_count, run_id),
            )

    def record_run_error(self, run_id: int, error: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET completed_at = ?, status = 'failed', error = ?
                WHERE run_id = ?
                """,
                (datetime.now().isoformat(), error, run_id),
            )

    def latest_run(self) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                "SELECT * FROM runs ORDER BY run_id DESC LIMIT 1"
            ).fetchone()
