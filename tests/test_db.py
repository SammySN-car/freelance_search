from freelance_search.models import Listing


def test_database_stores_listings_and_excludes_shown(database):
    listing = Listing(
        listing_id="remoteok:123",
        title="RAG Engineer",
        company="Acme",
        url="https://example.test/jobs/123",
        source="remoteok",
    )

    assert database.save_listings([listing]) == 1
    assert database.unshown_listing_ids([listing.listing_id]) == {listing.listing_id}

    database.mark_shown([listing.listing_id])

    assert database.unshown_listing_ids([listing.listing_id]) == set()


def test_database_excludes_a_listing_with_a_previously_shown_url(database):
    shown = Listing(
        listing_id="source-a:123",
        title="RAG Engineer",
        company="Acme",
        url="https://example.test/jobs/shared",
        source="source-a",
    )
    duplicate = Listing(
        listing_id="source-b:456",
        title="RAG Engineer",
        company="Acme",
        url=shown.url,
        source="source-b",
    )
    database.save_listings([shown, duplicate])
    database.mark_shown([shown.listing_id])

    assert database.unshown_listing_ids([duplicate.listing_id]) == set()


def test_database_migrates_old_schema_with_retrieval_and_judge_columns(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE listings (listing_id TEXT PRIMARY KEY, title TEXT, company TEXT, "
            "url TEXT, source TEXT, description TEXT, pay TEXT, location TEXT, published_at TEXT)"
        )
        connection.execute(
            "CREATE TABLE runs (run_id INTEGER PRIMARY KEY, status TEXT)"
        )

    from freelance_search.db import Database

    migrated = Database(path)
    migrated.initialize()

    with migrated.connect() as connection:
        listing_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(listings)")
        }
        run_columns = {row["name"] for row in connection.execute("PRAGMA table_info(runs)")}
    assert {"similarity", "embedded_at", "score", "score_reason", "judged_at"} <= listing_columns
    assert {"embedded_count", "judged_count", "embedding_errors"} <= run_columns
