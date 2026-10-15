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
