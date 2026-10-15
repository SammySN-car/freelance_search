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
