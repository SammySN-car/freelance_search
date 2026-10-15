import pytest
from pydantic import ValidationError

from freelance_search.config import Settings


def test_settings_require_secrets():
    with pytest.raises(ValidationError):
        Settings()


def test_settings_validate_query_count():
    with pytest.raises(ValidationError):
        Settings(
            discord_token="token",
            discord_owner_id=1,
            discord_channel_id=2,
            serpapi_api_key="key",
            serpapi_queries=["one", "two"],
        )
