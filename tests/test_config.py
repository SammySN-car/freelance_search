from pathlib import Path

import pytest
from pydantic import ValidationError

from freelance_search.config import Settings


def test_settings_require_secrets():
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_validate_query_count():
    with pytest.raises(ValidationError):
        Settings(
            discord_token="token",
            discord_owner_id=1,
            discord_channel_id=2,
            serpapi_api_key="key",
            serpapi_queries=["one", "two"],
        )


def test_settings_loads_example_dotenv_queries(tmp_path):
    dotenv = tmp_path / ".env.example"
    example = Path(__file__).parents[1] / ".env.example"
    dotenv.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")

    settings = Settings(_env_file=dotenv)

    assert settings.serpapi_queries == [
        "RAG engineer freelance",
        "LLM engineer freelance",
        "FastAPI machine learning freelance",
    ]
