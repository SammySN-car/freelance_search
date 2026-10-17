from pathlib import Path

import pytest
from pydantic import ValidationError

from freelance_search.config import Settings, compose_profile


def make_settings(**overrides):
    base = {
        "discord_token": "token",
        "discord_owner_id": 1,
        "discord_channel_id": 2,
        "serpapi_api_key": "key",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def test_settings_require_secrets():
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_validate_query_count():
    with pytest.raises(ValidationError):
        make_settings(serpapi_queries=["one", "two"])
    with pytest.raises(ValidationError):
        make_settings(serpapi_queries=[f"q{n}" for n in range(9)])


def test_settings_reject_out_of_range_digest_size():
    with pytest.raises(ValidationError):
        make_settings(digest_size=0)
    with pytest.raises(ValidationError):
        make_settings(digest_size=41)


def test_customization_defaults_are_unset():
    settings = make_settings()

    assert settings.location == ""
    assert settings.experience_level == ""
    assert settings.digest_size == 25
    assert settings.max_to_score == 50


def test_compose_profile_is_bare_profile_without_customization():
    settings = make_settings(profile="Python dev. SKIP: scrapers.")

    assert compose_profile(settings) == "Python dev. SKIP: scrapers."


def test_compose_profile_appends_location_and_level():
    settings = make_settings(
        profile="Python dev.",
        location="India",
        experience_level="beginner",
    )

    composed = compose_profile(settings)

    assert composed.startswith("Python dev.")
    assert "Candidate location: India" in composed
    assert "relocating" in composed
    assert "Candidate experience level: beginner" in composed
    assert "senior or lead expertise" in composed


def test_compose_profile_handles_whitespace_only_values():
    settings = make_settings(profile="  Python dev.  ", location="   ", experience_level=" ")

    assert compose_profile(settings) == "Python dev."


def test_settings_loads_example_dotenv_queries(tmp_path):
    dotenv = tmp_path / ".env.example"
    example = Path(__file__).parents[1] / ".env.example"
    dotenv.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")

    settings = Settings(_env_file=dotenv)

    assert settings.serpapi_queries == [
        "AI automation freelance",
        "RAG chatbot developer freelance",
        "AI engineer intern remote",
        "AI engineer entry level remote",
        "machine learning internship",
    ]
    assert settings.digest_size == 25
    assert settings.max_to_score == 50
    assert settings.location == ""
    assert settings.experience_level == "beginner"