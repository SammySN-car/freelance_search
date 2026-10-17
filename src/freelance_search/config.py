from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import GithubVerdict

DEFAULT_PROFILE = (
    "Example profile - replace with your own: skills, stack, what kind of work you "
    "want (freelance, internship, junior role), and what to skip "
    "(e.g. SKIP: data-entry, pure CRUD)."
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    discord_token: str = Field(min_length=1)
    discord_owner_id: int
    discord_channel_id: int
    serpapi_api_key: str = Field(min_length=1)
    digest_time: str = "08:00"
    digest_timezone: str = "Asia/Kolkata"
    digest_size: int = Field(default=25, ge=1, le=40)
    database_path: Path = Path("data/freelance_search.db")
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "gemma3:4b"
    embedding_model: str = "mxbai-embed-large"
    ollama_timeout_seconds: float = Field(default=180.0, gt=0)
    max_to_score: int = Field(default=50, gt=0)
    min_score: float = Field(default=5.0, ge=0, le=10)
    location: str = ""
    experience_level: str = ""
    github_url: str = ""
    github_refresh_days: int = Field(default=2, ge=1)
    serpapi_queries: list[str] = Field(
        default_factory=lambda: [
            "AI automation freelance",
            "RAG chatbot developer freelance",
            "AI engineer intern remote",
            "machine learning intern remote",
            "junior machine learning engineer remote",
        ]
    )
    profile: str = DEFAULT_PROFILE

    @field_validator("digest_time")
    @classmethod
    def validate_digest_time(cls, value: str) -> str:
        parts = value.split(":")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            raise ValueError("digest_time must use HH:MM format")
        hour, minute = (int(part) for part in parts)
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            raise ValueError("digest_time must be a valid 24-hour time")
        return value

    @field_validator("serpapi_queries")
    @classmethod
    def validate_query_count(cls, value: list[str]) -> list[str]:
        if not 3 <= len(value) <= 8:
            raise ValueError("serpapi_queries must contain 3 to 8 queries")
        if any(not query.strip() for query in value):
            raise ValueError("serpapi_queries cannot contain blank queries")
        return value


def compose_profile(
    settings: Settings, github: GithubVerdict | None = None
) -> str:
    """Combine the free-text profile with location and experience evidence.

    A GitHub verdict, when available, replaces the hand-typed EXPERIENCE_LEVEL.
    """
    parts = [settings.profile.strip()]
    location = settings.location.strip()
    if location:
        parts.append(
            f"Candidate location: {location}. Prefer remote roles or roles compatible "
            "with this location; score lower any role that requires relocating or "
            "being on-site somewhere else."
        )
    level = settings.experience_level.strip()
    if github is not None:
        parts.append(
            f"Candidate experience level: {github.level} (detected by reading their "
            f"GitHub code). Evidence: {github.summary}"
        )
    elif level:
        parts.append(
            f"Candidate experience level: {level}. Prefer opportunities that fit this "
            "level; score lower roles demanding senior or lead expertise."
        )
    return "\n".join(part for part in parts if part)


@lru_cache
def get_settings() -> Settings:
    return Settings()