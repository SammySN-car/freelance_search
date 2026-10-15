from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_PROFILE = (
    "AI/ML engineer - RAG & knowledge-base chatbots, LangGraph agents, "
    "Ollama/local LLM integration, FastAPI backends; Kaggle-proven applied ML "
    "(entity-resolution LB 0.924, MAP@3 0.767); junior-to-mid freelance. "
    "SKIP: data-entry, pure CRUD, scraping-only gigs."
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    discord_token: str = Field(min_length=1)
    discord_owner_id: int
    discord_channel_id: int
    serpapi_api_key: str = Field(min_length=1)
    digest_time: str = "08:00"
    digest_timezone: str = "Asia/Kolkata"
    database_path: Path = Path("data/freelance_search.db")
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "gemma3:4b"
    embedding_model: str = "mxbai-embed-large"
    ollama_timeout_seconds: float = Field(default=180.0, gt=0)
    max_to_score: int = Field(default=30, gt=0)
    min_score: float = Field(default=5.0, ge=0, le=10)
    serpapi_queries: list[str] = Field(
        default_factory=lambda: [
            "RAG engineer freelance",
            "LLM engineer freelance",
            "FastAPI machine learning freelance",
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
        if not 3 <= len(value) <= 5:
            raise ValueError("serpapi_queries must contain 3 to 5 queries")
        if any(not query.strip() for query in value):
            raise ValueError("serpapi_queries cannot contain blank queries")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
