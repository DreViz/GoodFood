# app/config.py
"""
Central application configuration.

Single source of truth for runtime settings, read from environment variables
(and an optional `.env` file at the project root). Every field has a safe
local-development default so the app boots without any `.env` present.

Use `get_settings()` instead of hardcoding values; the result is cached, so
all call sites share one Settings instance.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Base URL only; endpoint paths are derived (see ollama_generate_url).
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:4b"
    # qwen3 is a reasoning model. Disabling "think" keeps latency sane on CPU
    # and yields clean output. A short reasoning preamble may still appear and
    # is stripped at the call site (see strip_model_reasoning).
    ollama_think: bool = False
    # Per-request timeout (seconds). qwen3:4b runs ~6 tok/s CPU-only, so keep
    # this generous for the larger planner prompts.
    ollama_timeout: int = 180

    database_url: str = "postgresql://postgres:postgres@localhost:5432/goodfoods"

    goodfoods_email: str = ""
    goodfoods_email_password: str = ""

    # Comma-separated list of allowed CORS origins.
    api_cors_origins: str = "http://localhost:3000,http://localhost:8501"

    @property
    def ollama_generate_url(self) -> str:
        return f"{self.ollama_base_url.rstrip('/')}/api/generate"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.api_cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached Settings instance."""
    return Settings()
