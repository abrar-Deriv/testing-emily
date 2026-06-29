from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    litellm_api_key: str = Field("sk-no-key", alias="litellm_key")
    litellm_base_url: str = "https://api.openai.com/v1"

    model_extraction: str = "gpt-5"
    model_sentiment: str = "gpt-5"
    model_briefing: str = "gpt-5"

    # HTTP request timeout (seconds)
    http_timeout: int = 30

    # Maximum tokens per content chunk sent to LLM
    max_chunk_tokens: int = 4000

    # Playwright for JS-rendered pages
    use_playwright: bool = True


settings = Settings()
