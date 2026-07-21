"""Runtime configuration, sourced from environment variables / ``.env``.

The whole point of this module is that the pipeline is *runnable with zero
secrets*. When ``provider == "mock"`` (the default) or no API key is present,
every LLM-backed stage falls back to a deterministic heuristic so
``docker compose up`` produces a complete, reproducible dashboard offline.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "openai", "mock"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    # ---- LLM provider selection ------------------------------------------
    llm_provider: Provider = Field("mock", alias="CS_INTEL_LLM_PROVIDER")
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field("claude-haiku-4-5-20251001", alias="CS_INTEL_ANTHROPIC_MODEL")
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", alias="CS_INTEL_OPENAI_MODEL")

    # ---- Langfuse tracing -------------------------------------------------
    langfuse_public_key: str = Field("", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field("", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field("http://localhost:3000", alias="LANGFUSE_HOST")

    # ---- Data generation --------------------------------------------------
    seed: int = Field(42, alias="CS_INTEL_SEED")
    n_accounts: int = Field(500, alias="CS_INTEL_N_ACCOUNTS")
    n_months: int = Field(24, alias="CS_INTEL_N_MONTHS")

    @property
    def effective_provider(self) -> Provider:
        """Resolve to ``mock`` whenever the selected provider lacks a key.

        This is what makes the offline demo path automatic: a user can set
        ``CS_INTEL_LLM_PROVIDER=anthropic`` but if the key is missing we never
        crash mid-pipeline — we transparently label with the heuristic.
        """
        if self.llm_provider == "anthropic" and self.anthropic_api_key:
            return "anthropic"
        if self.llm_provider == "openai" and self.openai_api_key:
            return "openai"
        return "mock"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
