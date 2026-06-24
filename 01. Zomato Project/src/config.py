"""
src/config.py
─────────────
Centralised application settings loaded from environment variables via
pydantic-settings.  All runtime configuration lives here — nothing should
be hard-coded in other modules.

Usage
-----
    from src.config import settings

    print(settings.GROQ_MODEL)
    print(settings.BUDGET_THRESHOLDS)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# ── Project root (two levels up from this file: src/config.py → root) ────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Application-wide configuration.

    Values are resolved in this priority order:
      1. Environment variables
      2. .env file (auto-discovered from project root)
      3. Field defaults defined below
    """

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",          # ignore unknown env vars
    )

    # ── Hugging Face Dataset ──────────────────────────────────────────────────
    HF_DATASET_NAME: str = Field(
        default="ManikaSaini/zomato-restaurant-recommendation",
        description="Hugging Face dataset path.",
    )
    HF_TOKEN: str | None = Field(
        default=None,
        description="Optional HF token for gated datasets.",
    )

    # ── Local Cache ───────────────────────────────────────────────────────────
    DATA_CACHE_PATH: Path = Field(
        default=_PROJECT_ROOT / "data" / "zomato_cache.parquet",
        description="Local parquet snapshot of the HF dataset.",
    )
    MIN_EXPECTED_ROWS: int = Field(
        default=1000,
        description="Minimum rows expected after dataset load (sanity check).",
    )

    # ── Groq LLM ─────────────────────────────────────────────────────────────
    GROQ_API_KEY: str | None = Field(
        default=None,
        description="Groq API key. Required at runtime.",
    )
    GROQ_MODEL: str = Field(
        default="llama-3.3-70b-versatile",
        description="Primary Groq model for ranking and explanation.",
    )
    GROQ_FALLBACK_MODEL: str = Field(
        default="llama-3.1-8b-instant",
        description="Fallback Groq model (faster/cheaper).",
    )
    GROQ_TEMPERATURE: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Inference temperature. Keep low for consistent JSON.",
    )
    GROQ_TIMEOUT_SECONDS: int = Field(
        default=30,
        description="Request timeout for Groq API calls.",
    )

    # ── Filter & Recommendation Settings ─────────────────────────────────────
    MAX_CANDIDATES_FOR_LLM: int = Field(
        default=20,
        ge=1,
        description="Max restaurants passed to Groq (controls token cost).",
    )
    TOP_K_RECOMMENDATIONS: int = Field(
        default=5,
        ge=1,
        description="Number of recommendations displayed to the user.",
    )
    MAX_ADDITIONAL_LENGTH: int = Field(
        default=500,
        description="Max characters allowed in the 'additional' free-text field.",
    )
    MAX_EXPLANATION_LENGTH: int = Field(
        default=600,
        description="Max characters for a displayed LLM explanation.",
    )
    CUISINE_PARTIAL_MATCH: bool = Field(
        default=True,
        description="If True, 'Indian' matches 'North Indian', 'South Indian', etc.",
    )

    # ── Budget Tier Thresholds (INR cost_for_two) ─────────────────────────────
    # Restaurants with cost_for_two <= LOW_THRESHOLD  → "low"
    # Restaurants with cost_for_two <= HIGH_THRESHOLD → "medium"
    # Restaurants with cost_for_two >  HIGH_THRESHOLD → "high"
    BUDGET_LOW_THRESHOLD: int = Field(
        default=500,
        description="Max INR cost_for_two for 'low' budget tier.",
    )
    BUDGET_HIGH_THRESHOLD: int = Field(
        default=1500,
        description="Max INR cost_for_two for 'medium' budget tier (above = 'high').",
    )

    # ── Derived property ──────────────────────────────────────────────────────
    @property
    def BUDGET_THRESHOLDS(self) -> Dict[str, int]:
        """
        Returns a dict usable by the preprocessor and filter:
            {"low": 500, "medium": 1500}
        Budget tier is derived as:
            cost_for_two <= low   → "low"
            cost_for_two <= medium → "medium"
            else                  → "high"
        """
        return {
            "low": self.BUDGET_LOW_THRESHOLD,
            "medium": self.BUDGET_HIGH_THRESHOLD,
        }

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("DATA_CACHE_PATH", mode="before")
    @classmethod
    def _resolve_cache_path(cls, v: str | Path) -> Path:
        """Ensure DATA_CACHE_PATH is always an absolute Path object."""
        p = Path(v)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return p

    def validate_groq_key(self) -> None:
        """
        Raises ConfigurationError if GROQ_API_KEY is missing.
        Call this at application startup (not at import time so tests can run
        without a real key).
        """
        if not self.GROQ_API_KEY:
            raise ConfigurationError(
                "GROQ_API_KEY is not set. "
                "Please copy .env.example to .env and fill in your key."
            )

    # ── Security: mask key in repr ────────────────────────────────────────────
    def __repr__(self) -> str:
        key_preview = (
            f"{self.GROQ_API_KEY[:8]}***" if self.GROQ_API_KEY else "NOT SET"
        )
        return (
            f"Settings("
            f"GROQ_MODEL={self.GROQ_MODEL!r}, "
            f"GROQ_API_KEY={key_preview!r}, "
            f"HF_DATASET_NAME={self.HF_DATASET_NAME!r})"
        )


# ── Custom Exceptions ─────────────────────────────────────────────────────────

class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


class DataLoadError(RuntimeError):
    """Raised when the dataset cannot be loaded from HF or local cache."""


class SchemaError(ValueError):
    """Raised when the loaded dataset is missing expected columns."""


class ParseError(ValueError):
    """Raised when an LLM response cannot be parsed into the expected schema."""


class ValidationError(ValueError):
    """Raised when user input fails validation rules."""


class PromptBuildError(RuntimeError):
    """Raised when the prompt builder receives invalid input."""


# ── Singleton settings instance ───────────────────────────────────────────────
# Import this everywhere: `from src.config import settings`
settings = Settings()

logger.debug("Settings loaded: %r", settings)
