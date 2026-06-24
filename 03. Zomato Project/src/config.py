"""
Centralized configuration using pydantic-settings.

All settings are loaded from environment variables and/or a `.env` file.
Configuration is validated at import time — the application fails fast
if required settings (e.g., GROQ_API_KEY) are missing.

Usage:
    from src.config import settings

    print(settings.GROQ_MODEL)         # "llama-3.3-70b-versatile"
    print(settings.GROQ_API_KEY)       # loaded from .env
"""

import logging
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with typed defaults and .env support."""

    # ── Hugging Face ──
    HF_DATASET_NAME: str = "ManikaSaini/zomato-restaurant-recommendation"
    HF_DATASET_SPLIT: str = "train"

    # ── Budget Thresholds (INR) ──
    BUDGET_LOW_MAX: int = 500
    BUDGET_MEDIUM_MAX: int = 1500

    # ── Candidates & Results ──
    MAX_CANDIDATES_FOR_LLM: int = 20
    TOP_K_RECOMMENDATIONS: int = 5

    # ── Groq ──
    GROQ_API_KEY: str  # Required — no default; fails fast if missing
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_FALLBACK_MODEL: str = "llama-3.1-8b-instant"
    GROQ_TEMPERATURE: float = 0.3
    GROQ_RETRY_TEMPERATURE: float = 0.1
    GROQ_MAX_TOKENS: int = 2048
    GROQ_MAX_RETRIES: int = 3

    # ── Data Cache ──
    DATA_CACHE_PATH: str = "./data/zomato_cache.parquet"

    # ── Logging ──
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def validate_config(self) -> list[str]:
        """
        Run additional validation rules beyond type checks.
        Returns a list of warning messages (empty if all OK).
        """
        warnings = []

        if self.BUDGET_LOW_MAX >= self.BUDGET_MEDIUM_MAX:
            warnings.append(
                f"BUDGET_LOW_MAX ({self.BUDGET_LOW_MAX}) must be less than "
                f"BUDGET_MEDIUM_MAX ({self.BUDGET_MEDIUM_MAX})"
            )

        if self.MAX_CANDIDATES_FOR_LLM < 1:
            warnings.append(
                f"MAX_CANDIDATES_FOR_LLM must be >= 1, got {self.MAX_CANDIDATES_FOR_LLM}"
            )

        if self.TOP_K_RECOMMENDATIONS < 1:
            warnings.append(
                f"TOP_K_RECOMMENDATIONS must be >= 1, got {self.TOP_K_RECOMMENDATIONS}"
            )

        if not (0.0 <= self.GROQ_TEMPERATURE <= 2.0):
            warnings.append(
                f"GROQ_TEMPERATURE should be between 0.0 and 2.0, got {self.GROQ_TEMPERATURE}"
            )

        if not (0.0 <= self.GROQ_RETRY_TEMPERATURE <= 2.0):
            warnings.append(
                f"GROQ_RETRY_TEMPERATURE should be between 0.0 and 2.0, "
                f"got {self.GROQ_RETRY_TEMPERATURE}"
            )

        return warnings


# ── Instantiate settings (validates at import time) ──
settings = Settings()

# ── Configure logging ──
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Run config validation and log warnings ──
_config_warnings = settings.validate_config()
if _config_warnings:
    _logger = logging.getLogger(__name__)
    for warning in _config_warnings:
        _logger.warning(f"Config validation: {warning}")
