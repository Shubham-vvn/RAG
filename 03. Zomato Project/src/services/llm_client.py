"""
Groq LLM client — adapter over the official groq Python SDK.

Handles primary model call, temperature retry on parse failure,
model fallback (primary → fallback), rate limit backoff, and
latency/token logging.

Implementation: Phase 3 (Groq LLM Integration)
"""

import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Optional

from groq import Groq, RateLimitError, APITimeoutError, InternalServerError

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    raw_json: str
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMClient:
    """Groq API adapter with retry, fallback, and error handling."""

    def __init__(self):
        # Allow passing mock or dummy keys for tests where GROQ_API_KEY is dummy
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def generate(self, system_prompt: str, user_prompt: str) -> Optional[LLMResult]:
        """
        Call Groq API with multi-tier fallback:
        1. Primary model (temperature=settings.GROQ_TEMPERATURE)
        2. Primary model (temperature=settings.GROQ_RETRY_TEMPERATURE) — retry on parse failure
        3. Fallback model (temperature=settings.GROQ_TEMPERATURE)
        Returns None if all attempts fail (caller should use heuristic fallback).
        """
        # Attempt 1: Primary model
        logger.info(f"LLM Attempt 1: Calling primary model {settings.GROQ_MODEL}")
        result = self._call_groq(
            system_prompt,
            user_prompt,
            model=settings.GROQ_MODEL,
            temperature=settings.GROQ_TEMPERATURE,
        )
        if result and self._is_valid_json(result.raw_json):
            return result

        # Attempt 2: Primary model with lower temperature on JSON parse failure
        logger.warning("Invalid JSON from primary model. Retrying with lower temperature.")
        result = self._call_groq(
            system_prompt,
            user_prompt,
            model=settings.GROQ_MODEL,
            temperature=settings.GROQ_RETRY_TEMPERATURE,
        )
        if result and self._is_valid_json(result.raw_json):
            return result

        # Attempt 3: Fallback model
        logger.warning(
            f"Primary model failed to yield valid JSON. Switching to fallback: {settings.GROQ_FALLBACK_MODEL}"
        )
        result = self._call_groq(
            system_prompt,
            user_prompt,
            model=settings.GROQ_FALLBACK_MODEL,
            temperature=settings.GROQ_TEMPERATURE,
        )
        if result and self._is_valid_json(result.raw_json):
            return result

        logger.error("All LLM attempts failed. Falling back to heuristic ranking.")
        return None

    def _call_groq(
        self, system_prompt: str, user_prompt: str, model: str, temperature: float
    ) -> Optional[LLMResult]:
        """Single Groq API call with rate limit retry and backoff."""
        max_attempts = settings.GROQ_MAX_RETRIES
        for attempt in range(1, max_attempts + 1):
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=settings.GROQ_MAX_TOKENS,
                    response_format={"type": "json_object"},
                )
                latency_ms = (time.time() - start) * 1000

                # Validate usage metadata presence
                prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
                completion_tokens = getattr(response.usage, "completion_tokens", 0)
                total_tokens = getattr(response.usage, "total_tokens", 0)

                result = LLMResult(
                    raw_json=response.choices[0].message.content,
                    model=response.model,
                    latency_ms=round(latency_ms, 1),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

                logger.info(
                    f"Groq response: model={result.model} "
                    f"latency={result.latency_ms}ms "
                    f"tokens={result.total_tokens}"
                )
                return result

            except RateLimitError:
                wait = (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    f"Groq 429 rate limit. Retrying in {wait:.1f}s (attempt {attempt}/{max_attempts})"
                )
                time.sleep(wait)

            except (APITimeoutError, InternalServerError) as e:
                logger.warning(f"Groq error ({e.__class__.__name__}): {e}. Attempt {attempt}/{max_attempts}")
                if attempt == max_attempts:
                    return None
                time.sleep(2**attempt)

            except Exception as e:
                logger.error(f"Unexpected Groq error: {e}")
                return None

        return None

    @staticmethod
    def _is_valid_json(text: str) -> bool:
        """Helper to verify string is valid JSON."""
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, TypeError):
            return False

