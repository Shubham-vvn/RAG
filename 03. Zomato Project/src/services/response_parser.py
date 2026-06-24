"""
Response parser — parses and validates JSON responses from Groq LLM.

Extracts summary and recommendations, validates schema,
handles malformed/partial output gracefully.

Implementation: Phase 3 (Groq LLM Integration)
"""

import json
import logging

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Raised when Groq LLM response cannot be parsed or does not match schema."""
    pass


class ResponseParser:
    """Parses and validates JSON responses from Groq LLM."""

    def parse(self, raw_json: str) -> dict:
        """
        Parse Groq response JSON. Returns dict with 'summary' and 'recommendations'.
        Raises ParseError if response is fundamentally invalid or schema is violated.
        """
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid JSON from Groq: {e}") from e

        if not isinstance(data, dict):
            raise ParseError("Groq response is not a JSON object")

        # Validate top-level structure
        if "recommendations" not in data:
            raise ParseError("Groq response missing 'recommendations' field")

        if not isinstance(data["recommendations"], list):
            raise ParseError("'recommendations' must be a list")

        # Validate each recommendation record
        valid_recs = []
        for i, rec in enumerate(data["recommendations"]):
            if not isinstance(rec, dict):
                logger.warning(f"Recommendation {i} is not a dictionary, skipping.")
                continue
            if "id" not in rec or "explanation" not in rec:
                logger.warning(f"Recommendation {i} is missing 'id' or 'explanation', skipping.")
                continue
            
            # Coerce rank to int if present, otherwise default to i + 1
            rank = rec.get("rank")
            try:
                rank = int(rank) if rank is not None else i + 1
            except (ValueError, TypeError):
                rank = i + 1

            valid_recs.append(
                {
                    "id": str(rec["id"]).strip(),
                    "rank": rank,
                    "explanation": str(rec["explanation"]).strip(),
                }
            )

        if not valid_recs:
            raise ParseError("No valid recommendations found in Groq response")

        return {
            "summary": str(data.get("summary", "")).strip() if data.get("summary") else None,
            "recommendations": valid_recs,
        }

