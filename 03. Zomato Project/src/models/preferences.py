"""
User preferences model — input contract for the recommendation pipeline.

Handles validation and normalization of user-supplied preferences
before they enter the filter and LLM layers.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class UserPreferences:
    """
    Validated and normalized user preferences.

    Fields:
        location: City or locality name (required).
        budget: Budget tier — "low", "medium", or "high" (required).
        min_rating: Minimum acceptable restaurant rating 0.0–5.0 (required).
        cuisine: Optional primary cuisine preference (e.g. "Italian").
        additional: Optional free-text preferences passed to LLM as soft signal
                    (e.g. "family-friendly, outdoor seating").
    """

    location: str
    budget: str                     # "low" | "medium" | "high"
    min_rating: float = 3.5
    cuisine: Optional[str] = None
    additional: Optional[str] = None

    # ── Validation ──

    def validate(self) -> list[str]:
        """
        Validate preferences against business rules.

        Returns:
            List of error message strings. Empty list means valid.
        """
        errors = []

        # Location: required, non-empty
        if not self.location or not self.location.strip():
            errors.append("Location is required and cannot be empty.")

        # Budget: must be one of the valid tiers
        if self.budget not in ("low", "medium", "high"):
            errors.append(
                f"Budget must be 'low', 'medium', or 'high'. Got: '{self.budget}'"
            )

        # Min rating: must be in valid range
        if not (0.0 <= self.min_rating <= 5.0):
            errors.append(
                f"Min rating must be between 0.0 and 5.0. Got: {self.min_rating}"
            )

        # Additional: length limit (prevent prompt token explosion)
        if self.additional and len(self.additional) > 500:
            errors.append(
                "Additional preferences must be 500 characters or less."
            )

        return errors

    # ── Normalization ──

    def normalize(self) -> "UserPreferences":
        """
        Return a normalized copy of preferences.

        Applies:
            - Title-case for location and cuisine
            - Lowercase for budget
            - Rating clamping to [0.0, 5.0]
            - Whitespace trimming on all string fields
        """
        return UserPreferences(
            location=self.location.strip().title(),
            budget=self.budget.lower().strip(),
            min_rating=max(0.0, min(5.0, self.min_rating)),
            cuisine=self.cuisine.strip().title() if self.cuisine else None,
            additional=self.additional.strip() if self.additional else None,
        )

    def to_dict(self) -> dict:
        """Serialize for logging and prompt building."""
        return {
            "location": self.location,
            "budget": self.budget,
            "min_rating": self.min_rating,
            "cuisine": self.cuisine,
            "additional": self.additional,
        }
