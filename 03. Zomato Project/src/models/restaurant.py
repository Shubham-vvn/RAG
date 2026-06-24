"""
Restaurant data model — canonical schema for the Zomato dataset.

All restaurants are normalized to this schema during preprocessing.
See architecture.md §5.1 for the full field mapping.
"""

from dataclasses import dataclass, asdict
from enum import Enum


class BudgetTier(str, Enum):
    """Budget classification derived from cost_for_two."""

    LOW = "low"       # cost_for_two ≤ 500 INR
    MEDIUM = "medium"  # 501 – 1500 INR
    HIGH = "high"      # > 1500 INR


@dataclass
class Restaurant:
    """
    Canonical restaurant record.

    Fields are normalized from the raw Hugging Face dataset during preprocessing.
    The `budget_tier` is derived from `cost_for_two` using configurable thresholds.
    """

    id: str                    # stable identifier (index or dataset id)
    name: str                  # restaurant name
    location: str              # city / locality (normalized, title-case)
    cuisines: list[str]        # e.g. ["Italian", "Continental"]
    cost_for_two: int          # numeric cost indicator (INR)
    rating: float              # e.g. 4.2 (0.0–5.0)
    votes: int = 0             # popularity signal for tie-breaking
    rest_type: str = "Unknown"  # casual dining, cafe, quick bites, etc.
    budget_tier: BudgetTier = BudgetTier.MEDIUM  # derived from cost_for_two

    def to_dict(self) -> dict:
        """Full serialization for storage and debugging."""
        d = asdict(self)
        d["budget_tier"] = self.budget_tier.value
        return d

    def to_compact_dict(self) -> dict:
        """
        Minimal serialization for LLM prompt — saves tokens.

        Only includes the fields the LLM needs for ranking:
        id, name, cuisines (joined string), cost_for_two, rating.
        """
        return {
            "id": self.id,
            "name": self.name,
            "cuisines": ", ".join(self.cuisines),
            "cost_for_two": self.cost_for_two,
            "rating": self.rating,
        }

    def cuisine_display(self) -> str:
        """Joined cuisine string for UI display."""
        return ", ".join(self.cuisines)
