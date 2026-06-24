"""
src/models/preferences.py
──────────────────────────
UserPreferences dataclass — represents validated, normalised user input
before it is passed to the filter and LLM layers.

All fields are set by PreferenceValidator + PreferenceNormalizer in
src/services/filter.py.  Nothing in this file mutates data; it is a
pure value object.
"""
from __future__ import annotations

from dataclasses import dataclass, field


VALID_BUDGETS: frozenset[str] = frozenset({"low", "medium", "high"})
RATING_MIN: float = 0.0
RATING_MAX: float = 5.0


@dataclass
class UserPreferences:
    """
    Validated and normalised representation of what the user wants.

    Fields
    ------
    location : str
        Title-cased city / locality name (e.g. "Bangalore").
        Required; must match at least one restaurant in the dataset.
    budget : str
        One of ``"low"``, ``"medium"``, ``"high"``.
    cuisine : str | None
        Lowercased primary cuisine preference (e.g. ``"north indian"``).
        ``None`` means "no preference" — cuisine filter is skipped.
    min_rating : float
        Minimum acceptable aggregate rating in ``[0.0, 5.0]``.
        ``0.0`` means "accept any rating".
    additional : str | None
        Free-text soft preferences (e.g. ``"family-friendly, outdoor seating"``).
        Passed as-is to the Groq LLM prompt; never used for hard filtering.
    relaxed_constraints : list[str]
        Populated by RestaurantFilter when it had to relax one or more
        hard constraints to find results.  Empty when all constraints held.
    """

    location: str
    budget: str
    min_rating: float
    cuisine: str | None = None
    additional: str | None = None

    # Set by RestaurantFilter when constraints are relaxed (EC-F-01)
    relaxed_constraints: list[str] = field(default_factory=list)

    # ── Convenience ───────────────────────────────────────────────────────────

    def cuisine_display(self) -> str:
        """Return a display-friendly cuisine string."""
        return self.cuisine.title() if self.cuisine else "No preference"

    def budget_display(self) -> str:
        """Return a display-friendly budget label."""
        labels = {"low": "Low (≤ ₹500)", "medium": "Medium (₹500–₹1500)", "high": "High (₹1500+)"}
        return labels.get(self.budget, self.budget.title())

    def to_filter_dict(self) -> dict:
        """
        Return a dict of the active hard-filter values.
        Used to populate ``RecommendationResponse.metadata.filters_applied``.
        """
        d: dict = {
            "location": self.location,
            "budget": self.budget,
            "min_rating": self.min_rating,
        }
        if self.cuisine:
            d["cuisine"] = self.cuisine
        if self.relaxed_constraints:
            d["relaxed"] = self.relaxed_constraints
        return d

    def __repr__(self) -> str:
        return (
            f"UserPreferences(location={self.location!r}, budget={self.budget!r}, "
            f"cuisine={self.cuisine!r}, min_rating={self.min_rating}, "
            f"additional={self.additional!r})"
        )
