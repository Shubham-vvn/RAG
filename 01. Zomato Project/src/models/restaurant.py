"""
src/models/restaurant.py
────────────────────────
Canonical Restaurant data model.

All raw Hugging Face dataset rows are normalised into this schema by
DataPreprocessor before any other layer touches the data.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Restaurant:
    """
    Canonical representation of a single restaurant record.

    Fields
    ------
    id : str
        Stable unique identifier (string-cast row index from the dataset).
    name : str
        Restaurant name, as-is from the dataset (Unicode preserved).
    location : str
        City / locality — title-cased and alias-mapped during preprocessing.
    cuisines : list[str]
        Lowercased cuisine types, e.g. ["north indian", "chinese"].
        Empty list if the raw value was null or unparseable.
    cost_for_two : int
        Estimated cost for two people in INR.  0 means "not available".
    rating : float
        Aggregate rating on a 0.0–5.0 scale.
        0.0 means "not yet rated" (was "NEW" or "-" in the raw data).
    votes : int
        Number of user votes — used as a popularity tie-breaker.
    rest_type : str
        Restaurant category, e.g. "Casual Dining", "Café".
        Empty string if not available.
    budget_tier : str
        Derived budget category: "low" | "medium" | "high".
        Set by DataPreprocessor using BUDGET_THRESHOLDS from config.
    """

    id: str
    name: str
    location: str
    cuisines: list[str] = field(default_factory=list)
    cost_for_two: int = 0
    rating: float = 0.0
    votes: int = 0
    rest_type: str = ""
    budget_tier: str = ""

    # ── Convenience helpers ───────────────────────────────────────────────────

    def cuisine_display(self, max_items: int = 3) -> str:
        """
        Returns a comma-separated cuisine string for UI display.
        Caps at *max_items* items and appends '& more' if truncated.

        Examples
        --------
        >>> r = Restaurant("1", "X", "Delhi", ["north indian", "chinese", "italian", "thai"])
        >>> r.cuisine_display()
        'North Indian, Chinese, Italian & more'
        """
        items = [c.title() for c in self.cuisines]
        if len(items) <= max_items:
            return ", ".join(items) if items else "Not listed"
        return ", ".join(items[:max_items]) + " & more"

    def rating_display(self) -> str:
        """
        Returns a human-readable rating string.

        Examples
        --------
        >>> Restaurant("1", "X", "Y", rating=0.0).rating_display()
        'Not yet rated'
        >>> Restaurant("1", "X", "Y", rating=4.2).rating_display()
        '4.2 ⭐'
        """
        if self.rating == 0.0:
            return "Not yet rated"
        return f"{self.rating:.1f} ⭐"

    def cost_display(self) -> str:
        """
        Returns a human-readable cost string.

        Examples
        --------
        >>> Restaurant("1", "X", "Y", cost_for_two=0).cost_display()
        'Cost not available'
        >>> Restaurant("1", "X", "Y", cost_for_two=800).cost_display()
        '₹800'
        """
        if self.cost_for_two <= 0:
            return "Cost not available"
        return f"₹{self.cost_for_two:,}"

    def matches_cuisine(self, preference: str, partial: bool = True) -> bool:
        """
        Returns True if *preference* matches any of this restaurant's cuisines.

        Parameters
        ----------
        preference : str
            Lowercased cuisine preference from the user.
        partial : bool
            If True, 'indian' matches 'north indian', 'south indian', etc.
            If False, only exact matches count.
        """
        pref = preference.lower().strip()
        if partial:
            return any(pref in c for c in self.cuisines)
        return pref in self.cuisines

    def to_prompt_dict(self) -> dict:
        """
        Returns a compact dict suitable for inclusion in the LLM prompt.
        Excludes heavy/irrelevant fields to keep token count low.
        """
        return {
            "id": self.id,
            "name": self.name,
            "location": self.location,
            "cuisines": [c.title() for c in self.cuisines],
            "cost_for_two": self.cost_for_two,
            "rating": self.rating,
        }

    def __repr__(self) -> str:
        return (
            f"Restaurant(id={self.id!r}, name={self.name!r}, "
            f"location={self.location!r}, rating={self.rating}, "
            f"budget_tier={self.budget_tier!r})"
        )
