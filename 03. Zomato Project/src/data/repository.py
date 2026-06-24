"""
In-memory restaurant repository — query interface over preprocessed data.

Provides get_all(), get_locations(), get_cuisines(), get_count()
for the filter layer and UI dropdowns.

Implementation: Phase 1 (Data Ingestion Layer)
"""

import pandas as pd
from src.models.restaurant import Restaurant, BudgetTier


class RestaurantRepository:
    """In-memory query interface over preprocessed restaurant data."""

    def __init__(self, restaurants: list[Restaurant]):
        self._restaurants = restaurants
        self._locations = sorted(set(r.location for r in restaurants))
        self._cuisines = sorted(set(c for r in restaurants for c in r.cuisines))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "RestaurantRepository":
        """Build repository from preprocessed DataFrame."""
        restaurants = [
            Restaurant(
                id=str(row.get("id", idx)),
                name=row["name"],
                location=row["location"],
                cuisines=row["cuisines"],
                cost_for_two=int(row["cost_for_two"]),
                rating=float(row["rating"]),
                votes=int(row.get("votes", 0)),
                rest_type=str(row.get("rest_type", "Unknown")),
                budget_tier=BudgetTier(row["budget_tier"]),
            )
            for idx, row in df.iterrows()
        ]
        return cls(restaurants)

    def get_all(self) -> list[Restaurant]:
        """Return all restaurants in the repository."""
        return self._restaurants

    def get_locations(self) -> list[str]:
        """Return sorted, deduplicated list of locations."""
        return self._locations

    def get_cuisines(self) -> list[str]:
        """Return sorted, deduplicated list of cuisines."""
        return self._cuisines

    def get_count(self) -> int:
        """Return total count of restaurants."""
        return len(self._restaurants)

