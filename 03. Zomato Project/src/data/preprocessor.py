"""
Data preprocessor — normalizes raw dataset columns to canonical schema.

Handles column renaming, cuisine string parsing, numeric coercion,
location normalization, budget tier derivation, and null handling.

Implementation: Phase 1 (Data Ingestion Layer)
"""

import pandas as pd
from src.config import settings
from src.models.restaurant import BudgetTier

CITY_ALIASES = {
    "Bengaluru": "Bangalore",
    "Bombay": "Mumbai",
    "Calcutta": "Kolkata",
    "Madras": "Chennai",
    "Ncr": "New Delhi",  # When title-cased, "NCR" becomes "Ncr"
    "New Delhi": "New Delhi",
}

COLUMN_MAP = {
    "name": "name",
    "rate": "rating",
    "approx_cost(for two people)": "cost_for_two",
    "location": "location",
    "cuisines": "cuisines",
    "votes": "votes",
    "rest_type": "rest_type",
}


class DataPreprocessor:
    """Normalizes raw dataset columns to the canonical schema."""

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the raw DataFrame columns, perform data validation,
        and generate derived columns like budget_tier.
        """
        # Copy to avoid side effects
        df = df.copy()

        # Rename columns to standard scheme
        rename_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        # Drop rows with null or empty name/location
        df = df.dropna(subset=["name", "location"])
        df = df[df["name"].str.strip() != ""]
        df = df[df["location"].str.strip() != ""]

        # Parse rating
        df["rating"] = df["rating"].apply(self._parse_rating)
        # Parse cost
        df["cost_for_two"] = df["cost_for_two"].apply(self._parse_cost)

        # Drop rows with invalid rating or cost
        df = df.dropna(subset=["rating", "cost_for_two"])

        # Coerce votes to integer
        df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0).astype(int)

        # Clean cuisines
        df["cuisines"] = df["cuisines"].apply(self._parse_cuisines)

        # Clean rest_type
        df["rest_type"] = df["rest_type"].fillna("Unknown").str.strip().str.title()

        # Normalize location
        df["location"] = df["location"].apply(self._normalize_location)

        # Derive budget_tier
        df["budget_tier"] = df["cost_for_two"].apply(self._derive_budget_tier)

        # Sort by rating and votes descending to keep the best rated/most popular restaurants
        df = df.sort_values(by=["rating", "votes"], ascending=[False, False])

        # Drop duplicate records of the same restaurant (based on name and location)
        df = df.drop_duplicates(subset=["name", "location"], keep="first")

        # Generate unique stable ID from index as string
        df["id"] = df.index.astype(str)

        # Return only standard columns to keep memory clean
        standard_cols = [
            "id",
            "name",
            "location",
            "cuisines",
            "cost_for_two",
            "rating",
            "votes",
            "rest_type",
            "budget_tier",
        ]
        return df[standard_cols]

    def _parse_rating(self, val) -> float | None:
        """Parse rating value into float. Returns None if invalid."""
        if pd.isna(val):
            return None
        val_str = str(val).strip()
        if not val_str or val_str in ("NEW", "-", "NEW/5", "-/5"):
            return None
        if "/" in val_str:
            val_str = val_str.split("/")[0].strip()
        try:
            r = float(val_str)
            if 0.0 <= r <= 5.0:
                return r
        except ValueError:
            pass
        return None

    def _parse_cost(self, val) -> int | None:
        """Parse approx_cost(for two people) into integer. Returns None if invalid."""
        if pd.isna(val):
            return None
        val_str = str(val).replace(",", "").strip()
        try:
            c = int(val_str)
            if c > 0:
                return c
        except ValueError:
            pass
        return None

    def _parse_cuisines(self, val) -> list[str]:
        """Split cuisines string into a clean list."""
        if pd.isna(val) or not str(val).strip():
            return ["Unknown"]
        return [c.strip() for c in str(val).split(",") if c.strip()]

    def _normalize_location(self, val) -> str:
        """Title case and apply city alias mappings."""
        if pd.isna(val):
            return "Unknown"
        loc = str(val).strip().title()
        return CITY_ALIASES.get(loc, loc)

    def _derive_budget_tier(self, cost: int) -> str:
        """Derive low/medium/high budget tier from cost."""
        if cost <= settings.BUDGET_LOW_MAX:
            return BudgetTier.LOW.value
        elif cost <= settings.BUDGET_MEDIUM_MAX:
            return BudgetTier.MEDIUM.value
        else:
            return BudgetTier.HIGH.value

