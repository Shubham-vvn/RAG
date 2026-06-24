"""
tests/test_preprocessor.py
───────────────────────────
Unit tests for DataPreprocessor covering all acceptance criteria from
Phase 1 §1.6 and the edge cases from edge-cases.md (EC-P-* and EC-D-04).

Run with:
    pytest tests/test_preprocessor.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.config import SchemaError
from src.data.preprocessor import COLUMN_MAP, DataPreprocessor
from src.models.restaurant import Restaurant


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_raw(**overrides) -> pd.DataFrame:
    """Build a single-row raw DataFrame with sensible defaults."""
    defaults = {
        "name":                 "Test Restaurant",
        "location":             "Bangalore",
        "cuisines":             "North Indian, Chinese",
        "average_cost_for_two": 800,
        "aggregate_rating":     4.2,
        "votes":                500,
        "rest_type":            "Casual Dining",
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


# ── 1. Column validation (EC-D-04) ───────────────────────────────────────────

class TestColumnValidation:
    def test_raises_schema_error_on_missing_column(self):
        """EC-D-04: Missing required column should raise SchemaError."""
        df = _make_raw()
        df = df.drop(columns=["aggregate_rating"])
        with pytest.raises(SchemaError, match="aggregate_rating"):
            DataPreprocessor().process(df)

    def test_raises_schema_error_lists_all_missing(self):
        """SchemaError message should list every missing column."""
        df = pd.DataFrame({"irrelevant_col": [1]})
        with pytest.raises(SchemaError) as exc_info:
            DataPreprocessor().process(df)
        # All required column keys should appear in error message
        for col in COLUMN_MAP:
            assert col in str(exc_info.value)

    def test_passes_with_extra_columns(self):
        """Extra columns in the DataFrame should be ignored silently."""
        df = _make_raw()
        df["extra_column"] = "ignored"
        result = DataPreprocessor().process(df)
        assert len(result) == 1


# ── 2. Cuisine parsing ────────────────────────────────────────────────────────

class TestCuisineParsing:
    def test_splits_comma_separated_cuisines(self):
        """'North Indian, Chinese' → ['north indian', 'chinese']"""
        result = DataPreprocessor().process(_make_raw(cuisines="North Indian, Chinese"))
        assert result[0].cuisines == ["north indian", "chinese"]

    def test_null_cuisine_gives_empty_list(self):
        """EC-P-01: Null cuisine field → []"""
        result = DataPreprocessor().process(_make_raw(cuisines=None))
        assert result[0].cuisines == []

    def test_integer_cuisine_gives_empty_list(self):
        """EC-P-02: Non-string cuisine (int 0) → []"""
        result = DataPreprocessor().process(_make_raw(cuisines=0))
        assert result[0].cuisines == []

    def test_boolean_false_cuisine_gives_empty_list(self):
        """EC-P-02: Non-string cuisine (False) → []"""
        result = DataPreprocessor().process(_make_raw(cuisines=False))
        assert result[0].cuisines == []

    def test_single_cuisine(self):
        """A single cuisine string should produce a one-element list."""
        result = DataPreprocessor().process(_make_raw(cuisines="Italian"))
        assert result[0].cuisines == ["italian"]

    def test_cuisines_are_lowercase(self):
        """All cuisines must be stored in lowercase."""
        result = DataPreprocessor().process(_make_raw(cuisines="NORTH INDIAN, CHINESE"))
        assert all(c == c.lower() for c in result[0].cuisines)

    def test_many_cuisines(self):
        """EC-P-07: More than 10 cuisines should all be preserved."""
        many = ", ".join([f"Cuisine{i}" for i in range(12)])
        result = DataPreprocessor().process(_make_raw(cuisines=many))
        assert len(result[0].cuisines) == 12


# ── 3. Rating coercion ────────────────────────────────────────────────────────

class TestRatingCoercion:
    def test_string_new_becomes_zero(self):
        """EC-P-03: 'NEW' rating → 0.0"""
        result = DataPreprocessor().process(_make_raw(aggregate_rating="NEW"))
        assert result[0].rating == 0.0

    def test_dash_becomes_zero(self):
        """EC-P-03: '-' rating → 0.0"""
        result = DataPreprocessor().process(_make_raw(aggregate_rating="-"))
        assert result[0].rating == 0.0

    def test_valid_float_rating(self):
        """Numeric rating should be preserved as float."""
        result = DataPreprocessor().process(_make_raw(aggregate_rating=4.5))
        assert result[0].rating == pytest.approx(4.5)

    def test_rating_clamped_above_five(self):
        """Ratings > 5.0 should be clamped to 5.0."""
        result = DataPreprocessor().process(_make_raw(aggregate_rating=6.0))
        assert result[0].rating == 5.0

    def test_rating_clamped_below_zero(self):
        """Ratings < 0.0 should be clamped to 0.0."""
        result = DataPreprocessor().process(_make_raw(aggregate_rating=-1.0))
        assert result[0].rating == 0.0


# ── 4. Cost coercion ──────────────────────────────────────────────────────────

class TestCostCoercion:
    def test_zero_cost(self):
        """EC-P-04: cost_for_two = 0 is valid and preserved."""
        result = DataPreprocessor().process(_make_raw(average_cost_for_two=0))
        assert result[0].cost_for_two == 0

    def test_negative_cost_becomes_zero(self):
        """EC-P-04: Negative cost → 0."""
        result = DataPreprocessor().process(_make_raw(average_cost_for_two=-500))
        assert result[0].cost_for_two == 0

    def test_null_cost_becomes_zero(self):
        """Null cost → 0."""
        result = DataPreprocessor().process(_make_raw(average_cost_for_two=None))
        assert result[0].cost_for_two == 0


# ── 5. Location normalization ─────────────────────────────────────────────────

class TestLocationNormalization:
    def test_title_case_applied(self):
        """'bangalore' → 'Bangalore'"""
        result = DataPreprocessor().process(_make_raw(location="bangalore"))
        assert result[0].location == "Bangalore"

    def test_strips_whitespace(self):
        """'  Bangalore  ' → 'Bangalore'"""
        result = DataPreprocessor().process(_make_raw(location="  Bangalore  "))
        assert result[0].location == "Bangalore"

    def test_bengaluru_alias(self):
        """EC-P-06: 'Bengaluru' → 'Bangalore'"""
        result = DataPreprocessor().process(_make_raw(location="Bengaluru"))
        assert result[0].location == "Bangalore"

    def test_bombay_alias(self):
        """EC-P-06: 'Bombay' → 'Mumbai'"""
        result = DataPreprocessor().process(_make_raw(location="Bombay"))
        assert result[0].location == "Mumbai"

    def test_all_caps_location(self):
        """'DELHI' → 'Delhi'"""
        result = DataPreprocessor().process(_make_raw(location="DELHI"))
        assert result[0].location == "Delhi"


# ── 6. Null row dropping ──────────────────────────────────────────────────────

class TestNullRowDropping:
    def test_drops_row_with_null_name(self):
        """EC-P-08: Row with null name is dropped."""
        result = DataPreprocessor().process(_make_raw(name=None))
        assert len(result) == 0

    def test_drops_row_with_empty_name(self):
        """EC-P-08: Row with empty string name is dropped."""
        result = DataPreprocessor().process(_make_raw(name=""))
        assert len(result) == 0

    def test_drops_row_with_null_location(self):
        """Row with null location is dropped."""
        result = DataPreprocessor().process(_make_raw(location=None))
        assert len(result) == 0

    def test_keeps_row_with_zero_rating(self):
        """A row with rating=0.0 is NOT dropped (unrated, but valid)."""
        result = DataPreprocessor().process(_make_raw(aggregate_rating="NEW"))
        assert len(result) == 1


# ── 7. Deduplication ──────────────────────────────────────────────────────────

class TestDeduplication:
    def test_exact_duplicate_removed(self):
        """EC-P-05: Two identical rows → 1 kept."""
        df = pd.concat([_make_raw(), _make_raw()], ignore_index=True)
        result = DataPreprocessor().process(df)
        assert len(result) == 1

    def test_duplicate_keeps_higher_votes(self):
        """EC-P-05: Among duplicates, the one with more votes is kept."""
        low_votes  = _make_raw(votes=100)
        high_votes = _make_raw(votes=999)
        df = pd.concat([low_votes, high_votes], ignore_index=True)
        result = DataPreprocessor().process(df)
        assert result[0].votes == 999

    def test_case_insensitive_dedup(self):
        """'Test Restaurant' and 'test restaurant' in same city → 1 kept."""
        r1 = _make_raw(name="Test Restaurant", votes=50)
        r2 = _make_raw(name="test restaurant", votes=200)
        df = pd.concat([r1, r2], ignore_index=True)
        result = DataPreprocessor().process(df)
        assert len(result) == 1
        assert result[0].votes == 200


# ── 8. Budget tier derivation ─────────────────────────────────────────────────

class TestBudgetTier:
    def test_low_tier(self):
        """cost_for_two <= 500 → 'low'"""
        result = DataPreprocessor().process(_make_raw(average_cost_for_two=500))
        assert result[0].budget_tier == "low"

    def test_medium_tier(self):
        """501 <= cost_for_two <= 1500 → 'medium'"""
        result = DataPreprocessor().process(_make_raw(average_cost_for_two=1000))
        assert result[0].budget_tier == "medium"

    def test_high_tier(self):
        """cost_for_two > 1500 → 'high'"""
        result = DataPreprocessor().process(_make_raw(average_cost_for_two=2000))
        assert result[0].budget_tier == "high"

    def test_boundary_low_threshold(self):
        """EC-P-10: Exactly at low threshold (500) → 'low'"""
        result = DataPreprocessor().process(_make_raw(average_cost_for_two=500))
        assert result[0].budget_tier == "low"

    def test_boundary_medium_threshold(self):
        """EC-P-10: Exactly at medium threshold (1500) → 'medium'"""
        result = DataPreprocessor().process(_make_raw(average_cost_for_two=1500))
        assert result[0].budget_tier == "medium"

    def test_just_above_medium_threshold(self):
        """EC-P-10: 1501 → 'high'"""
        result = DataPreprocessor().process(_make_raw(average_cost_for_two=1501))
        assert result[0].budget_tier == "high"

    def test_zero_cost_is_low(self):
        """EC-P-04: cost=0 (unknown) → 'low' tier."""
        result = DataPreprocessor().process(_make_raw(average_cost_for_two=0))
        assert result[0].budget_tier == "low"

    def test_custom_thresholds(self):
        """Preprocessor respects custom BUDGET_THRESHOLDS."""
        preprocessor = DataPreprocessor(budget_thresholds={"low": 100, "medium": 300})
        result = preprocessor.process(_make_raw(average_cost_for_two=150))
        assert result[0].budget_tier == "medium"


# ── 9. Full pipeline with frozen fixture ──────────────────────────────────────

class TestFullPipeline:
    def test_all_restaurants_have_id(self, preprocessed):
        """Every Restaurant must have a non-empty id."""
        assert all(r.id for r in preprocessed)

    def test_all_ids_are_unique(self, preprocessed):
        """No two restaurants should share the same id."""
        ids = [r.id for r in preprocessed]
        assert len(ids) == len(set(ids))

    def test_no_null_names(self, preprocessed):
        """Phase 1 AC: No restaurant should have a null or empty name."""
        assert all(r.name for r in preprocessed)

    def test_no_null_locations(self, preprocessed):
        """Phase 1 AC: No restaurant should have a null or empty location."""
        assert all(r.location for r in preprocessed)

    def test_cuisines_are_lists(self, preprocessed):
        """Phase 1 AC: cuisines field must be a list[str] for every restaurant."""
        for r in preprocessed:
            assert isinstance(r.cuisines, list)
            assert all(isinstance(c, str) for c in r.cuisines)

    def test_budget_tiers_are_valid(self, preprocessed):
        """Phase 1 AC: budget_tier must be one of the three valid values."""
        valid_tiers = {"low", "medium", "high"}
        for r in preprocessed:
            assert r.budget_tier in valid_tiers

    def test_ratings_in_range(self, preprocessed):
        """All ratings should be in [0.0, 5.0]."""
        for r in preprocessed:
            assert 0.0 <= r.rating <= 5.0

    def test_costs_non_negative(self, preprocessed):
        """All cost_for_two values should be >= 0."""
        for r in preprocessed:
            assert r.cost_for_two >= 0
