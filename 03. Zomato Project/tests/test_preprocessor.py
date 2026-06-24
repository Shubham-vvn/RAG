"""
Tests for DataPreprocessor — cuisine parsing, numeric coercion, null handling, budget tier.

Implementation: Phase 1 (alongside src/data/preprocessor.py)
"""

import pandas as pd
import pytest
from src.data.preprocessor import DataPreprocessor
from src.models.restaurant import BudgetTier


def test_parse_rating():
    preprocessor = DataPreprocessor()
    assert preprocessor._parse_rating("4.1/5") == 4.1
    assert preprocessor._parse_rating(" 4.2 / 5 ") == 4.2
    assert preprocessor._parse_rating(4.5) == 4.5
    assert preprocessor._parse_rating("NEW") is None
    assert preprocessor._parse_rating("-") is None
    assert preprocessor._parse_rating(None) is None
    assert preprocessor._parse_rating("6.0/5") is None  # Out of range rating


def test_parse_cost():
    preprocessor = DataPreprocessor()
    assert preprocessor._parse_cost("800") == 800
    assert preprocessor._parse_cost("1,200") == 1200
    assert preprocessor._parse_cost(600) == 600
    assert preprocessor._parse_cost("abc") is None
    assert preprocessor._parse_cost(None) is None
    assert preprocessor._parse_cost("-100") is None  # Invalid negative cost


def test_parse_cuisines():
    preprocessor = DataPreprocessor()
    assert preprocessor._parse_cuisines("Italian, Chinese") == ["Italian", "Chinese"]
    assert preprocessor._parse_cuisines("North Indian, Mughlai, ") == ["North Indian", "Mughlai"]
    assert preprocessor._parse_cuisines("") == ["Unknown"]
    assert preprocessor._parse_cuisines(None) == ["Unknown"]


def test_normalize_location():
    preprocessor = DataPreprocessor()
    assert preprocessor._normalize_location("bengaluru") == "Bangalore"
    assert preprocessor._normalize_location("Bengaluru") == "Bangalore"
    assert preprocessor._normalize_location("bombay") == "Mumbai"
    assert preprocessor._normalize_location("Delhi") == "Delhi"
    assert preprocessor._normalize_location("ncr") == "New Delhi"
    assert preprocessor._normalize_location(None) == "Unknown"


def test_derive_budget_tier():
    preprocessor = DataPreprocessor()
    assert preprocessor._derive_budget_tier(300) == BudgetTier.LOW.value
    assert preprocessor._derive_budget_tier(500) == BudgetTier.LOW.value
    assert preprocessor._derive_budget_tier(1000) == BudgetTier.MEDIUM.value
    assert preprocessor._derive_budget_tier(1500) == BudgetTier.MEDIUM.value
    assert preprocessor._derive_budget_tier(2000) == BudgetTier.HIGH.value


def test_preprocess_pipeline():
    preprocessor = DataPreprocessor()
    raw_data = {
        "name": ["Trattoria", "Spice Palace", "", None, "Quick Bites"],
        "rate": ["4.5/5", "3.9", "4.0", "4.2/5", "NEW"],
        "approx_cost(for two people)": ["1,200", "400", "800", "150", "300"],
        "location": ["Bengaluru", "Delhi", "Mumbai", None, "Ncr"],
        "cuisines": ["Italian, Continental", "North Indian", "Street Food", "Chinese", "Fast Food"],
        "votes": [120, "350", 0, None, 10],
        "rest_type": ["Casual Dining", "Casual Dining", "Quick Bites", "Cafe", "Quick Bites"],
    }
    raw_df = pd.DataFrame(raw_data)
    clean_df = preprocessor.preprocess(raw_df)

    # Valid rows:
    # 0: "Trattoria", rating 4.5, cost 1200, location Bangalore, cuisines ["Italian", "Continental"], votes 120, budget_tier medium, id "0"
    # 1: "Spice Palace", rating 3.9, cost 400, location Delhi, cuisines ["North Indian"], votes 350, budget_tier low, id "1"
    # 4: "Quick Bites" has rating "NEW" which parses to None, so it gets dropped.
    # Row 2 has empty name, row 3 has null location, so they are dropped.
    assert len(clean_df) == 2

    # Check mapping & types
    assert clean_df.iloc[0]["name"] == "Trattoria"
    assert clean_df.iloc[0]["rating"] == 4.5
    assert clean_df.iloc[0]["cost_for_two"] == 1200
    assert clean_df.iloc[0]["location"] == "Bangalore"
    assert clean_df.iloc[0]["cuisines"] == ["Italian", "Continental"]
    assert clean_df.iloc[0]["votes"] == 120
    assert clean_df.iloc[0]["budget_tier"] == BudgetTier.MEDIUM.value
    assert clean_df.iloc[0]["id"] == "0"

    assert clean_df.iloc[1]["name"] == "Spice Palace"
    assert clean_df.iloc[1]["rating"] == 3.9
    assert clean_df.iloc[1]["cost_for_two"] == 400
    assert clean_df.iloc[1]["location"] == "Delhi"
    assert clean_df.iloc[1]["votes"] == 350
    assert clean_df.iloc[1]["budget_tier"] == BudgetTier.LOW.value
    assert clean_df.iloc[1]["id"] == "1"


def test_preprocess_deduplication():
    preprocessor = DataPreprocessor()
    raw_data = {
        "name": ["Trattoria", "Trattoria", "Trattoria"],
        "rate": ["4.1/5", "4.5/5", "4.1/5"],
        "approx_cost(for two people)": ["1,200", "1,200", "1,200"],
        "location": ["Bengaluru", "Bengaluru", "Bengaluru"],
        "cuisines": ["Italian", "Italian", "Italian"],
        "votes": [100, 200, 300],
        "rest_type": ["Casual Dining", "Casual Dining", "Casual Dining"],
    }
    raw_df = pd.DataFrame(raw_data)
    clean_df = preprocessor.preprocess(raw_df)

    # Should deduplicate down to a single row: the one with highest rating and votes (index 1: rating 4.5, votes 200)
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["rating"] == 4.5
    assert clean_df.iloc[0]["votes"] == 200


