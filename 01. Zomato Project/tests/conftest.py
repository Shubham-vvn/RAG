"""
tests/conftest.py
─────────────────
Shared pytest fixtures used across all test modules.

Fixtures
--------
raw_df              — A frozen 20-row DataFrame mimicking HF dataset columns
preprocessed        — Cleaned list[Restaurant] from raw_df
repo                — RestaurantRepository built from preprocessed
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocessor import DataPreprocessor
from src.data.repository import RestaurantRepository
from src.models.restaurant import Restaurant


# ── Raw dataset fixture (20 deterministic rows) ───────────────────────────────

RAW_DATA = [
    # (name, location, cuisines, avg_cost, rating, votes, rest_type)
    ("Spice Garden",       "Bangalore",  "North Indian, Chinese",  800,  4.5, 1200, "Casual Dining"),
    ("Pizza Paradise",     "Bangalore",  "Italian, Continental",   1200, 4.2,  800, "Casual Dining"),
    ("Wok Express",        "Bangalore",  "Chinese",                500,  3.8,  400, "Quick Bites"),
    ("Biryani Blues",      "Bangalore",  "Biryani, North Indian",  600,  4.0,  950, "Casual Dining"),
    ("The French Cafe",    "Bangalore",  "Continental, Italian",   1600, 4.7,  300, "Cafe"),
    ("Haldiram's",         "Delhi",      "North Indian, Sweets",   350,  3.5,  500, "Quick Bites"),
    ("Bukhara",            "Delhi",      "North Indian",           3500, 4.9, 2500, "Fine Dining"),
    ("Momo Land",          "Delhi",      "Tibetan, Chinese",       250,  3.2,  150, "Quick Bites"),
    ("Indian Accent",      "Delhi",      "Modern Indian",          4500, 4.8, 1800, "Fine Dining"),
    ("Karim's",            "Delhi",      "Mughlai, North Indian",  600,  4.3, 3000, "Casual Dining"),
    ("Trishna",            "Mumbai",     "Seafood, Continental",   2000, 4.6,  900, "Casual Dining"),
    ("Vada Pav Corner",    "Mumbai",     "Street Food",            100,  3.9,  700, "Quick Bites"),
    ("Britannia & Co",     "Mumbai",     "Parsi, Continental",     800,  4.4,  600, "Casual Dining"),
    ("Leopold Cafe",       "Mumbai",     "Continental, Chinese",   1200, 3.7,  400, "Casual Dining"),
    ("Bademiya",           "Mumbai",     "Mughlai, North Indian",  400,  4.1,  800, "Quick Bites"),
    ("Pind Balluchi",      "Chennai",    "North Indian, Mughlai",  900,  3.8,  300, "Casual Dining"),
    ("Murugan Idli Shop",  "Chennai",    "South Indian",           200,  4.6, 2000, "Quick Bites"),
    ("The Flying Elephant","Chennai",    "Continental, Italian",   2500, 4.5,  450, "Bar"),
    ("Sangeetha Veg",      "Chennai",    "South Indian",           300,  4.0, 1100, "Quick Bites"),
    # Edge-case rows
    ("No Cuisine Place",   "Bangalore",  None,                     700,  4.0,  100, "Casual Dining"),  # null cuisine
]


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """
    A frozen 20-row DataFrame that mirrors the HF dataset column schema.
    Used to test DataPreprocessor without any network calls.
    """
    rows = [
        {
            "name":                 name,
            "location":             loc,
            "cuisines":             cuisines,
            "average_cost_for_two": cost,
            "aggregate_rating":     rating,
            "votes":                votes,
            "rest_type":            rest_type,
        }
        for name, loc, cuisines, cost, rating, votes, rest_type in RAW_DATA
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def preprocessed(raw_df: pd.DataFrame) -> list[Restaurant]:
    """Cleaned list[Restaurant] derived from the frozen raw_df."""
    return DataPreprocessor().process(raw_df)


@pytest.fixture(scope="session")
def repo(preprocessed: list[Restaurant]) -> RestaurantRepository:
    """RestaurantRepository seeded with the frozen test dataset."""
    return RestaurantRepository(preprocessed)
