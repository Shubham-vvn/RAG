"""
Tests for RestaurantFilter — filter pipeline, sort order, constraint relaxation.

Implementation: Phase 2 (alongside src/services/filter.py)
"""

import pytest
from src.services.filter import (
    suggest_locations,
    validate_preferences,
    ValidationError,
    RestaurantFilter,
    CandidateSelector,
)
from src.models.preferences import UserPreferences
from src.data.repository import RestaurantRepository


def test_suggest_locations():
    available = ["Bangalore", "Delhi", "Mumbai", "Kolkata", "Chennai"]
    # Prefix match
    assert suggest_locations("bang", available) == ["Bangalore"]
    assert suggest_locations("del", available) == ["Delhi"]
    # Substring match
    assert suggest_locations("umb", available) == ["Mumbai"]
    # Empty query
    assert suggest_locations("", available) == []
    # No match
    assert suggest_locations("xyz", available) == []


def test_validate_preferences(sample_restaurants):
    repo = RestaurantRepository(sample_restaurants)

    # Valid preferences
    raw_pref = {
        "location": "Bangalore",
        "budget": "medium",
        "min_rating": "4.0",
        "cuisine": "Italian",
    }
    pref = validate_preferences(raw_pref, repo)
    assert pref.location == "Bangalore"
    assert pref.budget == "medium"
    assert pref.min_rating == 4.0
    assert pref.cuisine == "Italian"

    # Missing location with suggestions
    raw_pref_bad_loc = {
        "location": "bangalor",
        "budget": "medium",
        "min_rating": 4.0,
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_preferences(raw_pref_bad_loc, repo)
    assert "Did you mean: Bangalore?" in str(exc_info.value)

    # Invalid cuisine
    raw_pref_bad_cuisine = {
        "location": "Bangalore",
        "budget": "medium",
        "min_rating": 4.0,
        "cuisine": "Martian Food",
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_preferences(raw_pref_bad_cuisine, repo)
    assert "Cuisine 'Martian Food' not found" in str(exc_info.value)

    # General validation bounds (e.g. rating > 5.0)
    raw_pref_bad_rating = {
        "location": "Bangalore",
        "budget": "medium",
        "min_rating": 6.0,
    }
    with pytest.raises(ValidationError) as exc_info:
        validate_preferences(raw_pref_bad_rating, repo)
    assert "Min rating must be between 0.0 and 5.0" in str(exc_info.value)


def test_restaurant_filter_basic(sample_restaurants):
    filter_service = RestaurantFilter(max_candidates=5)

    # Bangalore, medium budget, Italian, rating >= 4.0
    pref = UserPreferences(
        location="Bangalore",
        budget="medium",
        min_rating=4.0,
        cuisine="Italian",
    )
    candidates, warnings = filter_service.filter(sample_restaurants, pref)
    assert len(warnings) == 0
    # Expected matches: Trattoria Milano (rating 4.5, medium), La Piazza (rating 4.3, medium), Olive Bistro (rating 4.0, medium)
    assert len(candidates) == 3
    assert candidates[0].name == "Trattoria Milano"  # 4.5 rating
    assert candidates[1].name == "La Piazza"          # 4.3 rating
    assert candidates[2].name == "Olive Bistro"       # 4.0 rating


def test_restaurant_filter_sorting_and_votes(sample_restaurants):
    # Test deterministic tie-breaking on rating & votes
    selector = CandidateSelector(max_candidates=5)
    # Filter only Bangalore restaurants to see sorting
    bangalore_recs = [r for r in sample_restaurants if r.location == "Bangalore"]
    sorted_recs = selector.select(bangalore_recs)

    # Check that they are sorted by rating desc, then votes desc
    # Trattoria Milano: 4.5 rating, 892 votes
    # Royal Biryani House: 4.4 rating, 1800 votes
    # La Piazza: 4.3 rating, 723 votes
    # Sushi Samurai: 4.3 rating, 280 votes
    # Dragon Palace: 4.2 rating, 654 votes
    # Let's verify the ranks:
    assert sorted_recs[0].name == "The French Table"    # rating 4.6
    assert sorted_recs[1].name == "Trattoria Milano"    # rating 4.5
    assert sorted_recs[2].name == "Royal Biryani House" # rating 4.4
    assert sorted_recs[3].name == "La Piazza"          # rating 4.3 (723 votes)
    assert sorted_recs[4].name == "Sushi Samurai"       # rating 4.3 (280 votes)


def test_constraint_relaxation_cuisine(sample_restaurants):
    filter_service = RestaurantFilter(max_candidates=5)

    # Bangalore, medium budget, 4.0 rating, but Nonexistent Cuisine
    pref = UserPreferences(
        location="Bangalore",
        budget="medium",
        min_rating=4.0,
        cuisine="Martian",
    )
    candidates, warnings = filter_service.filter(sample_restaurants, pref)

    # It should fall back to showing all cuisines in Bangalore/medium/4.0
    assert len(warnings) == 1
    assert "No 'Martian' restaurants found. Showing all cuisines." in warnings[0]
    # Should get Italian/Chinese restaurants matching Bangalore/medium/4.0
    assert len(candidates) == 5  # capped at 5
    assert any(r.name == "Trattoria Milano" for r in candidates)


def test_constraint_relaxation_budget(sample_restaurants):
    filter_service = RestaurantFilter(max_candidates=5)

    # Bangalore, high budget, rating >= 4.8.
    # No high budget restaurant in Bangalore has rating >= 4.8.
    # So rating filter + budget filter produces 0.
    pref = UserPreferences(
        location="Bangalore",
        budget="high",
        min_rating=4.8,
        cuisine=None,
    )
    candidates, warnings = filter_service.filter(sample_restaurants, pref)

    # It should relax the budget first
    assert "No restaurants found in your budget. Showing all budget ranges." in warnings
    # Even after relaxing budget, is there any restaurant in Bangalore with rating >= 4.8?
    # No! Bangalore has max rating 4.6 (The French Table).
    # So it should also trigger the rating relaxation!
    assert any("Lowered minimum rating" in w for w in warnings)
    assert len(candidates) > 0

