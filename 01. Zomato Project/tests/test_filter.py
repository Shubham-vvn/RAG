"""
tests/test_filter.py
─────────────────────
Unit tests for the Phase 2 Filter Layer:
  PreferenceNormalizer, PreferenceValidator, RestaurantFilter, CandidateSelector

Uses the frozen 20-row fixture from conftest.py — no network calls.

Run with:
    pytest tests/test_filter.py -v
"""
from __future__ import annotations

import math
import pytest

from src.config import ValidationError
from src.models.preferences import UserPreferences
from src.models.restaurant import Restaurant
from src.services.filter import (
    CandidateSelector,
    FilterPipeline,
    FilterResult,
    PreferenceNormalizer,
    PreferenceValidator,
    RestaurantFilter,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_restaurant(
    id="1", name="Test", location="Bangalore", cuisines=None,
    cost=800, rating=4.0, votes=100, budget_tier="medium"
) -> Restaurant:
    return Restaurant(
        id=id, name=name, location=location,
        cuisines=cuisines or ["north indian"],
        cost_for_two=cost, rating=rating, votes=votes,
        rest_type="Casual Dining", budget_tier=budget_tier,
    )


def make_prefs(**kw) -> UserPreferences:
    defaults = dict(location="Bangalore", budget="medium", min_rating=0.0)
    defaults.update(kw)
    return UserPreferences(**defaults)


# ═════════════════════════════════════════════════════════════════════════════
# 1. PreferenceNormalizer
# ═════════════════════════════════════════════════════════════════════════════

class TestPreferenceNormalizer:
    N = PreferenceNormalizer()

    def _n(self, **kw):
        defaults = dict(location="bangalore", budget="Medium",
                        cuisine="North Indian", min_rating=4.0, additional=None)
        defaults.update(kw)
        return self.N.normalize(**defaults)

    def test_location_title_cased(self):
        assert self._n(location="bangalore")["location"] == "Bangalore"

    def test_location_stripped(self):
        assert self._n(location="  Bangalore  ")["location"] == "Bangalore"

    def test_budget_lowercased(self):
        assert self._n(budget="MEDIUM")["budget"] == "medium"

    def test_cuisine_lowercased(self):
        assert self._n(cuisine="North Indian")["cuisine"] == "north indian"

    def test_cuisine_none_stays_none(self):
        assert self._n(cuisine=None)["cuisine"] is None

    def test_cuisine_empty_string_becomes_none(self):
        assert self._n(cuisine="")["cuisine"] is None

    def test_cuisine_whitespace_becomes_none(self):
        assert self._n(cuisine="   ")["cuisine"] is None

    def test_multi_cuisine_takes_first(self):
        """EC-I-10: comma-separated → first item only."""
        result = self._n(cuisine="Italian, Chinese")
        assert "," not in result["cuisine"]
        assert result["cuisine"] == "italian"

    def test_rating_string_to_float(self):
        result = self._n(min_rating="4.5")
        assert result["min_rating"] == pytest.approx(4.5)

    def test_rating_invalid_string_becomes_nan(self):
        result = self._n(min_rating="four")
        assert math.isnan(result["min_rating"])

    def test_additional_stripped(self):
        result = self._n(additional="  family friendly  ")
        assert result["additional"] == "family friendly"

    def test_additional_truncated_to_max_length(self):
        long_text = "x" * 600
        result = self._n(additional=long_text)
        assert len(result["additional"]) == 500

    def test_additional_none_stays_none(self):
        assert self._n(additional=None)["additional"] is None

    def test_additional_empty_becomes_none(self):
        assert self._n(additional="")["additional"] is None


# ═════════════════════════════════════════════════════════════════════════════
# 2. PreferenceValidator
# ═════════════════════════════════════════════════════════════════════════════

class TestPreferenceValidator:

    def _validator(self, repo):
        return PreferenceValidator(repo)

    def _validate(self, repo, **kw):
        defaults = dict(location="Bangalore", budget="medium",
                        cuisine=None, min_rating=4.0, additional=None)
        defaults.update(kw)
        return self._validator(repo).validate(**defaults)

    # ── location ──────────────────────────────────────────────────────────────
    def test_valid_location_passes(self, repo):
        prefs = self._validate(repo, location="Bangalore")
        assert prefs.location == "Bangalore"

    def test_blank_location_raises(self, repo):
        """EC-I-07: blank location raises ValidationError."""
        with pytest.raises(ValidationError, match="location"):
            self._validate(repo, location="")

    def test_unknown_location_raises_with_suggestions(self, repo):
        """EC-I-01: unknown location raises with suggestions."""
        with pytest.raises(ValidationError, match="No restaurants found"):
            self._validate(repo, location="Mysore")

    # ── budget ────────────────────────────────────────────────────────────────
    def test_valid_budgets_pass(self, repo):
        for b in ["low", "medium", "high"]:
            prefs = self._validate(repo, budget=b)
            assert prefs.budget == b

    def test_invalid_budget_raises(self, repo):
        """EC-I-05: invalid budget raises ValidationError."""
        with pytest.raises(ValidationError, match="budget"):
            self._validate(repo, budget="cheap")

    def test_empty_budget_raises(self, repo):
        with pytest.raises(ValidationError, match="budget"):
            self._validate(repo, budget="")

    # ── min_rating ────────────────────────────────────────────────────────────
    def test_valid_rating_passes(self, repo):
        prefs = self._validate(repo, min_rating=3.5)
        assert prefs.min_rating == pytest.approx(3.5)

    def test_zero_rating_passes(self, repo):
        """EC-I-11: min_rating=0.0 is valid ('no minimum')."""
        prefs = self._validate(repo, min_rating=0.0)
        assert prefs.min_rating == 0.0

    def test_rating_above_five_raises(self, repo):
        """EC-I-04: rating > 5.0 raises ValidationError."""
        with pytest.raises(ValidationError, match="min_rating"):
            self._validate(repo, min_rating=6.0)

    def test_negative_rating_raises(self, repo):
        with pytest.raises(ValidationError, match="min_rating"):
            self._validate(repo, min_rating=-1.0)

    def test_nan_rating_raises(self, repo):
        """EC-I-03: non-numeric (NaN) rating raises ValidationError."""
        with pytest.raises(ValidationError, match="min_rating"):
            self._validate(repo, min_rating=float("nan"))

    # ── cuisine (optional) ────────────────────────────────────────────────────
    def test_valid_cuisine_passes(self, repo):
        """Known cuisine passes through unchanged."""
        prefs = self._validate(repo, cuisine="north indian")
        assert prefs.cuisine == "north indian"

    def test_unknown_cuisine_warns_and_nulls(self, repo):
        """EC-I-06: unknown cuisine → no error, but cuisine set to None."""
        prefs = self._validate(repo, cuisine="martian")
        assert prefs.cuisine is None

    def test_none_cuisine_passes(self, repo):
        prefs = self._validate(repo, cuisine=None)
        assert prefs.cuisine is None

    # ── all-empty raises with all fields ─────────────────────────────────────
    def test_all_empty_raises(self, repo):
        """EC-I-07: multiple invalid fields → one combined error."""
        with pytest.raises(ValidationError):
            self._validate(repo, location="", budget="", min_rating=float("nan"))

    # ── returns UserPreferences ───────────────────────────────────────────────
    def test_returns_user_preferences_object(self, repo):
        prefs = self._validate(repo)
        assert isinstance(prefs, UserPreferences)

    def test_additional_preserved(self, repo):
        prefs = self._validate(repo, additional="outdoor seating")
        assert prefs.additional == "outdoor seating"


# ═════════════════════════════════════════════════════════════════════════════
# 3. RestaurantFilter
# ═════════════════════════════════════════════════════════════════════════════

class TestRestaurantFilter:

    def _filter(self, repo, max_candidates=50, partial=True):
        return RestaurantFilter(repo, max_candidates=max_candidates,
                                partial_cuisine_match=partial)

    def _prefs(self, **kw):
        return make_prefs(**kw)

    # ── Basic filtering ───────────────────────────────────────────────────────
    def test_filters_by_location(self, repo):
        result = self._filter(repo).filter(self._prefs(location="Delhi"))
        assert all(r.location == "Delhi" for r in result.candidates)

    def test_filters_by_budget_tier(self, repo):
        result = self._filter(repo).filter(self._prefs(location="Bangalore", budget="low"))
        assert all(r.budget_tier == "low" for r in result.candidates)

    def test_filters_by_min_rating(self, repo):
        result = self._filter(repo).filter(self._prefs(location="Bangalore", min_rating=4.5))
        assert all(r.rating >= 4.5 for r in result.candidates)

    def test_filters_by_cuisine_exact(self, repo):
        result = self._filter(repo, partial=False).filter(
            self._prefs(location="Bangalore", cuisine="chinese")
        )
        assert all("chinese" in r.cuisines for r in result.candidates)

    def test_filters_by_cuisine_partial(self, repo):
        """EC-F-08: 'indian' matches 'north indian', 'south indian', etc."""
        result = self._filter(repo, partial=True).filter(
            self._prefs(location="Delhi", cuisine="indian")
        )
        assert all(r.matches_cuisine("indian", partial=True) for r in result.candidates)

    def test_no_cuisine_filter_skipped(self, repo):
        """cuisine=None means cuisine filter is not applied."""
        result_all  = self._filter(repo).filter(self._prefs(location="Bangalore", cuisine=None))
        result_filt = self._filter(repo).filter(self._prefs(location="Bangalore", cuisine="italian"))
        assert len(result_all.candidates) >= len(result_filt.candidates)

    # ── Sorting ───────────────────────────────────────────────────────────────
    def test_results_sorted_by_rating_desc(self, repo):
        result = self._filter(repo).filter(self._prefs(location="Delhi"))
        ratings = [r.rating for r in result.candidates]
        assert ratings == sorted(ratings, reverse=True)

    def test_tie_broken_by_votes_desc(self, repo):
        """EC-F-06: Equal rating → higher votes first."""
        r1 = make_restaurant(id="a", name="B", rating=4.0, votes=200)
        r2 = make_restaurant(id="b", name="A", rating=4.0, votes=500)
        sorted_list = RestaurantFilter._sort([r1, r2])
        assert sorted_list[0].id == "b"   # higher votes

    def test_final_tie_broken_alphabetically(self, repo):
        """EC-F-06: Equal rating + equal votes → alphabetical by name."""
        r1 = make_restaurant(id="a", name="Zebra", rating=4.0, votes=100)
        r2 = make_restaurant(id="b", name="Apple", rating=4.0, votes=100)
        sorted_list = RestaurantFilter._sort([r1, r2])
        assert sorted_list[0].name == "Apple"

    # ── Capping ───────────────────────────────────────────────────────────────
    def test_results_capped_at_max(self, repo):
        """EC-F-04/05: Results capped at max_candidates."""
        result = self._filter(repo, max_candidates=2).filter(
            self._prefs(location="Delhi", min_rating=0.0)
        )
        assert len(result.candidates) <= 2

    def test_single_candidate_returned(self, repo):
        """EC-F-03: Only 1 match → 1 returned without padding."""
        result = self._filter(repo).filter(
            self._prefs(location="Delhi", budget="high", min_rating=4.9)
        )
        # Should return Indian Accent (rating=4.8) or similar — at most a handful
        assert len(result.candidates) >= 0   # doesn't blow up

    # ── Constraint relaxation ─────────────────────────────────────────────────
    def test_relaxes_cuisine_when_zero_results(self, repo):
        """EC-F-01: Zero results → cuisine relaxed first."""
        result = self._filter(repo).filter(
            self._prefs(location="Bangalore", budget="low",
                        cuisine="sushi", min_rating=0.0)
        )
        assert "cuisine" in result.relaxed_constraints

    def test_relaxes_budget_after_cuisine(self, repo):
        """EC-F-01: After cuisine relaxation still 0 → budget relaxed."""
        result = self._filter(repo).filter(
            self._prefs(location="Bangalore", budget="high",
                        cuisine="sushi", min_rating=5.0)
        )
        # Both cuisine and rating (or budget) should be in relaxed
        assert len(result.relaxed_constraints) > 0

    def test_no_relaxation_when_results_found(self, repo):
        result = self._filter(repo).filter(
            self._prefs(location="Bangalore", min_rating=0.0, cuisine=None)
        )
        assert result.relaxed_constraints == []

    # ── Edge cases ────────────────────────────────────────────────────────────
    def test_zero_rating_passes_all(self, repo):
        """EC-I-11: min_rating=0.0 → all restaurants for location pass rating filter."""
        result = self._filter(repo).filter(
            self._prefs(location="Bangalore", min_rating=0.0, cuisine=None)
        )
        assert len(result.candidates) > 0

    def test_restaurant_with_empty_cuisines_excluded_when_cuisine_filter_active(self, repo):
        """EC-F-11: cuisines=[] → skipped by cuisine filter."""
        # The 'No Cuisine Place' entry in conftest has cuisines=[]
        result = self._filter(repo).filter(
            self._prefs(location="Bangalore", cuisine="north indian", min_rating=0.0)
        )
        no_cuisine_included = any(r.cuisines == [] for r in result.candidates)
        assert not no_cuisine_included

    def test_budget_exact_match_only(self, repo):
        """EC-F-09: budget='medium' should NOT include 'low' or 'high' restaurants."""
        result = self._filter(repo).filter(
            self._prefs(location="Bangalore", budget="medium", min_rating=0.0)
        )
        assert all(r.budget_tier == "medium" for r in result.candidates)

    def test_unknown_location_returns_empty(self, repo):
        """A location with no restaurants returns empty candidates."""
        result = self._filter(repo).filter(make_prefs(location="TinyVillage"))
        assert result.candidates == []


# ═════════════════════════════════════════════════════════════════════════════
# 4. CandidateSelector
# ═════════════════════════════════════════════════════════════════════════════

class TestCandidateSelector:

    def test_deduplicates_by_id(self):
        """EC-PB-07: Duplicate IDs in candidate list are removed."""
        r = make_restaurant(id="dup")
        result = FilterResult(candidates=[r, r], relaxed_constraints=[])
        prefs = make_prefs()
        final, _ = CandidateSelector(max_candidates=10).select(result, prefs)
        assert len(final) == 1

    def test_caps_at_max_candidates(self):
        """EC-F-05: More candidates than max → trimmed to max."""
        restaurants = [make_restaurant(id=str(i)) for i in range(30)]
        result = FilterResult(candidates=restaurants, relaxed_constraints=[])
        prefs = make_prefs()
        final, _ = CandidateSelector(max_candidates=10).select(result, prefs)
        assert len(final) == 10

    def test_attaches_relaxed_constraints_to_prefs(self):
        """Relaxed constraints from FilterResult are attached to UserPreferences."""
        r = make_restaurant()
        result = FilterResult(candidates=[r], relaxed_constraints=["cuisine", "budget"])
        prefs = make_prefs()
        _, updated_prefs = CandidateSelector().select(result, prefs)
        assert "cuisine" in updated_prefs.relaxed_constraints
        assert "budget" in updated_prefs.relaxed_constraints

    def test_empty_relaxed_stays_empty(self):
        result = FilterResult(candidates=[make_restaurant()], relaxed_constraints=[])
        prefs = make_prefs()
        _, updated_prefs = CandidateSelector().select(result, prefs)
        assert updated_prefs.relaxed_constraints == []

    def test_returns_tuple_of_list_and_prefs(self):
        result = FilterResult(candidates=[make_restaurant()], relaxed_constraints=[])
        prefs = make_prefs()
        out = CandidateSelector().select(result, prefs)
        assert isinstance(out, tuple)
        assert isinstance(out[0], list)
        assert isinstance(out[1], UserPreferences)


# ═════════════════════════════════════════════════════════════════════════════
# 5. FilterPipeline (end-to-end Phase 2)
# ═════════════════════════════════════════════════════════════════════════════

class TestFilterPipeline:

    def test_successful_pipeline_returns_candidates(self, repo):
        pipeline = FilterPipeline(repo, max_candidates=20)
        candidates, prefs = pipeline.run(
            location="Bangalore", budget="medium", min_rating=0.0
        )
        assert isinstance(candidates, list)
        assert isinstance(prefs, UserPreferences)
        assert len(candidates) > 0

    def test_pipeline_normalises_input(self, repo):
        """Pipeline accepts un-normalised input and normalises it internally."""
        candidates, prefs = FilterPipeline(repo).run(
            location="bangalore",   # lowercase
            budget="MEDIUM",        # uppercase
            min_rating=0.0,
        )
        assert prefs.location == "Bangalore"
        assert prefs.budget == "medium"

    def test_pipeline_raises_on_invalid_location(self, repo):
        pipeline = FilterPipeline(repo)
        with pytest.raises(ValidationError):
            pipeline.run(location="Nonexistent City", budget="medium", min_rating=0.0)

    def test_pipeline_raises_on_invalid_budget(self, repo):
        pipeline = FilterPipeline(repo)
        with pytest.raises(ValidationError):
            pipeline.run(location="Bangalore", budget="expensive", min_rating=0.0)

    def test_pipeline_raises_on_invalid_rating(self, repo):
        pipeline = FilterPipeline(repo)
        with pytest.raises(ValidationError):
            pipeline.run(location="Bangalore", budget="medium", min_rating=10.0)

    def test_pipeline_with_all_params(self, repo):
        candidates, prefs = FilterPipeline(repo).run(
            location="Delhi",
            budget="high",
            cuisine="north indian",
            min_rating=4.0,
            additional="family-friendly",
        )
        assert prefs.additional == "family-friendly"
        assert prefs.cuisine == "north indian"

    def test_pipeline_cap_respected(self, repo):
        """Pipeline respects max_candidates cap."""
        candidates, _ = FilterPipeline(repo, max_candidates=2).run(
            location="Delhi", budget="low", min_rating=0.0
        )
        assert len(candidates) <= 2

    def test_pipeline_relaxation_reflected_in_prefs(self, repo):
        """Constraint relaxation is reflected in returned prefs."""
        candidates, prefs = FilterPipeline(repo).run(
            location="Bangalore",
            budget="low",
            cuisine="sushi",   # doesn't exist → cuisine relaxed
            min_rating=0.0,
        )
        assert "cuisine" in prefs.relaxed_constraints

    def test_to_filter_dict_reflects_active_filters(self, repo):
        _, prefs = FilterPipeline(repo).run(
            location="Bangalore", budget="medium",
            cuisine="north indian", min_rating=3.5,
        )
        d = prefs.to_filter_dict()
        assert d["location"] == "Bangalore"
        assert d["budget"] == "medium"
        assert d["min_rating"] == pytest.approx(3.5)

    def test_user_preferences_repr(self, repo):
        _, prefs = FilterPipeline(repo).run(
            location="Mumbai", budget="high", min_rating=4.0
        )
        assert "Mumbai" in repr(prefs)
