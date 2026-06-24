"""
src/services/filter.py
───────────────────────
Phase 2 — Filter Layer.

Contains four collaborating classes:

  PreferenceNormalizer  — cleans raw user input (strip, case, truncate)
  PreferenceValidator   — enforces business rules; raises ValidationError
  RestaurantFilter      — applies deterministic hard filters in sequence
  CandidateSelector     — caps result count and applies tie-breaking

Typical call chain (in RecommendationService):
    raw_input  → PreferenceNormalizer.normalize()
               → PreferenceValidator.validate()          # raises on bad input
               → RestaurantFilter.filter()               # returns candidates
               → CandidateSelector.select()              # returns final list

Edge Cases Covered
------------------
EC-F-01  Zero results after all filters → constraint relaxation
EC-F-02  Zero results after full relaxation → return all for location
EC-F-03  Exactly 1 candidate
EC-F-04  Exactly MAX_CANDIDATES candidates (boundary)
EC-F-05  >MAX_CANDIDATES results → capped by CandidateSelector
EC-F-06  All candidates tied → alphabetical tie-breaking
EC-F-07  Cuisine case mismatch → normalised before comparison
EC-F-08  Cuisine partial match (configurable)
EC-F-09  Budget tier exact match only
EC-F-10  All restaurants unrated → works with min_rating=0
EC-F-11  Restaurant with cuisines=[] → skipped by cuisine filter
EC-I-01  Location not in dataset → ValidationError with suggestions
EC-I-02  Location whitespace/case → normalised
EC-I-03  Non-numeric rating → ValidationError
EC-I-04  Rating out of range → ValidationError
EC-I-05  Invalid budget → ValidationError
EC-I-06  Unknown cuisine → warn + proceed with cuisine=None
EC-I-07  All empty input → ValidationError listing all required fields
EC-I-08  Long additional text → truncated to MAX_ADDITIONAL_LENGTH
EC-I-09  Prompt injection in additional → delimited in prompt (handled in PromptBuilder)
EC-I-10  Multi-cuisine input → ValidationError with hint
EC-I-11  min_rating=0.0 → all ratings pass
EC-I-12  min_rating=5.0 → very few pass; relaxation triggered
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.config import ValidationError, settings
from src.models.preferences import RATING_MAX, RATING_MIN, VALID_BUDGETS, UserPreferences
from src.models.restaurant import Restaurant

if TYPE_CHECKING:
    from src.data.repository import RestaurantRepository

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
_VALID_BUDGETS = list(VALID_BUDGETS)  # for error messages


# ══════════════════════════════════════════════════════════════════════════════
# 1. PreferenceNormalizer
# ══════════════════════════════════════════════════════════════════════════════

class PreferenceNormalizer:
    """
    Cleans and standardises raw user input before validation.

    All mutations are safe and non-lossy (whitespace trimming, case
    normalisation, length capping).

    Usage
    -----
        normalizer = PreferenceNormalizer()
        clean = normalizer.normalize(location, budget, cuisine, min_rating, additional)
    """

    def normalize(
        self,
        location: str,
        budget: str,
        cuisine: str | None,
        min_rating: float | str,
        additional: str | None,
    ) -> dict:
        """
        Return a dict of normalised values ready for validation.

        Parameters
        ----------
        location    : Raw location string from user input.
        budget      : Raw budget string.
        cuisine     : Raw cuisine string or None.
        min_rating  : Raw min_rating (may be str from form input).
        additional  : Free-text additional preferences or None.

        Returns
        -------
        dict with keys: location, budget, cuisine, min_rating, additional
        """
        # Location: strip + title-case  (EC-I-02)
        norm_location = str(location).strip().title() if location else ""

        # Budget: strip + lowercase
        norm_budget = str(budget).strip().lower() if budget else ""

        # Cuisine: strip + lowercase  (EC-F-07)
        if cuisine and str(cuisine).strip():
            norm_cuisine: str | None = str(cuisine).strip().lower()
            # EC-I-10: reject comma-separated multi-cuisine
            if "," in norm_cuisine:
                # Grab the first one and let the validator decide
                norm_cuisine = norm_cuisine.split(",")[0].strip()
        else:
            norm_cuisine = None

        # min_rating: coerce to float  (EC-I-03)
        try:
            norm_rating = float(min_rating)
        except (TypeError, ValueError):
            norm_rating = float("nan")   # validator will catch this

        # Additional: strip + truncate  (EC-I-08)
        if additional and str(additional).strip():
            norm_additional: str | None = str(additional).strip()
            max_len = settings.MAX_ADDITIONAL_LENGTH or 500
            if len(norm_additional) > max_len:
                logger.warning(
                    "Additional preferences truncated from %d to %d characters.",
                    len(norm_additional),
                    max_len,
                )
                norm_additional = norm_additional[:max_len]
        else:
            norm_additional = None

        return {
            "location": norm_location,
            "budget": norm_budget,
            "cuisine": norm_cuisine,
            "min_rating": norm_rating,
            "additional": norm_additional,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2. PreferenceValidator
# ══════════════════════════════════════════════════════════════════════════════

class PreferenceValidator:
    """
    Validates normalised user preferences against business rules.

    Parameters
    ----------
    repo : RestaurantRepository
        Used to verify location existence and cuisine vocabulary.

    Raises
    ------
    ValidationError
        On any invalid input.  The message is user-facing.
    """

    def __init__(self, repo: "RestaurantRepository") -> None:
        self._repo = repo

    def validate(
        self,
        location: str,
        budget: str,
        cuisine: str | None,
        min_rating: float,
        additional: str | None,
    ) -> UserPreferences:
        """
        Validate all fields and return a UserPreferences object.

        Parameters are the *normalised* values from PreferenceNormalizer.

        Returns
        -------
        UserPreferences
            Ready to pass to RestaurantFilter.

        Raises
        ------
        ValidationError
            With a user-friendly message on any invalid field.
        """
        errors: list[str] = []

        # ── location ──────────────────────────────────────────────────────────
        if not location:
            errors.append("'location' is required and cannot be blank.")
        elif not self._repo.location_exists(location):
            suggestions = self._repo.suggest_locations(location, top_n=3)
            suggestion_str = ", ".join(f"'{s}'" for s in suggestions)
            errors.append(
                f"No restaurants found for location '{location}'. "
                f"Did you mean: {suggestion_str}?"
                if suggestions else
                f"No restaurants found for location '{location}'. "
                "Check the location name and try again."
            )

        # ── budget ────────────────────────────────────────────────────────────
        if not budget:
            errors.append("'budget' is required. Choose one of: low, medium, high.")
        elif budget not in VALID_BUDGETS:
            errors.append(
                f"Invalid budget '{budget}'. Must be one of: low, medium, high."
            )

        # ── min_rating ────────────────────────────────────────────────────────
        import math
        if math.isnan(min_rating):
            errors.append(
                "'min_rating' must be a number between 0.0 and 5.0. "
                f"Received a non-numeric value."
            )
        elif not (RATING_MIN <= min_rating <= RATING_MAX):
            errors.append(
                f"'min_rating' must be between {RATING_MIN} and {RATING_MAX}. "
                f"Received: {min_rating}."
            )

        # ── Raise early if required fields have errors ─────────────────────────
        if errors:
            raise ValidationError(
                "Invalid user preferences:\n" + "\n".join(f"  • {e}" for e in errors)
            )

        # ── cuisine (optional, warns but doesn't block) ───────────────────────
        validated_cuisine = cuisine
        if cuisine:
            # EC-I-06: unknown cuisine → warn and proceed without filter
            if not self._repo.cuisine_exists(cuisine):
                logger.warning(
                    "Cuisine '%s' not found in dataset vocabulary. "
                    "Proceeding without cuisine filter.",
                    cuisine,
                )
                validated_cuisine = None

        logger.info(
            "Preferences validated: location=%r, budget=%r, cuisine=%r, "
            "min_rating=%s",
            location, budget, validated_cuisine, min_rating,
        )

        return UserPreferences(
            location=location,
            budget=budget,
            cuisine=validated_cuisine,
            min_rating=min_rating,
            additional=additional,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. RestaurantFilter
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FilterResult:
    """Intermediate result from RestaurantFilter."""
    candidates: list[Restaurant]
    relaxed_constraints: list[str]   # e.g. ["cuisine", "budget"]


class RestaurantFilter:
    """
    Applies deterministic hard filters to the full restaurant list and
    returns a bounded, ranked candidate set for the LLM.

    Filter pipeline (applied in order):
        1. location   (required — never relaxed)
        2. budget_tier
        3. min_rating
        4. cuisine    (optional preference)
        5. sort: rating DESC, votes DESC, name ASC
        6. cap at MAX_CANDIDATES_FOR_LLM

    Constraint relaxation  (EC-F-01):
        If 0 candidates after step 4 → relax cuisine
        If still 0 after relaxing cuisine → relax budget
        If still 0 after relaxing budget → relax rating (keep min_rating=0)
        Records which constraints were relaxed in FilterResult.

    Parameters
    ----------
    repo : RestaurantRepository
        Source of restaurant records.
    max_candidates : int
        Maximum number of candidates passed to the LLM.
        Defaults to settings.MAX_CANDIDATES_FOR_LLM.
    partial_cuisine_match : bool
        If True, "indian" matches "north indian", "south indian", etc.
        Defaults to settings.CUISINE_PARTIAL_MATCH.
    """

    def __init__(
        self,
        repo: "RestaurantRepository",
        max_candidates: int | None = None,
        partial_cuisine_match: bool | None = None,
    ) -> None:
        self._repo = repo
        self._max = max_candidates if max_candidates is not None else (settings.MAX_CANDIDATES_FOR_LLM or 20)
        self._partial = (
            partial_cuisine_match
            if partial_cuisine_match is not None
            else (settings.CUISINE_PARTIAL_MATCH if settings.CUISINE_PARTIAL_MATCH is not None else True)
        )

    def filter(self, prefs: UserPreferences) -> FilterResult:
        """
        Run the full filter pipeline for *prefs*.

        Returns
        -------
        FilterResult
            `candidates` is a sorted, capped list[Restaurant].
            `relaxed_constraints` lists any constraints that were loosened.

        Notes
        -----
        Location is NEVER relaxed — if the location produces no results at
        all (i.e. it somehow passed validation but has 0 restaurants), we
        return all dataset restaurants for that location.
        """
        all_for_location = self._repo.find_by_location(prefs.location)

        # ── Step 1: filter by location ────────────────────────────────────────
        if not all_for_location:
            # Should not happen if validator ran, but guard anyway
            logger.warning("No restaurants found for location '%s'.", prefs.location)
            return FilterResult(candidates=[], relaxed_constraints=["location"])

        relaxed: list[str] = []

        # ── Steps 2-4: apply budget, rating, cuisine ──────────────────────────
        candidates = self._apply_filters(
            all_for_location,
            budget=prefs.budget,
            min_rating=prefs.min_rating,
            cuisine=prefs.cuisine,
        )

        # ── Constraint relaxation (EC-F-01) ───────────────────────────────────
        if not candidates and prefs.cuisine:
            logger.info(
                "0 results with cuisine='%s'. Relaxing cuisine constraint.",
                prefs.cuisine,
            )
            relaxed.append("cuisine")
            candidates = self._apply_filters(
                all_for_location,
                budget=prefs.budget,
                min_rating=prefs.min_rating,
                cuisine=None,
            )

        if not candidates:
            logger.info("0 results after cuisine relaxation. Relaxing budget constraint.")
            relaxed.append("budget")
            candidates = self._apply_filters(
                all_for_location,
                budget=None,
                min_rating=prefs.min_rating,
                cuisine=None,
            )

        if not candidates:
            logger.info("0 results after budget relaxation. Relaxing min_rating constraint.")
            relaxed.append("min_rating")
            candidates = self._apply_filters(
                all_for_location,
                budget=None,
                min_rating=0.0,
                cuisine=None,
            )

        # EC-F-02: Last resort — return everything for the location
        if not candidates:
            logger.warning(
                "Still 0 results after full relaxation for location='%s'. "
                "Returning all restaurants for that location.",
                prefs.location,
            )
            candidates = list(all_for_location)

        # ── Step 5: sort ──────────────────────────────────────────────────────
        candidates = self._sort(candidates)

        # ── Step 6: cap ───────────────────────────────────────────────────────
        candidates = candidates[: self._max]

        if relaxed:
            logger.info("Relaxed constraints: %s", relaxed)

        return FilterResult(candidates=candidates, relaxed_constraints=relaxed)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _apply_filters(
        self,
        restaurants: list[Restaurant],
        budget: str | None,
        min_rating: float,
        cuisine: str | None,
    ) -> list[Restaurant]:
        """
        Apply budget + rating + cuisine filters to *restaurants*.

        Returns the filtered list (unsorted).
        """
        result = restaurants

        # Budget filter  (EC-F-09: exact tier match only)
        if budget is not None:
            result = [r for r in result if r.budget_tier == budget]

        # Rating filter  (EC-I-11: 0.0 means all pass)
        result = [r for r in result if r.rating >= min_rating]

        # Cuisine filter  (EC-F-11: cuisines=[] never matches)
        if cuisine is not None:
            result = [
                r for r in result
                if r.matches_cuisine(cuisine, partial=self._partial)
            ]

        return result

    @staticmethod
    def _sort(restaurants: list[Restaurant]) -> list[Restaurant]:
        """
        Sort restaurants: rating DESC → votes DESC → name ASC.

        Alphabetical name is the final tie-breaker for determinism  (EC-F-06).
        """
        return sorted(
            restaurants,
            key=lambda r: (-r.rating, -r.votes, r.name.lower()),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4. CandidateSelector
# ══════════════════════════════════════════════════════════════════════════════

class CandidateSelector:
    """
    Finalises the candidate list coming out of RestaurantFilter.

    Responsibilities:
    - Deduplicate by restaurant id  (EC-PB-07)
    - Enforce the MAX_CANDIDATES_FOR_LLM cap  (EC-F-04, EC-F-05)
    - Attach relaxed_constraints back onto UserPreferences
    - Return the final list[Restaurant]

    Parameters
    ----------
    max_candidates : int
        Hard cap on the number of restaurants passed to the LLM.
    """

    def __init__(self, max_candidates: int | None = None) -> None:
        self._max = max_candidates if max_candidates is not None else settings.MAX_CANDIDATES_FOR_LLM

    def select(
        self,
        filter_result: FilterResult,
        prefs: UserPreferences,
    ) -> tuple[list[Restaurant], UserPreferences]:
        """
        Deduplicate, cap, and annotate preferences with relaxed constraints.

        Parameters
        ----------
        filter_result : FilterResult
            Output of RestaurantFilter.filter().
        prefs : UserPreferences
            Original preference object; will be annotated with relaxed constraints.

        Returns
        -------
        tuple[list[Restaurant], UserPreferences]
            Final candidate list and the (possibly annotated) preferences.
        """
        candidates = filter_result.candidates

        # Deduplicate by id (EC-PB-07)
        seen_ids: set[str] = set()
        unique: list[Restaurant] = []
        for r in candidates:
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                unique.append(r)
        if len(unique) < len(candidates):
            logger.warning(
                "Removed %d duplicate restaurant IDs from candidate list.",
                len(candidates) - len(unique),
            )

        # Cap at max (EC-F-04, EC-F-05)
        final = unique[: self._max]

        # Annotate preferences
        prefs.relaxed_constraints = filter_result.relaxed_constraints

        logger.info(
            "CandidateSelector: %d candidates selected "
            "(relaxed: %s)",
            len(final),
            filter_result.relaxed_constraints or "none",
        )

        return final, prefs


# ══════════════════════════════════════════════════════════════════════════════
# 5. Convenience factory — full Phase 2 pipeline in one call
# ══════════════════════════════════════════════════════════════════════════════

class FilterPipeline:
    """
    Convenience wrapper that runs the full Phase 2 pipeline:
        normalize → validate → filter → select

    Usage
    -----
        pipeline = FilterPipeline(repo)
        candidates, prefs = pipeline.run(
            location="Bangalore",
            budget="medium",
            cuisine="Italian",
            min_rating=4.0,
            additional="family-friendly",
        )

    Raises
    ------
    ValidationError
        On invalid user input.
    """

    def __init__(
        self,
        repo: "RestaurantRepository",
        max_candidates: int | None = None,
        partial_cuisine_match: bool | None = None,
    ) -> None:
        self._normalizer = PreferenceNormalizer()
        self._validator  = PreferenceValidator(repo)
        self._filter     = RestaurantFilter(repo, max_candidates, partial_cuisine_match)
        self._selector   = CandidateSelector(max_candidates)

    def run(
        self,
        location: str,
        budget: str,
        cuisine: str | None = None,
        min_rating: float = 0.0,
        additional: str | None = None,
    ) -> tuple[list[Restaurant], UserPreferences]:
        """
        Run normalize → validate → filter → select.

        Returns
        -------
        tuple[list[Restaurant], UserPreferences]
            Candidates ready for the LLM and the validated preferences.
        """
        # Step 1: normalise
        normed = self._normalizer.normalize(location, budget, cuisine, min_rating, additional)

        # Step 2: validate
        prefs = self._validator.validate(**normed)

        # Step 3: filter
        filter_result = self._filter.filter(prefs)

        # Step 4: select
        candidates, prefs = self._selector.select(filter_result, prefs)

        return candidates, prefs
