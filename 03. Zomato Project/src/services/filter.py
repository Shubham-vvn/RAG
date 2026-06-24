import logging
from src.config import settings
from src.models.restaurant import Restaurant
from src.models.preferences import UserPreferences
from src.data.repository import RestaurantRepository

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when user preferences fail validation rules."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def suggest_locations(query: str, available: list[str], max_suggestions: int = 3) -> list[str]:
    """Find closest matching locations for a mistyped query."""
    query_lower = query.lower().strip()
    if not query_lower:
        return []
    # Prefix matches (case-insensitive)
    prefix_matches = [loc for loc in available if loc.lower().startswith(query_lower)]
    if prefix_matches:
        return prefix_matches[:max_suggestions]
    # Substring matches (case-insensitive)
    substring_matches = [loc for loc in available if query_lower in loc.lower()]
    return substring_matches[:max_suggestions]


def validate_preferences(raw_preferences: dict, repository: RestaurantRepository) -> UserPreferences:
    """
    Parse, validate, and normalize raw preferences.
    Checks location and cuisine existence in the dataset.
    Raises ValidationError with detailed errors and suggestions on failure.
    """
    location = raw_preferences.get("location", "")
    budget = raw_preferences.get("budget", "")
    min_rating_val = raw_preferences.get("min_rating", 3.5)
    cuisine = raw_preferences.get("cuisine")
    additional = raw_preferences.get("additional")

    # Coerce rating to float
    if min_rating_val is not None:
        try:
            min_rating_val = float(min_rating_val)
        except (ValueError, TypeError):
            # Basic validation inside UserPreferences will catch it if it's not a float
            pass

    pref = UserPreferences(
        location=location,
        budget=budget,
        min_rating=min_rating_val,
        cuisine=cuisine,
        additional=additional,
    )

    # Perform baseline validations
    errors = pref.validate()

    # Normalize to format location/cuisine/budget correctly
    pref = pref.normalize()

    # Validate location existence
    if pref.location:
        available_locations = repository.get_locations()
        if pref.location not in available_locations:
            suggestions = suggest_locations(pref.location, available_locations)
            if suggestions:
                errors.append(
                    f"Location '{pref.location}' not found. Did you mean: {', '.join(suggestions)}?"
                )
            else:
                errors.append(f"Location '{pref.location}' not found in the dataset.")

    # Validate cuisine existence (if provided)
    if pref.cuisine:
        available_cuisines = repository.get_cuisines()
        if pref.cuisine not in available_cuisines:
            errors.append(f"Cuisine '{pref.cuisine}' not found in the dataset.")

    if errors:
        raise ValidationError(errors)

    return pref


class CandidateSelector:
    """Caps result count and applies deterministic sort."""

    def __init__(self, max_candidates: int = None):
        self.max_candidates = max_candidates or settings.MAX_CANDIDATES_FOR_LLM

    def select(self, restaurants: list[Restaurant]) -> list[Restaurant]:
        """Sort by rating desc, then votes desc, and cap at max_candidates."""
        sorted_list = sorted(restaurants, key=lambda r: (-r.rating, -r.votes))
        return sorted_list[:self.max_candidates]


class RestaurantFilter:
    """Deterministic filter pipeline for restaurant candidates."""

    def __init__(self, max_candidates: int = None):
        self.max_candidates = max_candidates or settings.MAX_CANDIDATES_FOR_LLM
        self.selector = CandidateSelector(self.max_candidates)

    def filter(
        self, restaurants: list[Restaurant], preferences: UserPreferences
    ) -> tuple[list[Restaurant], list[str]]:
        """
        Apply filter pipeline. Returns (candidates, warnings).
        Warnings track constraint relaxations applied to resolve empty candidate lists.
        """
        warnings = []
        candidates = restaurants

        # 1. Filter by location (mandatory first constraint)
        candidates = self._filter_by_location(candidates, preferences.location)
        if not candidates:
            return [], warnings

        # Preserve the location filtered set for relaxations
        location_candidates = candidates

        # 2. Filter by budget
        budget_candidates = self._filter_by_budget(location_candidates, preferences.budget)

        # 3. Filter by rating
        rating_candidates = self._filter_by_rating(budget_candidates, preferences.min_rating)

        # 4. Filter by cuisine (if provided)
        final_candidates = rating_candidates
        if preferences.cuisine:
            cuisine_candidates = self._filter_by_cuisine(rating_candidates, preferences.cuisine)
            if cuisine_candidates:
                final_candidates = cuisine_candidates
            else:
                warnings.append(
                    f"No '{preferences.cuisine}' restaurants found. Showing all cuisines."
                )
                final_candidates = rating_candidates

        # Constraint relaxation steps if no results matched
        if not final_candidates:
            warnings.append("No restaurants found in your budget. Showing all budget ranges.")
            # Drop budget & cuisine: filter by location and min_rating
            final_candidates = self._filter_by_rating(location_candidates, preferences.min_rating)

        if not final_candidates:
            relaxed_rating = max(0.0, preferences.min_rating - 0.5)
            warnings.append(
                f"Lowered minimum rating from {preferences.min_rating} to {relaxed_rating}."
            )
            # Re-filter location candidates by relaxed rating only
            final_candidates = self._filter_by_rating(location_candidates, relaxed_rating)

        # Sort and select top K
        final_candidates = self.selector.select(final_candidates)

        return final_candidates, warnings

    def _filter_by_location(self, restaurants: list[Restaurant], location: str) -> list[Restaurant]:
        loc_lower = location.lower().strip()
        return [r for r in restaurants if r.location.lower() == loc_lower]

    def _filter_by_budget(self, restaurants: list[Restaurant], budget: str) -> list[Restaurant]:
        budget_lower = budget.lower().strip()
        return [r for r in restaurants if r.budget_tier.value == budget_lower]

    def _filter_by_rating(self, restaurants: list[Restaurant], min_rating: float) -> list[Restaurant]:
        return [r for r in restaurants if r.rating >= min_rating]

    def _filter_by_cuisine(self, restaurants: list[Restaurant], cuisine: str) -> list[Restaurant]:
        cuisine_lower = cuisine.lower().strip()
        return [
            r for r in restaurants
            if any(c.lower() == cuisine_lower for c in r.cuisines)
        ]

