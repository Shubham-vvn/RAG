"""
CLI interface — interactive terminal-based recommendation flow.

Implementation: Phase 4 (User Interface)
"""

import sys
from src.models.preferences import UserPreferences
from src.services.recommendation import RecommendationService
from src.services.filter import ValidationError
from src.data.repository import RestaurantRepository


def run_cli(repository: RestaurantRepository):
    """Launch the interactive terminal-based UI."""
    print("==================================================")
    print("   🍽️  Zomato AI Restaurant Recommender CLI       ")
    print("==================================================")

    # 1. Guide user with sample locations
    available_locs = repository.get_locations()
    print(f"Sample Locations in dataset: {', '.join(available_locs[:10])} ...")

    # Prompt Location (required)
    while True:
        location_input = input("📍 Enter Location: ").strip()
        if location_input:
            break
        print("❌ Location is required. Please enter a valid location.")

    # Prompt Budget (required)
    while True:
        budget_input = input("💰 Enter Budget (low / medium / high): ").strip().lower()
        if budget_input in ("low", "medium", "high"):
            break
        print("❌ Invalid budget. Please type 'low', 'medium', or 'high'.")

    # Guide user with sample cuisines available in the selected location
    loc_restaurants = [r for r in repository.get_all() if r.location.lower() == location_input.lower().strip()]
    available_cuisines = sorted(set(c for r in loc_restaurants for c in r.cuisines))
    if not available_cuisines:
        available_cuisines = repository.get_cuisines()
    print(f"Sample Cuisines: {', '.join(available_cuisines[:12])} ...")

    # Prompt Cuisine (optional)
    cuisine_input = input("🍜 Enter Cuisine (optional, press Enter to skip): ").strip()
    cuisine_val = cuisine_input if cuisine_input else None

    # Prompt Min Rating (optional, default 3.5)
    rating_input = input("⭐ Enter Minimum Rating (0.0 - 5.0, default 3.5): ").strip()
    if rating_input:
        try:
            min_rating_val = float(rating_input)
        except ValueError:
            print("⚠️ Invalid rating format. Defaulting to 3.5.")
            min_rating_val = 3.5
    else:
        min_rating_val = 3.5

    # Prompt Additional Preferences (optional)
    additional_input = input("📝 Enter any additional preferences (optional, press Enter to skip): ").strip()
    additional_val = additional_input if additional_input else None

    print("\n⏳ Processing your preferences and fetching recommendations...")

    # Build UserPreferences
    preferences = UserPreferences(
        location=location_input,
        budget=budget_input,
        min_rating=min_rating_val,
        cuisine=cuisine_val,
        additional=additional_val,
    )

    # Run recommendation pipeline
    service = RecommendationService(repository)
    try:
        response = service.recommend(preferences)
    except ValidationError as e:
        print("\n❌ Input Validation Failed:")
        for err in e.errors:
            print(f"   - {err}")
        return
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        return

    # Render results
    print("\n" + "=" * 60)
    print(f" 🍽️  Restaurant Recommendations for {preferences.location.title()}")
    cuisine_disp = preferences.cuisine.title() if preferences.cuisine else "Any Cuisines"
    print(f" Budget: {preferences.budget.upper()} | Cuisine: {cuisine_disp} | Min Rating: ⭐ {preferences.min_rating}")
    print("=" * 60)

    # Render warnings (constraint relaxation notices)
    if response.metadata.constraints_relaxed:
        print("\n⚠️  Constraint Adjustments Applied:")
        for w in response.metadata.constraints_relaxed:
            print(f"   - {w}")
        print("-" * 60)

    # Render AI Summary
    if response.summary:
        print(f"\n🤖 AI Summary:\n\"{response.summary}\"")
        print("\n" + "-" * 60)

    # Render Recommendations
    if not response.recommendations:
        print("\n😞 No recommendations matched your criteria.")
    else:
        for rec in response.recommendations:
            print(f"\n🏆 #{rec.rank}  {rec.name}")
            print(f"   🍜 Cuisine: {rec.cuisine}")
            print(f"   ⭐ Rating:  {rec.rating} / 5.0")
            print(f"   💰 Cost:    ₹{rec.estimated_cost} for two")
            print(f"   🤖 AI Rationale:")
            print(f"      \"{rec.explanation}\"")
            print("\n" + "-" * 60)

    # Render Metadata
    meta = response.metadata
    print(f"\n📊 Execution Metadata:")
    print(f"   Candidates considered: {meta.candidates_considered}")
    print(f"   Model used:            {meta.model}")
    print(f"   Latency:               {meta.latency_ms}ms")
    print(f"   Tokens consumed:       {meta.tokens_used}")
    print(f"   Fallback triggered:    {meta.fallback_used}")
    print("=" * 60 + "\n")

# See implementation-plan.md §5.1 for full specification
