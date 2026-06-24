"""
Shared test fixtures for the Zomato recommendation system.

Provides:
    - sample_restaurants: Frozen list of Restaurant objects from fixtures JSON
    - sample_preferences: Valid UserPreferences for common test scenarios
    - mock_groq_response: Fixed JSON response simulating Groq LLM output
    - mock_llm_client: Mocked LLMClient returning fixed responses
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.models.restaurant import Restaurant, BudgetTier
from src.models.preferences import UserPreferences


# ── Path to test fixtures ──
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_restaurants() -> list[Restaurant]:
    """Load frozen test dataset (15 restaurants across multiple cities/cuisines)."""
    fixtures_path = FIXTURES_DIR / "sample_restaurants.json"
    with open(fixtures_path) as f:
        data = json.load(f)
    return [
        Restaurant(
            id=r["id"],
            name=r["name"],
            location=r["location"],
            cuisines=r["cuisines"],
            cost_for_two=r["cost_for_two"],
            rating=r["rating"],
            votes=r.get("votes", 0),
            rest_type=r.get("rest_type", "Unknown"),
            budget_tier=BudgetTier(r.get("budget_tier", "medium")),
        )
        for r in data
    ]


@pytest.fixture
def sample_preferences() -> UserPreferences:
    """Valid baseline preferences for testing — Bangalore, medium budget, Italian."""
    return UserPreferences(
        location="Bangalore",
        budget="medium",
        min_rating=4.0,
        cuisine="Italian",
        additional="family-friendly",
    )


@pytest.fixture
def broad_preferences() -> UserPreferences:
    """Broad preferences that match many restaurants — no cuisine filter, low rating."""
    return UserPreferences(
        location="Bangalore",
        budget="medium",
        min_rating=3.0,
        cuisine=None,
        additional=None,
    )


@pytest.fixture
def strict_preferences() -> UserPreferences:
    """Very strict preferences that may match zero restaurants."""
    return UserPreferences(
        location="Bangalore",
        budget="low",
        min_rating=4.8,
        cuisine="Molecular Gastronomy",
        additional="Michelin star only",
    )


@pytest.fixture
def mock_groq_response() -> dict:
    """Fixed Groq LLM JSON response for testing the parse → enrich pipeline."""
    return {
        "summary": "Here are the top Italian restaurants in Bangalore matching your preferences.",
        "recommendations": [
            {
                "id": "R001",
                "rank": 1,
                "explanation": "Highest-rated Italian restaurant in your budget. Known for wood-fired pizzas and family-friendly atmosphere.",
            },
            {
                "id": "R003",
                "rank": 2,
                "explanation": "Authentic Italian dining with outdoor seating. Great pasta selection within your budget range.",
            },
            {
                "id": "R005",
                "rank": 3,
                "explanation": "Popular Italian spot with a wide variety of dishes. Well-reviewed for family dining.",
            },
        ],
    }


@pytest.fixture
def mock_llm_client(mock_groq_response):
    """
    Mocked LLMClient that returns a fixed successful response.

    Usage in tests:
        service = RecommendationService(repository)
        service.llm_client = mock_llm_client
        result = service.recommend(preferences)
    """
    client = MagicMock()
    client.generate.return_value = MagicMock(
        raw_json=json.dumps(mock_groq_response),
        model="llama-3.3-70b-versatile",
        latency_ms=250.0,
        prompt_tokens=1200,
        completion_tokens=600,
        total_tokens=1800,
    )
    return client


@pytest.fixture
def mock_llm_client_failure():
    """Mocked LLMClient that simulates complete Groq failure (returns None)."""
    client = MagicMock()
    client.generate.return_value = None
    return client
