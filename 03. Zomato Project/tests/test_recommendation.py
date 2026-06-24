"""
Tests for RecommendationService — full pipeline with mocked Groq, enrichment, fallback.

Implementation: Phase 3 (alongside src/services/recommendation.py)
"""

import pytest 
from src.services.recommendation import RecommendationService
from src.services.filter import ValidationError
from src.data.repository import RestaurantRepository
from src.models.recommendation import RecommendationResponse


def test_recommend_success(sample_restaurants, sample_preferences, mock_llm_client):
    repo = RestaurantRepository(sample_restaurants)
    service = RecommendationService(repo)
    # Inject mocked client
    service.llm_client = mock_llm_client

    response = service.recommend(sample_preferences)
    assert isinstance(response, RecommendationResponse)
    assert not response.metadata.fallback_used
    assert response.metadata.model == "llama-3.3-70b-versatile"
    assert response.metadata.latency_ms == 250.0
    assert response.metadata.tokens_used == 1800

    # Verify recommendations are enriched
    assert len(response.recommendations) == 3
    rec1 = response.recommendations[0]
    assert rec1.rank == 1
    assert rec1.name == "Trattoria Milano"  # Matched R001
    assert rec1.rating == 4.5
    assert rec1.estimated_cost == 1200
    assert "wood-fired pizzas" in rec1.explanation


def test_recommend_fallback(sample_restaurants, sample_preferences, mock_llm_client_failure):
    repo = RestaurantRepository(sample_restaurants)
    service = RecommendationService(repo)
    # Inject failing client mock
    service.llm_client = mock_llm_client_failure

    response = service.recommend(sample_preferences)
    assert isinstance(response, RecommendationResponse)
    assert response.metadata.fallback_used
    assert response.metadata.model == "Heuristic-Fallback"
    assert "AI recommendations are currently offline" in response.summary

    # Should fall back to sorting by rating desc, votes desc
    # Candidates matched: R001 (4.5), R003 (4.3, 723 votes), R005 (4.0, 512 votes)
    assert len(response.recommendations) == 3
    assert response.recommendations[0].name == "Trattoria Milano"
    assert response.recommendations[1].name == "La Piazza"
    assert response.recommendations[2].name == "Olive Bistro"
    assert "AI-powered explanation is currently unavailable" in response.recommendations[0].explanation


def test_recommend_invalid_preferences(sample_restaurants):
    repo = RestaurantRepository(sample_restaurants)
    service = RecommendationService(repo)

    # Missing location should raise ValidationError immediately
    from src.models.preferences import UserPreferences
    bad_pref = UserPreferences(location="", budget="medium")
    
    with pytest.raises(ValidationError):
        service.recommend(bad_pref)
