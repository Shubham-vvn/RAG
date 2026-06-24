"""
Recommendation service — main orchestrator for the recommendation pipeline.

Coordinates: validate → filter → prompt → LLM → parse → enrich → response.
Includes heuristic fallback when LLM is unavailable.

Implementation: Phase 3 (Groq LLM Integration)
"""

import logging
from src.config import settings
from src.models.preferences import UserPreferences
from src.models.restaurant import Restaurant
from src.models.recommendation import (
    Recommendation,
    RecommendationMetadata,
    RecommendationResponse,
)
from src.data.repository import RestaurantRepository
from src.services.filter import validate_preferences, RestaurantFilter, ValidationError
from src.services.prompt_builder import PromptBuilder
from src.services.llm_client import LLMClient
from src.services.response_parser import ResponseParser, ParseError

logger = logging.getLogger(__name__)


class RecommendationService:
    """Main orchestrator for the AI restaurant recommendation pipeline."""

    def __init__(self, repository: RestaurantRepository):
        self.repository = repository
        self.filter = RestaurantFilter()
        self.prompt_builder = PromptBuilder()
        self.llm_client = LLMClient()
        self.parser = ResponseParser()

    def recommend(self, preferences: UserPreferences) -> RecommendationResponse:
        """
        Full recommendation pipeline:
        validate → filter → prompt → LLM → parse → enrich → response.
        If LLM is offline or output is malformed, falls back to heuristic ranking.
        """
        # 1. Validate and normalize preferences (handles locations and cuisines check)
        try:
            preferences = validate_preferences(preferences.to_dict(), self.repository)
        except ValidationError as e:
            # Re-raise validation errors for presentation layer
            raise e

        # 2. Filter candidate list deterministically
        all_restaurants = self.repository.get_all()
        candidates, warnings = self.filter.filter(all_restaurants, preferences)

        if not candidates:
            return self._empty_response(preferences, warnings)

        # 3. Build prompts
        system_prompt, user_prompt = self.prompt_builder.build(
            preferences=preferences, candidates=candidates
        )

        # 4. Invoke LLM Layer
        llm_result = self.llm_client.generate(system_prompt, user_prompt)

        # 5. Parse and Enrich
        if llm_result:
            try:
                parsed = self.parser.parse(llm_result.raw_json)
                recommendations = self._enrich(parsed, candidates)
                return RecommendationResponse(
                    summary=parsed.get("summary"),
                    recommendations=recommendations,
                    metadata=RecommendationMetadata(
                        candidates_considered=len(candidates),
                        filters_applied=preferences.to_dict(),
                        model=llm_result.model,
                        latency_ms=llm_result.latency_ms,
                        tokens_used=llm_result.total_tokens,
                        fallback_used=False,
                        constraints_relaxed=warnings,
                    ),
                )
            except ParseError as e:
                logger.warning(f"Parse failed after successful LLM call: {e}")

        # 6. Heuristic Fallback on LLM failure
        return self._heuristic_fallback(candidates, preferences, warnings)

    def _enrich(self, parsed: dict, candidates: list[Restaurant]) -> list[Recommendation]:
        """Join LLM rankings and explanations with structured candidate metadata."""
        candidate_map = {r.id: r for r in candidates}
        recommendations = []

        for rec in parsed["recommendations"]:
            rec_id = rec["id"]
            if rec_id in candidate_map:
                r = candidate_map[rec_id]
                recommendations.append(
                    Recommendation(
                        rank=rec["rank"],
                        name=r.name,
                        cuisine=", ".join(r.cuisines),
                        rating=r.rating,
                        estimated_cost=r.cost_for_two,
                        explanation=rec["explanation"],
                    )
                )
            else:
                logger.warning(
                    f"LLM recommended restaurant ID '{rec_id}' not found in candidate list. Skipping."
                )

        # Re-sort by rank just to ensure list order matches LLM intention
        recommendations.sort(key=lambda x: x.rank)
        return recommendations

    def _empty_response(
        self, preferences: UserPreferences, warnings: list[str]
    ) -> RecommendationResponse:
        """Create a response when zero candidates pass deterministic filtering."""
        return RecommendationResponse(
            summary="No restaurants matched your filters in this location.",
            recommendations=[],
            metadata=RecommendationMetadata(
                candidates_considered=0,
                filters_applied=preferences.to_dict(),
                model="None",
                latency_ms=0.0,
                tokens_used=0,
                fallback_used=False,
                constraints_relaxed=warnings,
            ),
        )

    def _heuristic_fallback(
        self, candidates: list[Restaurant], preferences: UserPreferences, warnings: list[str]
    ) -> RecommendationResponse:
        """Sort candidates by rating desc, votes desc when LLM is unavailable."""
        top_k = settings.TOP_K_RECOMMENDATIONS
        # Sort by rating descending, then votes descending
        sorted_candidates = sorted(candidates, key=lambda r: (-r.rating, -r.votes))
        shortlist = sorted_candidates[:top_k]

        recommendations = [
            Recommendation(
                rank=i + 1,
                name=r.name,
                cuisine=", ".join(r.cuisines),
                rating=r.rating,
                estimated_cost=r.cost_for_two,
                explanation=(
                    f"Ranked #{i+1} based on rating ({r.rating}⭐) and popularity ({r.votes} votes). "
                    f"AI-powered explanation is currently unavailable."
                ),
            )
            for i, r in enumerate(shortlist)
        ]

        return RecommendationResponse(
            summary="AI recommendations are currently offline. Showing top choices based on reviews and popularity.",
            recommendations=recommendations,
            metadata=RecommendationMetadata(
                candidates_considered=len(candidates),
                filters_applied=preferences.to_dict(),
                model="Heuristic-Fallback",
                latency_ms=0.0,
                tokens_used=0,
                fallback_used=True,
                constraints_relaxed=warnings,
            ),
        )

