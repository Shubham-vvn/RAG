"""
Prompt builder — crafts system and user messages for Groq LLM.

Constructs structured prompts with grounding instructions, user preferences,
candidate restaurants (compact JSON), and output schema specification.

Implementation: Phase 3 (Groq LLM Integration)
"""

import json
from src.config import settings
from src.models.preferences import UserPreferences
from src.models.restaurant import Restaurant

SYSTEM_PROMPT = """You are an expert restaurant recommendation assistant specializing in Indian cities.

RULES:
1. You MUST only recommend restaurants from the CANDIDATES list provided below.
2. Do NOT invent, fabricate, or hallucinate any restaurant names or details.
3. Rank the top {top_k} restaurants based on how well they match the user's preferences.
4. For each recommendation, provide a specific explanation of WHY this restaurant fits the user's preferences.
5. Explanations should reference the user's stated preferences (budget, cuisine, location, additional requests).
6. Write a brief overall summary of your recommendations.
7. Return your response as valid JSON matching the exact schema specified below.

OUTPUT JSON SCHEMA:
{{
  "summary": "Brief 1-2 sentence summary of the recommendations",
  "recommendations": [
    {{
      "id": "restaurant_id",
      "rank": 1,
      "explanation": "2-3 sentence explanation of why this restaurant fits the user's preferences"
    }}
  ]
}}"""

USER_PROMPT = """
USER PREFERENCES:
- Location: {location}
- Budget: {budget}
- Cuisine: {cuisine}
- Minimum Rating: {min_rating}
- Additional Preferences: {additional}

CANDIDATES ({count} restaurants):
{candidates_json}

TASK: Rank the top {top_k} restaurants from the CANDIDATES list above.
Return valid JSON with your summary and ranked recommendations."""


class PromptBuilder:
    """Crafts system and user messages for Groq LLM."""

    def build(self, preferences: UserPreferences, candidates: list[Restaurant]) -> tuple[str, str]:
        """
        Build system and user messages.
        Compactly serializes candidates to save prompt tokens.
        """
        top_k = settings.TOP_K_RECOMMENDATIONS
        # Cap top_k at candidate length to avoid recommending more than available
        top_k = min(top_k, len(candidates))

        # Serialize candidates to a compact JSON structure
        candidates_compact = [c.to_compact_dict() for c in candidates]
        candidates_json = json.dumps(candidates_compact, ensure_ascii=False, indent=2)

        # Handle optionals
        cuisine_str = preferences.cuisine if preferences.cuisine else "Any Cuisines"
        additional_str = preferences.additional if preferences.additional else "None specified"

        sys_prompt = SYSTEM_PROMPT.format(top_k=top_k)
        usr_prompt = USER_PROMPT.format(
            location=preferences.location,
            budget=preferences.budget,
            cuisine=cuisine_str,
            min_rating=preferences.min_rating,
            additional=additional_str,
            count=len(candidates),
            candidates_json=candidates_json,
            top_k=top_k,
        )

        return sys_prompt, usr_prompt

