"""
Tests for PromptBuilder — prompt structure, candidate serialization, token budget.

Implementation: Phase 3 (alongside src/services/prompt_builder.py)
"""

import json
from src.services.prompt_builder import PromptBuilder
from src.models.preferences import UserPreferences


def test_prompt_builder_basic(sample_restaurants, sample_preferences):
    builder = PromptBuilder()
    sys_prompt, usr_prompt = builder.build(sample_preferences, sample_restaurants)

    # Verify system prompt has JSON schema instructions and grounding rules
    assert "You MUST only recommend restaurants from the CANDIDATES list" in sys_prompt
    assert "OUTPUT JSON SCHEMA" in sys_prompt
    assert "summary" in sys_prompt
    assert "recommendations" in sys_prompt

    # Verify user prompt includes preferences
    assert f"Location: {sample_preferences.location}" in usr_prompt
    assert f"Budget: {sample_preferences.budget}" in usr_prompt
    assert f"Cuisine: {sample_preferences.cuisine}" in usr_prompt

    # Check candidate serialization
    parsed_candidates = json.loads(usr_prompt.split("CANDIDATES (")[1].split("):\n")[1].split("\n\nTASK:")[0])
    assert len(parsed_candidates) == len(sample_restaurants)
    # Check that candidate JSON uses compact fields: id, name, cuisines, cost_for_two, rating
    first_cand = parsed_candidates[0]
    assert "id" in first_cand
    assert "name" in first_cand
    assert "cuisines" in first_cand
    assert "cost_for_two" in first_cand
    assert "rating" in first_cand
    assert "votes" not in first_cand  # compacted

