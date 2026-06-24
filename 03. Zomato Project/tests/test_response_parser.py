"""
Tests for ResponseParser — valid JSON, invalid JSON, partial responses, schema validation.

Implementation: Phase 3 (alongside src/services/response_parser.py)
"""

import pytest 
from src.services.response_parser import ResponseParser, ParseError


def test_response_parser_valid():
    parser = ResponseParser()
    valid_json = """{
        "summary": "These are recommendations",
        "recommendations": [
            {"id": "R001", "rank": 1, "explanation": "Fits perfectly"},
            {"id": "R002", "rank": 2, "explanation": "Very nice pizza"}
        ]
    }"""
    parsed = parser.parse(valid_json)
    assert parsed["summary"] == "These are recommendations"
    assert len(parsed["recommendations"]) == 2
    assert parsed["recommendations"][0]["id"] == "R001"
    assert parsed["recommendations"][0]["rank"] == 1
    assert parsed["recommendations"][0]["explanation"] == "Fits perfectly"


def test_response_parser_invalid_json():
    parser = ResponseParser()
    bad_json = "{ malformed json "
    with pytest.raises(ParseError):
        parser.parse(bad_json)


def test_response_parser_missing_fields():
    parser = ResponseParser()
    missing_recs = """{
        "summary": "Missing recommendations"
    }"""
    with pytest.raises(ParseError) as exc_info:
        parser.parse(missing_recs)
    assert "missing 'recommendations' field" in str(exc_info.value)


def test_response_parser_partial_recommendations():
    parser = ResponseParser()
    # One recommendation is valid, one is missing explanation
    partial_json = """{
        "summary": "Some valid, some invalid",
        "recommendations": [
            {"id": "R001", "rank": 1, "explanation": "Fits perfectly"},
            {"id": "R002", "rank": 2}
        ]
    }"""
    parsed = parser.parse(partial_json)
    assert len(parsed["recommendations"]) == 1
    assert parsed["recommendations"][0]["id"] == "R001"


def test_response_parser_no_valid_recommendations():
    parser = ResponseParser()
    all_bad_json = """{
        "summary": "All invalid",
        "recommendations": [
            {"id": "R001", "rank": 1},
            {"rank": 2, "explanation": "No ID"}
        ]
    }"""
    with pytest.raises(ParseError) as exc_info:
        parser.parse(all_bad_json)
    assert "No valid recommendations" in str(exc_info.value)

