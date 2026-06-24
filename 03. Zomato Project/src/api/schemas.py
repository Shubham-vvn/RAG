"""
Pydantic request/response schemas for the FastAPI layer.

Implementation: Phase 5 (Hardening & Production Readiness)
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """Input payload representing search filters and user food preferences."""

    location: str = Field(..., description="Target locality or city name (e.g. Bangalore)")
    budget: str = Field(..., description="Budget category (low, medium, or high)")
    cuisine: Optional[str] = Field(None, description="Primary cuisine preference (e.g. Italian)")
    min_rating: Optional[float] = Field(3.5, description="Minimum acceptable restaurant rating (0.0 to 5.0)")
    additional: Optional[str] = Field(None, description="Optional natural language notes")


class RecommendationSchema(BaseModel):
    """Structured data for an individual ranked recommendation."""

    rank: int = Field(..., description="Rank sequence starting from 1")
    name: str = Field(..., description="Name of the restaurant")
    cuisine: str = Field(..., description="Commas separated list of cuisines served")
    rating: float = Field(..., description="Structured average rating (0.0 to 5.0)")
    estimated_cost: int = Field(..., description="Average cost for two people in INR")
    explanation: str = Field(..., description="AI-generated rationale of fit")


class MetadataSchema(BaseModel):
    """Operational and logging telemetry data from the pipeline."""

    candidates_considered: int = Field(..., description="Count of candidate restaurants filtered")
    filters_applied: dict = Field(..., description="The query parameters executed")
    model: str = Field(..., description="Groq inference engine model identifier")
    latency_ms: float = Field(..., description="API execution latency in milliseconds")
    tokens_used: int = Field(..., description="Total LLM prompt and completion tokens consumed")
    fallback_used: bool = Field(..., description="Flag indicating if the fallback engine ran")
    constraints_relaxed: List[str] = Field(..., description="List of filter relaxation adjustments logged")


class RecommendResponse(BaseModel):
    """JSON response schema returned by the recommendations API."""

    summary: Optional[str] = Field(None, description="Groq LLM-generated overall recommendation summary")
    recommendations: List[RecommendationSchema] = Field(..., description="Ranked list of top recommendations")
    metadata: MetadataSchema = Field(..., description="Execution performance diagnostics")

