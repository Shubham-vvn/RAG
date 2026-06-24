from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status

from src.api.schemas import RecommendRequest, RecommendResponse
from src.api.middleware import (
    RequestTracingMiddleware,
    validation_exception_handler,
    general_exception_handler,
)
from src.data import initialize_data
from src.services.recommendation import RecommendationService
from src.models.preferences import UserPreferences
from src.services.filter import ValidationError

repository = None
service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context to initialize dataset repository at boot."""
    global repository, service
    try:
        repository = initialize_data()
        service = RecommendationService(repository)
    except Exception as e:
        # FastAPI will catch this and prevent app boot on critical data load failures
        raise RuntimeError(f"Could not load dataset repository on startup: {e}") from e
    yield
    # Cleanup logic (if any) goes here
    repository = None
    service = None


app = FastAPI(
    title="Zomato AI Restaurant Recommender API",
    description="REST API to query, filter, and rank restaurants using Groq LLM reasoning.",
    version="1.0.0",
    lifespan=lifespan,
)

# Register request tracking middleware
app.add_middleware(RequestTracingMiddleware)

# Register exception handlers
app.add_exception_handler(ValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


@app.post(
    "/api/v1/recommend",
    response_model=RecommendResponse,
    status_code=status.HTTP_200_OK,
    summary="Get restaurant recommendations",
)
async def recommend(request: RecommendRequest):
    """
    Accept user search constraints and natural language preferences,
    filter candidate list, and return top recommendations with AI reasoning.
    """
    if not service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation engine service is currently offline.",
        )

    preferences = UserPreferences(
        location=request.location,
        budget=request.budget,
        cuisine=request.cuisine,
        min_rating=request.min_rating,
        additional=request.additional,
    )

    return service.recommend(preferences)


@app.get(
    "/api/v1/health",
    status_code=status.HTTP_200_OK,
    summary="Check API status",
)
async def health():
    """Verify backend and database ingestion health."""
    return {
        "status": "healthy",
        "dataset_loaded": repository is not None,
        "dataset_size": repository.get_count() if repository else 0,
    }


@app.get(
    "/api/v1/locations",
    status_code=status.HTTP_200_OK,
    summary="Get distinct locations",
)
async def locations():
    """Retrieve all normalized localities available in the ingested database."""
    if not repository:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Locality listings are offline.",
        )
    return {"locations": repository.get_locations()}


@app.get(
    "/api/v1/cuisines",
    status_code=status.HTTP_200_OK,
    summary="Get distinct cuisines",
)
async def cuisines(location: str | None = None):
    """Retrieve all cuisines served by restaurants in the ingested database (optionally filtered by location)."""
    if not repository:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cuisine listings are offline.",
        )
    if location:
        loc_lower = location.lower().strip()
        loc_restaurants = [r for r in repository.get_all() if r.location.lower() == loc_lower]
        loc_cuisines = sorted(list(set(c for r in loc_restaurants for c in r.cuisines)))
        return {"cuisines": loc_cuisines}
    return {"cuisines": repository.get_cuisines()}

