# 🚀 Implementation Plan: AI-Powered Restaurant Recommendation System

> **Production-Level Implementation Plan** for the Zomato-inspired restaurant recommendation service.
> This document provides a **step-by-step, file-by-file blueprint** for building the entire system from scratch.
>
> **Source Documents:**
> - [`problemStatement.txt`](./problemStatement.txt) — Original problem statement and objectives
> - [`context.md`](./context.md) — Product requirements, user input spec, workflow
> - [`architecture.md`](./architecture.md) — Full technical architecture and component design

---

## Table of Contents

1. [Pre-Implementation Setup](#1-pre-implementation-setup)
2. [Phase 1 — Data Ingestion Layer](#2-phase-1--data-ingestion-layer)
3. [Phase 2 — User Input & Filtering Layer](#3-phase-2--user-input--filtering-layer)
4. [Phase 3 — Groq LLM Integration](#4-phase-3--groq-llm-integration)
5. [Phase 4 — User Interface](#5-phase-4--user-interface)
6. [Phase 5 — Hardening & Production Readiness](#6-phase-5--hardening--production-readiness)
7. [Dependency Graph](#7-dependency-graph)
8. [Risk Register](#8-risk-register)
9. [Definition of Done (Global)](#9-definition-of-done-global)

---

## 1. Pre-Implementation Setup

### 1.1 Project Scaffolding

**Goal:** Create the full directory structure, virtual environment, and base configuration files so all subsequent phases can begin immediately.

#### Tasks

| # | Task | File(s) | Priority |
|---|------|---------|----------|
| 1.1.1 | Create project directory structure | All directories below | 🔴 Critical |
| 1.1.2 | Initialize Python virtual environment | `venv/` | 🔴 Critical |
| 1.1.3 | Create `requirements.txt` with pinned dependencies | `requirements.txt` | 🔴 Critical |
| 1.1.4 | Create `requirements-dev.txt` for dev tools | `requirements-dev.txt` | 🟡 Medium |
| 1.1.5 | Create `.env.example` with placeholder values | `.env.example` | 🔴 Critical |
| 1.1.6 | Create `.gitignore` | `.gitignore` | 🔴 Critical |
| 1.1.7 | Create `config.py` with `pydantic-settings` | `src/config.py` | 🔴 Critical |
| 1.1.8 | Create `README.md` skeleton | `README.md` | 🟡 Medium |

#### Directory Structure to Create

```
zomato-recommendation/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── restaurant.py
│   │   ├── preferences.py
│   │   └── recommendation.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   │   └── repository.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── filter.py
│   │   ├── prompt_builder.py
│   │   ├── llm_client.py
│   │   ├── response_parser.py
│   │   └── recommendation.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   └── middleware.py
│   └── ui/
│       ├── __init__.py
│       ├── cli.py
│       └── streamlit_app.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_preprocessor.py
│   ├── test_filter.py
│   ├── test_prompt_builder.py
│   ├── test_response_parser.py
│   ├── test_recommendation.py
│   └── fixtures/
│       └── sample_restaurants.json
├── data/
├── docs/
│   ├── context.md
│   ├── architecture.md
│   ├── problemStatement.txt
│   └── implementation-plan.md
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

#### 1.1.3 — `requirements.txt`

```
groq>=0.5.0
datasets>=2.14.0
pandas>=2.0.0
pyarrow>=14.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
streamlit>=1.28.0
```

#### 1.1.4 — `requirements-dev.txt`

```
-r requirements.txt
pytest>=8.0.0
pytest-cov>=4.0.0
pytest-asyncio>=0.23.0
black>=24.0.0
flake8>=7.0.0
mypy>=1.8.0
```

#### 1.1.5 — `.env.example`

```env
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_FALLBACK_MODEL=llama-3.1-8b-instant
GROQ_TEMPERATURE=0.3
GROQ_MAX_TOKENS=2048

# Hugging Face Dataset
HF_DATASET_NAME=ManikaSaini/zomato-restaurant-recommendation
HF_DATASET_SPLIT=train

# Application Settings
MAX_CANDIDATES_FOR_LLM=20
TOP_K_RECOMMENDATIONS=5
DATA_CACHE_PATH=./data/zomato_cache.parquet
LOG_LEVEL=INFO
```

#### 1.1.6 — `.gitignore`

```
.env
data/*.parquet
data/*.csv
__pycache__/
*.pyc
.pytest_cache/
venv/
.mypy_cache/
*.egg-info/
dist/
build/
```

#### 1.1.7 — `config.py` Implementation

```python
"""
Centralized configuration using pydantic-settings.
All settings are loaded from environment variables / .env file.
"""
import logging
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with typed defaults and .env support."""

    # ── Hugging Face ──
    HF_DATASET_NAME: str = "ManikaSaini/zomato-restaurant-recommendation"
    HF_DATASET_SPLIT: str = "train"

    # ── Budget Thresholds (INR) ──
    BUDGET_LOW_MAX: int = 500
    BUDGET_MEDIUM_MAX: int = 1500

    # ── Candidates & Results ──
    MAX_CANDIDATES_FOR_LLM: int = 20
    TOP_K_RECOMMENDATIONS: int = 5

    # ── Groq ──
    GROQ_API_KEY: str  # Required — no default; will fail fast if missing
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_FALLBACK_MODEL: str = "llama-3.1-8b-instant"
    GROQ_TEMPERATURE: float = 0.3
    GROQ_RETRY_TEMPERATURE: float = 0.1
    GROQ_MAX_TOKENS: int = 2048
    GROQ_MAX_RETRIES: int = 3

    # ── Data Cache ──
    DATA_CACHE_PATH: str = "./data/zomato_cache.parquet"

    # ── Logging ──
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
```

#### Acceptance Criteria for Setup

- [ ] All directories exist with `__init__.py` files
- [ ] `pip install -r requirements.txt` succeeds without errors
- [ ] `from src.config import settings` works (with valid `.env`)
- [ ] `.env` is gitignored; `.env.example` is committed
- [ ] `pytest --collect-only` discovers test directory

---

## 2. Phase 1 — Data Ingestion Layer

**Goal:** Load the Zomato dataset from Hugging Face, preprocess it into a canonical schema, cache it locally, and expose it via an in-memory repository.

**Dependencies:** Phase 0 (Setup) complete

### 2.1 Data Models — `src/models/restaurant.py`

| # | Task | Details |
|---|------|---------|
| 2.1.1 | Define `Restaurant` dataclass | All canonical fields: id, name, location, cuisines, cost_for_two, rating, votes, rest_type, budget_tier |
| 2.1.2 | Define `BudgetTier` enum | `LOW`, `MEDIUM`, `HIGH` with string values |
| 2.1.3 | Add `to_dict()` and `from_dict()` methods | For serialization to/from JSON (used in prompt building) |
| 2.1.4 | Add `to_compact_dict()` | Returns only fields needed for LLM prompt (id, name, cuisines, cost_for_two, rating) — minimizes token usage |

#### Implementation Pattern

```python
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class BudgetTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Restaurant:
    id: str
    name: str
    location: str
    cuisines: list[str]
    cost_for_two: int
    rating: float
    votes: int = 0
    rest_type: str = "Unknown"
    budget_tier: BudgetTier = BudgetTier.MEDIUM

    def to_dict(self) -> dict:
        """Full serialization for storage/debugging."""
        d = asdict(self)
        d["budget_tier"] = self.budget_tier.value
        return d

    def to_compact_dict(self) -> dict:
        """Minimal serialization for LLM prompt (save tokens)."""
        return {
            "id": self.id,
            "name": self.name,
            "cuisines": ", ".join(self.cuisines),
            "cost_for_two": self.cost_for_two,
            "rating": self.rating,
        }
```

#### Acceptance Criteria

- [ ] `Restaurant` dataclass instantiates with all required fields
- [ ] `BudgetTier` enum has exactly 3 values: `low`, `medium`, `high`
- [ ] `to_compact_dict()` returns only 5 fields (id, name, cuisines, cost_for_two, rating)
- [ ] Unit test: create a Restaurant, verify `to_dict()` roundtrip

---

### 2.2 Dataset Loader — `src/data/loader.py`

| # | Task | Details |
|---|------|---------|
| 2.2.1 | Implement `DatasetLoader` class | Fetches dataset from Hugging Face `datasets` library |
| 2.2.2 | Add local cache check | If `DATA_CACHE_PATH` exists, load from parquet instead of downloading |
| 2.2.3 | Add cache write | After first download + preprocess, save to parquet |
| 2.2.4 | Add retry logic | Retry Hugging Face download up to 3 times with exponential backoff |
| 2.2.5 | Add logging | Log download start, row count, cache hit/miss, time taken |

#### Implementation Pattern

```python
import logging
import os
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from src.config import settings

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads the Zomato dataset from Hugging Face or local cache."""

    def __init__(self, cache_path: str = None):
        self.cache_path = Path(cache_path or settings.DATA_CACHE_PATH)

    def load(self) -> pd.DataFrame:
        """Load dataset from cache or Hugging Face."""
        if self.cache_path.exists():
            logger.info(f"Loading dataset from cache: {self.cache_path}")
            start = time.time()
            df = pd.read_parquet(self.cache_path)
            logger.info(f"Loaded {len(df)} rows from cache in {time.time() - start:.2f}s")
            return df

        logger.info(f"Cache not found. Downloading from Hugging Face: {settings.HF_DATASET_NAME}")
        return self._download_and_cache()

    def _download_and_cache(self) -> pd.DataFrame:
        """Download from Hugging Face with retry, then cache locally."""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                start = time.time()
                dataset = load_dataset(settings.HF_DATASET_NAME, split=settings.HF_DATASET_SPLIT)
                df = dataset.to_pandas()
                elapsed = time.time() - start
                logger.info(f"Downloaded {len(df)} rows in {elapsed:.2f}s (attempt {attempt})")

                # Cache locally
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(self.cache_path, index=False)
                logger.info(f"Cached dataset to {self.cache_path}")
                return df

            except Exception as e:
                logger.warning(f"Download attempt {attempt}/{max_retries} failed: {e}")
                if attempt == max_retries:
                    raise RuntimeError(f"Failed to download dataset after {max_retries} attempts") from e
                time.sleep(2 ** attempt)  # Exponential backoff

    def clear_cache(self):
        """Delete local cache to force re-download."""
        if self.cache_path.exists():
            self.cache_path.unlink()
            logger.info(f"Cache cleared: {self.cache_path}")
```

#### Acceptance Criteria

- [ ] `loader.load()` downloads from Hugging Face on first run
- [ ] `loader.load()` reads from parquet cache on subsequent runs
- [ ] Download retries 3 times with exponential backoff on failure
- [ ] Parquet cache file is created in `./data/` directory
- [ ] Logging shows: cache hit/miss, row count, time taken
- [ ] `clear_cache()` deletes the cache file

---

### 2.3 Data Preprocessor — `src/data/preprocessor.py`

| # | Task | Details |
|---|------|---------|
| 2.3.1 | Implement `DataPreprocessor` class | Maps raw columns to canonical schema |
| 2.3.2 | Implement column renaming | Map raw field names to canonical names |
| 2.3.3 | Implement cuisine parsing | Split `"Italian, Chinese"` → `["Italian", "Chinese"]` |
| 2.3.4 | Implement numeric coercion | `rating` → float, `cost_for_two` → int; drop invalid rows |
| 2.3.5 | Implement location normalization | Trim, title-case, city alias mapping |
| 2.3.6 | Implement budget tier derivation | Map `cost_for_two` to `low`/`medium`/`high` using thresholds |
| 2.3.7 | Implement null handling | Drop rows with null name/location; impute defaults for optional fields |
| 2.3.8 | Add ID generation | Generate stable `id` from row index if not present in dataset |

#### City Alias Map (Initial)

```python
CITY_ALIASES = {
    "Bengaluru": "Bangalore",
    "Bombay": "Mumbai",
    "Calcutta": "Kolkata",
    "Madras": "Chennai",
    "NCR": "New Delhi",
}
```

#### Column Mapping

```python
COLUMN_MAP = {
    "restaurant_name": "name",      # or "name" depending on dataset schema
    "location": "location",
    "city": "location",             # fallback if 'location' not present
    "cuisines": "cuisines",
    "average_cost_for_two": "cost_for_two",
    "aggregate_rating": "rating",
    "votes": "votes",
    "rest_type": "rest_type",
}
```

> **Important:** The actual column names in the Hugging Face dataset may differ from the expected names above. **Task 2.3.2 must include a dataset inspection step** where the developer prints `df.columns` and adjusts the mapping accordingly.

#### Acceptance Criteria

- [ ] Raw DataFrame is converted to canonical schema with all expected columns
- [ ] Cuisine strings are correctly split into lists (handle edge cases: nulls, single cuisine, trailing commas)
- [ ] Rating is float in [0.0, 5.0]; rows with non-numeric ratings are dropped
- [ ] Cost is int > 0; rows with non-numeric cost are dropped
- [ ] Locations are title-cased and alias-mapped (e.g., "Bengaluru" → "Bangalore")
- [ ] Budget tier is correctly derived: ≤500 → low, 501–1500 → medium, >1500 → high
- [ ] Null names/locations are dropped; null votes default to 0
- [ ] Every row has a unique `id`
- [ ] Unit test: preprocess a 5-row raw DataFrame, verify all transformations

---

### 2.4 Restaurant Repository — `src/data/repository.py`

| # | Task | Details |
|---|------|---------|
| 2.4.1 | Implement `RestaurantRepository` class | In-memory query interface over preprocessed data |
| 2.4.2 | `get_all()` → `list[Restaurant]` | Return all restaurants |
| 2.4.3 | `get_locations()` → `list[str]` | Return distinct locations (sorted, for UI dropdowns) |
| 2.4.4 | `get_cuisines()` → `list[str]` | Return distinct cuisines (sorted, for UI dropdowns) |
| 2.4.5 | `get_count()` → `int` | Total restaurant count |
| 2.4.6 | Convert DataFrame rows to `Restaurant` objects | Map each row to a `Restaurant` dataclass instance |

#### Implementation Pattern

```python
class RestaurantRepository:
    """In-memory query interface over preprocessed restaurant data."""

    def __init__(self, restaurants: list[Restaurant]):
        self._restaurants = restaurants
        self._locations = sorted(set(r.location for r in restaurants))
        self._cuisines = sorted(set(c for r in restaurants for c in r.cuisines))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "RestaurantRepository":
        """Build repository from preprocessed DataFrame."""
        restaurants = [
            Restaurant(
                id=str(row.get("id", idx)),
                name=row["name"],
                location=row["location"],
                cuisines=row["cuisines"],
                cost_for_two=int(row["cost_for_two"]),
                rating=float(row["rating"]),
                votes=int(row.get("votes", 0)),
                rest_type=str(row.get("rest_type", "Unknown")),
                budget_tier=BudgetTier(row["budget_tier"]),
            )
            for idx, row in df.iterrows()
        ]
        return cls(restaurants)

    def get_all(self) -> list[Restaurant]:
        return self._restaurants

    def get_locations(self) -> list[str]:
        return self._locations

    def get_cuisines(self) -> list[str]:
        return self._cuisines

    def get_count(self) -> int:
        return len(self._restaurants)
```

#### Acceptance Criteria

- [ ] `from_dataframe()` correctly converts all DataFrame rows to `Restaurant` objects
- [ ] `get_locations()` returns sorted, deduplicated location names
- [ ] `get_cuisines()` returns sorted, deduplicated cuisine names (flattened from lists)
- [ ] `get_count()` matches the input DataFrame row count
- [ ] Repository is immutable after construction (no add/delete operations)

---

### 2.5 Phase 1 Integration & Verification

| # | Task | Details |
|---|------|---------|
| 2.5.1 | Create `src/data/__init__.py` with `initialize_data()` | Combines loader → preprocessor → repository in a single callable |
| 2.5.2 | Write `test_preprocessor.py` | Cover: cuisine parsing, numeric coercion, null handling, budget tier |
| 2.5.3 | Create `tests/fixtures/sample_restaurants.json` | Frozen 15–20 row dataset for deterministic tests |
| 2.5.4 | Manual verification script | Print dataset stats: row count, location distribution, rating histogram |

#### `initialize_data()` Pattern

```python
def initialize_data() -> RestaurantRepository:
    """Load, preprocess, and cache the dataset. Returns ready-to-query repository."""
    loader = DatasetLoader()
    raw_df = loader.load()

    preprocessor = DataPreprocessor()
    clean_df = preprocessor.preprocess(raw_df)

    repository = RestaurantRepository.from_dataframe(clean_df)

    logger.info(
        f"Data initialized: {repository.get_count()} restaurants, "
        f"{len(repository.get_locations())} locations, "
        f"{len(repository.get_cuisines())} cuisines"
    )
    return repository
```

#### Phase 1 Verification Checklist

- [ ] `initialize_data()` completes without errors
- [ ] Dataset has >1,000 restaurants after preprocessing
- [ ] At least 5 distinct locations exist
- [ ] At least 10 distinct cuisines exist
- [ ] Parquet cache is created on first run
- [ ] Second run loads from cache (faster, logged as cache hit)
- [ ] All unit tests in `test_preprocessor.py` pass

---

## 3. Phase 2 — User Input & Filtering Layer

**Goal:** Implement preference collection, validation, normalization, and the deterministic restaurant filter pipeline with constraint relaxation.

**Dependencies:** Phase 1 complete (need `Restaurant`, `RestaurantRepository`)

### 3.1 Preferences Model — `src/models/preferences.py`

| # | Task | Details |
|---|------|---------|
| 3.1.1 | Define `UserPreferences` dataclass | Fields: location, budget, cuisine (optional), min_rating, additional (optional) |
| 3.1.2 | Add validation method | `validate()` raises `ValueError` with specific messages |
| 3.1.3 | Add normalization method | `normalize()` returns cleaned copy |

#### Implementation Pattern

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class UserPreferences:
    location: str
    budget: str                    # "low" | "medium" | "high"
    min_rating: float = 3.5
    cuisine: Optional[str] = None
    additional: Optional[str] = None

    def validate(self) -> list[str]:
        """Validate preferences. Returns list of error messages (empty if valid)."""
        errors = []

        if not self.location or not self.location.strip():
            errors.append("Location is required and cannot be empty.")

        if self.budget not in ("low", "medium", "high"):
            errors.append(f"Budget must be 'low', 'medium', or 'high'. Got: '{self.budget}'")

        if not (0.0 <= self.min_rating <= 5.0):
            errors.append(f"Min rating must be between 0.0 and 5.0. Got: {self.min_rating}")

        if self.additional and len(self.additional) > 500:
            errors.append("Additional preferences must be 500 characters or less.")

        return errors

    def normalize(self) -> "UserPreferences":
        """Return a normalized copy of preferences."""
        return UserPreferences(
            location=self.location.strip().title(),
            budget=self.budget.lower().strip(),
            min_rating=max(0.0, min(5.0, self.min_rating)),
            cuisine=self.cuisine.strip().title() if self.cuisine else None,
            additional=self.additional.strip() if self.additional else None,
        )
```

#### Acceptance Criteria

- [ ] `validate()` catches: empty location, invalid budget, out-of-range rating, oversized additional
- [ ] `normalize()` title-cases location and cuisine, clamps rating, trims whitespace
- [ ] Unit test: valid preferences produce empty error list
- [ ] Unit test: each invalid field produces specific error message

---

### 3.2 Preference Validator — `src/services/filter.py` (Part 1)

| # | Task | Details |
|---|------|---------|
| 3.2.1 | Implement `validate_preferences()` | Takes raw input dict, returns `UserPreferences` or raises `ValidationError` |
| 3.2.2 | Implement location validation against dataset | Check if location exists in `repository.get_locations()` |
| 3.2.3 | Implement location suggestion | If location not found, suggest top-3 closest matches (case-insensitive partial match) |
| 3.2.4 | Implement cuisine validation against dataset | If cuisine provided, check if it exists in `repository.get_cuisines()` |

#### Location Suggestion Pattern

```python
def suggest_locations(query: str, available: list[str], max_suggestions: int = 3) -> list[str]:
    """Find closest matching locations for a mistyped query."""
    query_lower = query.lower()
    # Exact prefix match first
    prefix_matches = [loc for loc in available if loc.lower().startswith(query_lower)]
    if prefix_matches:
        return prefix_matches[:max_suggestions]
    # Substring match
    substring_matches = [loc for loc in available if query_lower in loc.lower()]
    return substring_matches[:max_suggestions]
```

---

### 3.3 Restaurant Filter — `src/services/filter.py` (Part 2)

| # | Task | Details |
|---|------|---------|
| 3.3.1 | Implement `RestaurantFilter` class | Executes the full deterministic filter pipeline |
| 3.3.2 | Implement `filter_by_location()` | Case-insensitive exact match on `restaurant.location` |
| 3.3.3 | Implement `filter_by_budget()` | Match `restaurant.budget_tier` to `preferences.budget` |
| 3.3.4 | Implement `filter_by_rating()` | `restaurant.rating >= preferences.min_rating` |
| 3.3.5 | Implement `filter_by_cuisine()` | If cuisine provided, match if `cuisine in restaurant.cuisines` (case-insensitive) |
| 3.3.6 | Implement `sort_and_select()` | Sort by rating desc, votes desc; take top N |
| 3.3.7 | Implement constraint relaxation | If 0 results: drop cuisine → drop budget → lower rating by 0.5 |
| 3.3.8 | Track and return relaxation warnings | Return `(candidates, warnings)` tuple |

#### Filter Pipeline Implementation

```python
class RestaurantFilter:
    """Deterministic filter pipeline for restaurant candidates."""

    def __init__(self, max_candidates: int = None):
        self.max_candidates = max_candidates or settings.MAX_CANDIDATES_FOR_LLM

    def filter(
        self, restaurants: list[Restaurant], preferences: UserPreferences
    ) -> tuple[list[Restaurant], list[str]]:
        """
        Apply filter pipeline. Returns (candidates, warnings).
        Warnings list constraint relaxations if any were applied.
        """
        warnings = []
        candidates = restaurants

        # Apply filters in sequence
        candidates = self._filter_by_location(candidates, preferences.location)
        candidates = self._filter_by_budget(candidates, preferences.budget)
        candidates = self._filter_by_rating(candidates, preferences.min_rating)

        if preferences.cuisine:
            cuisine_filtered = self._filter_by_cuisine(candidates, preferences.cuisine)
            if cuisine_filtered:
                candidates = cuisine_filtered
            else:
                warnings.append(
                    f"No '{preferences.cuisine}' restaurants found. Showing all cuisines."
                )

        # If still 0 after location + budget + rating, relax budget
        if not candidates:
            warnings.append("No restaurants found in your budget. Showing all budget ranges.")
            candidates = self._filter_by_location(restaurants, preferences.location)
            candidates = self._filter_by_rating(candidates, preferences.min_rating)

        # If still 0, relax rating
        if not candidates:
            relaxed_rating = max(0.0, preferences.min_rating - 0.5)
            warnings.append(
                f"Lowered minimum rating from {preferences.min_rating} to {relaxed_rating}."
            )
            candidates = self._filter_by_location(restaurants, preferences.location)
            candidates = self._filter_by_rating(candidates, relaxed_rating)

        # Sort and select top N
        candidates = self._sort_and_select(candidates)

        return candidates, warnings
```

#### Acceptance Criteria

- [ ] Filter pipeline narrows results correctly for each filter type
- [ ] Cuisine filter is case-insensitive
- [ ] Sort order: rating desc, then votes desc (deterministic tie-breaking)
- [ ] Top N selection respects `MAX_CANDIDATES_FOR_LLM` config
- [ ] Constraint relaxation: drops cuisine first, then budget, then lowers rating
- [ ] Warnings accurately describe which constraints were relaxed
- [ ] Unit test: 20 restaurants → filter by location → 5 remain
- [ ] Unit test: 0 results triggers cuisine relaxation → non-empty results
- [ ] Unit test: sorting produces deterministic order

---

### 3.4 Phase 2 Verification

| # | Task | Details |
|---|------|---------|
| 3.4.1 | Write `test_filter.py` | Cover: each filter type, sort order, constraint relaxation, edge cases |
| 3.4.2 | Integration test: preferences → filter → candidates | Use frozen fixture dataset, verify expected candidates returned |
| 3.4.3 | Manual test | Run filter with real dataset, verify reasonable results for "Bangalore, medium, Italian, 4.0" |

#### Phase 2 Verification Checklist

- [ ] All `test_filter.py` tests pass
- [ ] Filtering a real dataset for "Bangalore, medium, 4.0" returns 10+ candidates
- [ ] Filtering for a non-existent cuisine triggers constraint relaxation
- [ ] Location suggestions work for misspelled cities (e.g., "bangalor" → "Bangalore")

---

## 4. Phase 3 — Groq LLM Integration

**Goal:** Build the prompt builder, Groq API adapter, response parser, recommendation enricher, and the orchestrating `RecommendationService`.

**Dependencies:** Phases 1 & 2 complete

### 4.1 Recommendation Model — `src/models/recommendation.py`

| # | Task | Details |
|---|------|---------|
| 4.1.1 | Define `Recommendation` dataclass | rank, name, cuisine, rating, estimated_cost, explanation |
| 4.1.2 | Define `RecommendationMetadata` dataclass | candidates_considered, filters_applied, model, latency_ms, tokens_used, fallback_used, constraints_relaxed |
| 4.1.3 | Define `RecommendationResponse` dataclass | summary, recommendations list, metadata |

#### Implementation

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Recommendation:
    rank: int
    name: str
    cuisine: str
    rating: float
    estimated_cost: int
    explanation: str


@dataclass
class RecommendationMetadata:
    candidates_considered: int
    filters_applied: dict
    model: str
    latency_ms: float = 0.0
    tokens_used: int = 0
    fallback_used: bool = False
    constraints_relaxed: list[str] = field(default_factory=list)


@dataclass
class RecommendationResponse:
    summary: Optional[str]
    recommendations: list[Recommendation]
    metadata: RecommendationMetadata
```

---

### 4.2 Prompt Builder — `src/services/prompt_builder.py`

| # | Task | Details |
|---|------|---------|
| 4.2.1 | Implement `PromptBuilder` class | Constructs system and user messages for Groq |
| 4.2.2 | Build system prompt | Role definition, JSON output format, ranking rules, grounding instruction |
| 4.2.3 | Build user prompt | Serialize preferences + candidates + task description |
| 4.2.4 | Serialize candidates compactly | Use `to_compact_dict()` to minimize token usage |
| 4.2.5 | Include output schema example | Show exact JSON structure expected from LLM |

#### System Prompt Template

```python
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
```

#### User Prompt Template

```python
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
```

#### Acceptance Criteria

- [ ] System prompt includes grounding instruction ("ONLY from CANDIDATES list")
- [ ] System prompt includes explicit JSON schema
- [ ] User prompt correctly serializes preferences and candidates
- [ ] Candidate JSON uses compact format (5 fields per restaurant)
- [ ] Total token count for 20 candidates stays under ~2,500 tokens
- [ ] Snapshot test: prompt output matches expected format

---

### 4.3 Groq LLM Client — `src/services/llm_client.py`

| # | Task | Details |
|---|------|---------|
| 4.3.1 | Implement `LLMClient` class | Wraps `groq.Groq` SDK with retry and fallback logic |
| 4.3.2 | Implement primary model call | `chat.completions.create()` with JSON mode |
| 4.3.3 | Implement temperature retry | On invalid JSON, retry with `temperature=0.1` |
| 4.3.4 | Implement model fallback | On persistent failure, switch to fallback model |
| 4.3.5 | Implement rate limit handling | Exponential backoff with jitter for 429 errors |
| 4.3.6 | Implement latency + token logging | Log wall-clock time, prompt/completion tokens per request |
| 4.3.7 | Return structured result | Include raw JSON, model used, latency, token usage |

#### Implementation Pattern

```python
import json
import logging
import time
import random
from dataclasses import dataclass
from typing import Optional

from groq import Groq, RateLimitError, APITimeoutError, InternalServerError

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    raw_json: str
    model: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMClient:
    """Groq API adapter with retry, fallback, and error handling."""

    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def generate(self, system_prompt: str, user_prompt: str) -> Optional[LLMResult]:
        """
        Call Groq API with multi-tier fallback:
        1. Primary model (temperature=0.3)
        2. Primary model (temperature=0.1) — retry on parse failure
        3. Fallback model (temperature=0.3)
        Returns None if all attempts fail (caller should use heuristic fallback).
        """
        # Attempt 1: Primary model
        result = self._call_groq(system_prompt, user_prompt,
                                  model=settings.GROQ_MODEL,
                                  temperature=settings.GROQ_TEMPERATURE)
        if result and self._is_valid_json(result.raw_json):
            return result

        # Attempt 2: Primary model with lower temperature
        logger.warning("Invalid JSON from primary model. Retrying with lower temperature.")
        result = self._call_groq(system_prompt, user_prompt,
                                  model=settings.GROQ_MODEL,
                                  temperature=settings.GROQ_RETRY_TEMPERATURE)
        if result and self._is_valid_json(result.raw_json):
            return result

        # Attempt 3: Fallback model
        logger.warning(f"Primary model failed. Switching to fallback: {settings.GROQ_FALLBACK_MODEL}")
        result = self._call_groq(system_prompt, user_prompt,
                                  model=settings.GROQ_FALLBACK_MODEL,
                                  temperature=settings.GROQ_TEMPERATURE)
        if result and self._is_valid_json(result.raw_json):
            return result

        logger.error("All LLM attempts failed. Falling back to heuristic ranking.")
        return None

    def _call_groq(self, system_prompt: str, user_prompt: str,
                   model: str, temperature: float) -> Optional[LLMResult]:
        """Single Groq API call with rate limit retry."""
        for attempt in range(1, settings.GROQ_MAX_RETRIES + 1):
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=settings.GROQ_MAX_TOKENS,
                    response_format={"type": "json_object"},
                )
                latency_ms = (time.time() - start) * 1000

                result = LLMResult(
                    raw_json=response.choices[0].message.content,
                    model=response.model,
                    latency_ms=round(latency_ms, 1),
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

                logger.info(
                    f"Groq response: model={result.model} "
                    f"latency={result.latency_ms}ms "
                    f"tokens={result.total_tokens}"
                )
                return result

            except RateLimitError:
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Groq 429 rate limit. Retrying in {wait:.1f}s (attempt {attempt}/{settings.GROQ_MAX_RETRIES})")
                time.sleep(wait)

            except (APITimeoutError, InternalServerError) as e:
                logger.warning(f"Groq error: {e}. Attempt {attempt}/{settings.GROQ_MAX_RETRIES}")
                if attempt == settings.GROQ_MAX_RETRIES:
                    return None
                time.sleep(2 ** attempt)

            except Exception as e:
                logger.error(f"Unexpected Groq error: {e}")
                return None

        return None

    @staticmethod
    def _is_valid_json(text: str) -> bool:
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
```

#### Acceptance Criteria

- [ ] Primary model call works with valid `GROQ_API_KEY`
- [ ] JSON mode is enabled (`response_format={"type": "json_object"}`)
- [ ] Invalid JSON triggers retry with lower temperature
- [ ] Persistent failure triggers fallback model
- [ ] Rate limit (429) triggers exponential backoff with jitter
- [ ] Returns `None` when all attempts fail (caller uses heuristic)
- [ ] Latency and token usage are logged for every request
- [ ] Model ID is tracked in `LLMResult`

---

### 4.4 Response Parser — `src/services/response_parser.py`

| # | Task | Details |
|---|------|---------|
| 4.4.1 | Implement `ResponseParser` class | Parses and validates Groq JSON response |
| 4.4.2 | Validate expected schema | Must have `summary` (str) and `recommendations` (list) |
| 4.4.3 | Validate each recommendation | Must have `id`, `rank`, `explanation` |
| 4.4.4 | Handle malformed responses | Return partial results if possible; log warnings |

#### Implementation Pattern

```python
class ResponseParser:
    """Parses and validates JSON responses from Groq LLM."""

    def parse(self, raw_json: str) -> dict:
        """
        Parse Groq response JSON. Returns dict with 'summary' and 'recommendations'.
        Raises ParseError if response is fundamentally invalid.
        """
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid JSON from Groq: {e}") from e

        # Validate top-level structure
        if "recommendations" not in data:
            raise ParseError("Groq response missing 'recommendations' field")

        if not isinstance(data["recommendations"], list):
            raise ParseError("'recommendations' must be a list")

        # Validate each recommendation
        valid_recs = []
        for i, rec in enumerate(data["recommendations"]):
            if not isinstance(rec, dict):
                logger.warning(f"Recommendation {i} is not a dict, skipping")
                continue
            if "id" not in rec or "explanation" not in rec:
                logger.warning(f"Recommendation {i} missing 'id' or 'explanation', skipping")
                continue
            valid_recs.append(rec)

        if not valid_recs:
            raise ParseError("No valid recommendations in Groq response")

        return {
            "summary": data.get("summary"),
            "recommendations": valid_recs,
        }


class ParseError(Exception):
    """Raised when Groq response cannot be parsed."""
    pass
```

#### Acceptance Criteria

- [ ] Valid JSON with correct schema → returns parsed dict
- [ ] Invalid JSON → raises `ParseError`
- [ ] Missing `recommendations` field → raises `ParseError`
- [ ] Partially valid recommendations → returns only valid ones, logs warnings
- [ ] All valid recommendations are empty → raises `ParseError`
- [ ] Unit test: 5 different valid/invalid JSON strings

---

### 4.5 Recommendation Service — `src/services/recommendation.py`

| # | Task | Details |
|---|------|---------|
| 4.5.1 | Implement `RecommendationService` class | **Main orchestrator** — coordinates full pipeline |
| 4.5.2 | Implement `recommend()` method | preferences → filter → prompt → LLM → parse → enrich → response |
| 4.5.3 | Implement enrichment logic | Join LLM output (id, rank, explanation) with full Restaurant records |
| 4.5.4 | Implement heuristic fallback | When LLM returns `None`, generate top-K by rating with generic explanation |
| 4.5.5 | Build `RecommendationMetadata` | Track: candidates count, filters, model, latency, tokens, fallback, relaxations |

#### Orchestration Flow

```python
class RecommendationService:
    """Main orchestrator: preferences → recommendations."""

    def __init__(self, repository: RestaurantRepository):
        self.repository = repository
        self.filter = RestaurantFilter()
        self.prompt_builder = PromptBuilder()
        self.llm_client = LLMClient()
        self.parser = ResponseParser()

    def recommend(self, preferences: UserPreferences) -> RecommendationResponse:
        """Full recommendation pipeline."""
        # 1. Validate and normalize
        preferences = preferences.normalize()
        errors = preferences.validate()
        if errors:
            raise ValidationError(errors)

        # 2. Filter candidates
        all_restaurants = self.repository.get_all()
        candidates, warnings = self.filter.filter(all_restaurants, preferences)

        if not candidates:
            return self._empty_response(preferences, warnings)

        # 3. Build prompt
        system_prompt, user_prompt = self.prompt_builder.build(
            preferences=preferences,
            candidates=candidates,
        )

        # 4. Call Groq LLM
        llm_result = self.llm_client.generate(system_prompt, user_prompt)

        # 5. Parse and enrich (or fallback)
        if llm_result:
            try:
                parsed = self.parser.parse(llm_result.raw_json)
                recommendations = self._enrich(parsed, candidates)
                return RecommendationResponse(
                    summary=parsed.get("summary"),
                    recommendations=recommendations,
                    metadata=RecommendationMetadata(
                        candidates_considered=len(candidates),
                        filters_applied=self._build_filters_dict(preferences),
                        model=llm_result.model,
                        latency_ms=llm_result.latency_ms,
                        tokens_used=llm_result.total_tokens,
                        fallback_used=False,
                        constraints_relaxed=warnings,
                    ),
                )
            except ParseError as e:
                logger.warning(f"Parse failed after successful LLM call: {e}")

        # 6. Heuristic fallback
        return self._heuristic_fallback(candidates, preferences, warnings)
```

#### Acceptance Criteria

- [ ] `recommend()` returns `RecommendationResponse` for valid inputs
- [ ] Pipeline: validate → filter → prompt → LLM → parse → enrich works end-to-end
- [ ] Enrichment correctly joins LLM output with full restaurant data (name, cuisine, rating, cost)
- [ ] Heuristic fallback triggers when LLM returns `None`
- [ ] Heuristic fallback returns top-K sorted by rating desc with generic explanations
- [ ] Metadata tracks: candidates count, model, latency, tokens, fallback flag, relaxed constraints
- [ ] Integration test with mocked Groq client: verify full pipeline output

---

### 4.6 Phase 3 Verification

| # | Task | Details |
|---|------|---------|
| 4.6.1 | Write `test_prompt_builder.py` | Verify prompt structure, candidate serialization, token budget |
| 4.6.2 | Write `test_response_parser.py` | Valid JSON, invalid JSON, partial responses, schema validation |
| 4.6.3 | Write `test_recommendation.py` | Full pipeline with mocked Groq client; verify enrichment and fallback |
| 4.6.4 | Manual test with real Groq API | Run recommendation for "Bangalore, medium, Italian, 4.0" with real API key |

#### Phase 3 Verification Checklist

- [ ] All unit tests pass for prompt_builder, response_parser, recommendation
- [ ] Manual test produces 5 ranked restaurants with explanations from Groq
- [ ] Explanations reference user preferences (location, budget, cuisine)
- [ ] Latency is logged (expect ~200–500ms from Groq)
- [ ] Token usage is logged (expect ~1,500–2,500 total tokens)
- [ ] Heuristic fallback works when `GROQ_API_KEY` is intentionally invalid

---

## 5. Phase 4 — User Interface

**Goal:** Build a CLI interface and a Streamlit web UI that collects preferences and displays recommendations.

**Dependencies:** Phases 1–3 complete

### 5.1 CLI Interface — `src/ui/cli.py`

| # | Task | Details |
|---|------|---------|
| 5.1.1 | Implement CLI input collection | Prompt for: location, budget, cuisine, min_rating, additional |
| 5.1.2 | Display available options | Show locations and cuisines from repository for guidance |
| 5.1.3 | Call `RecommendationService.recommend()` | Pass collected preferences through pipeline |
| 5.1.4 | Display results | Print formatted recommendation cards with all fields |
| 5.1.5 | Display warnings | Show constraint relaxation warnings |
| 5.1.6 | Display metadata | Show candidates considered, model used, response time |
| 5.1.7 | Handle errors gracefully | Catch validation errors, display friendly messages |

#### CLI Output Format

```
╔══════════════════════════════════════════════════════════════╗
║  🍽️  Restaurant Recommendations for Bangalore               ║
║  Budget: Medium | Cuisine: Italian | Min Rating: ⭐ 4.0     ║
╚══════════════════════════════════════════════════════════════╝

🤖 AI Summary:
"Based on your preferences for Italian cuisine in Bangalore..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏆 #1  Trattoria Milano
   🍜 Italian, Continental
   ⭐ 4.5 / 5.0
   💰 ₹1,200 for two
   🤖 "Highest-rated Italian restaurant in your budget..."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Metadata:
   Candidates considered: 18
   Model: llama-3.3-70b-versatile
   Response time: 342ms
```

#### Acceptance Criteria

- [ ] CLI collects all 5 preference fields interactively
- [ ] Available locations and cuisines are displayed for guidance
- [ ] Results show all required fields: rank, name, cuisine, rating, cost, explanation
- [ ] Warnings (constraint relaxation) are displayed prominently
- [ ] Metadata (candidates, model, latency) is shown after results
- [ ] Validation errors show friendly messages (not stack traces)

---

### 5.2 Streamlit Web UI — `src/ui/streamlit_app.py`

| # | Task | Details |
|---|------|---------|
| 5.2.1 | Create page layout and title | Set page config, title, description |
| 5.2.2 | Build sidebar preference form | Dropdown for location, radio for budget, text input for cuisine, slider for rating, text area for additional |
| 5.2.3 | Populate dropdowns from repository | Location and cuisine options from `get_locations()` and `get_cuisines()` |
| 5.2.4 | Add "Get Recommendations" button | Triggers `RecommendationService.recommend()` |
| 5.2.5 | Display loading spinner | Show "Finding the best restaurants for you..." while Groq responds |
| 5.2.6 | Display summary banner | Show LLM-generated summary at top of results |
| 5.2.7 | Display applied filters | Show active filter badges above results |
| 5.2.8 | Display recommendation cards | Cards with rank, name, cuisine, rating, cost, explanation |
| 5.2.9 | Display warnings | Yellow banner for constraint relaxation notices |
| 5.2.10 | Display metadata expander | Collapsible section showing model, latency, tokens, candidates |
| 5.2.11 | Handle empty results | Show friendly "no results" message with suggestions |
| 5.2.12 | Handle errors | Catch exceptions, show `st.error()` messages |

#### Streamlit Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  🍽️ Zomato AI Restaurant Recommender                               │
│  Powered by Groq (llama-3.3-70b-versatile)                         │
├──────────────┬──────────────────────────────────────────────────────┤
│              │                                                      │
│  SIDEBAR:    │  MAIN AREA:                                         │
│              │                                                      │
│  📍 Location │  [Applied Filters: Bangalore | Medium | Italian]    │
│  [dropdown]  │                                                      │
│              │  ⚠️ Warning: Cuisine filter relaxed (if applicable) │
│  💰 Budget   │                                                      │
│  ○ Low       │  🤖 AI Summary Banner                               │
│  ● Medium    │  "Based on your preferences..."                     │
│  ○ High      │                                                      │
│              │  ┌─ Card #1 ──────────────────────────────────────┐ │
│  🍜 Cuisine  │  │ 🏆 #1  Trattoria Milano                       │ │
│  [dropdown]  │  │ 🍜 Italian, Continental                        │ │
│              │  │ ⭐ 4.5  💰 ₹1,200                              │ │
│  ⭐ Min Rate │  │ 🤖 "Highest-rated Italian restaurant..."       │ │
│  [slider]    │  └────────────────────────────────────────────────┘ │
│              │                                                      │
│  📝 Additional│  ┌─ Card #2 ──────────────────────────────────────┐│
│  [text area] │  │ ...                                             ││
│              │  └────────────────────────────────────────────────┘ │
│  [🔍 Get     │                                                      │
│  Recommend-  │  ▼ Metadata (expandable)                            │
│  ations]     │  Candidates: 18 | Model: llama-3.3-70b | 342ms    │
│              │                                                      │
└──────────────┴──────────────────────────────────────────────────────┘
```

#### Acceptance Criteria

- [ ] Page loads with title, description, and sidebar form
- [ ] Location dropdown is populated from dataset
- [ ] Cuisine dropdown is populated from dataset (with "Any" option)
- [ ] Rating slider ranges 0.0–5.0 with 0.5 step
- [ ] "Get Recommendations" button triggers pipeline
- [ ] Loading spinner shows while Groq processes
- [ ] Summary banner displays LLM-generated summary
- [ ] 5 recommendation cards display with all fields
- [ ] Warning banners appear for constraint relaxation
- [ ] Metadata expander shows model, latency, tokens
- [ ] Empty results show friendly message with suggestions
- [ ] Errors show `st.error()` (not stack traces)
- [ ] `streamlit run src/ui/streamlit_app.py` launches successfully

---

### 5.3 Entry Point — `src/main.py`

| # | Task | Details |
|---|------|---------|
| 5.3.1 | Implement `main()` function | Initialize data, launch CLI or Streamlit |
| 5.3.2 | Add command-line argument for mode | `--mode cli` or `--mode web` |
| 5.3.3 | Initialize data at startup | Call `initialize_data()` once |
| 5.3.4 | Pass repository to `RecommendationService` | Dependency injection |

---

### 5.4 Phase 4 Verification

- [ ] CLI: full interactive flow works end-to-end
- [ ] Streamlit: full web flow works end-to-end
- [ ] Both UIs display: applied filters, summary, cards, warnings, metadata
- [ ] Both UIs handle: empty results, validation errors, LLM failures
- [ ] Streamlit runs on `localhost:8501`

---

## 6. Phase 5 — Hardening & Production Readiness

**Goal:** Add comprehensive error handling, logging, testing, API layer, and documentation.

**Dependencies:** Phases 1–4 complete

### 6.1 Comprehensive Error Handling

| # | Task | Details |
|---|------|---------|
| 6.1.1 | Add try/except to all external calls | Dataset download, Groq API, file I/O |
| 6.1.2 | Implement custom exception classes | `ValidationError`, `ParseError`, `DataLoadError`, `LLMError` |
| 6.1.3 | Add user-friendly error messages | Map internal exceptions to user-facing strings |
| 6.1.4 | Add startup validation | Verify `GROQ_API_KEY` is set before starting |
| 6.1.5 | Add graceful shutdown | Handle KeyboardInterrupt in CLI |

---

### 6.2 Logging & Observability

| # | Task | Details |
|---|------|---------|
| 6.2.1 | Add structured logging throughout | Use `logging` module with consistent format |
| 6.2.2 | Log filter pipeline counts | `INFO: location=1200 → budget=400 → rating=120 → cuisine=18` |
| 6.2.3 | Log Groq metrics | Model, latency, tokens, success/failure |
| 6.2.4 | Log constraint relaxation | `WARNING: Relaxed cuisine filter (0 results with "Italian")` |
| 6.2.5 | Suppress sensitive data | Never log API keys or full prompt text |
| 6.2.6 | Add request tracing | Optional request ID for each recommendation |

---

### 6.3 FastAPI REST Layer (Optional)

| # | Task | Details |
|---|------|---------|
| 6.3.1 | Implement `POST /api/v1/recommend` | Main recommendation endpoint |
| 6.3.2 | Implement `GET /api/v1/health` | Health check with dataset + Groq status |
| 6.3.3 | Implement `GET /api/v1/locations` | Available locations for frontend |
| 6.3.4 | Implement `GET /api/v1/cuisines` | Available cuisines for frontend |
| 6.3.5 | Create Pydantic schemas | `RecommendRequest`, `RecommendResponse` in `api/schemas.py` |
| 6.3.6 | Add error handling middleware | Map exceptions to HTTP status codes |
| 6.3.7 | Add request ID middleware | Generate UUID per request for tracing |

#### API Route Implementation

```python
# src/api/routes.py
from fastapi import FastAPI, HTTPException
from src.api.schemas import RecommendRequest, RecommendResponse
from src.services.recommendation import RecommendationService

app = FastAPI(title="Zomato AI Recommender", version="1.0.0")

@app.post("/api/v1/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    preferences = UserPreferences(
        location=request.location,
        budget=request.budget,
        cuisine=request.cuisine,
        min_rating=request.min_rating,
        additional=request.additional,
    )
    try:
        response = recommendation_service.recommend(preferences)
        return RecommendResponse.from_domain(response)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/v1/health")
async def health():
    return {
        "status": "healthy",
        "dataset_loaded": repository is not None,
        "dataset_size": repository.get_count() if repository else 0,
    }

@app.get("/api/v1/locations")
async def locations():
    return {"locations": repository.get_locations()}

@app.get("/api/v1/cuisines")
async def cuisines():
    return {"cuisines": repository.get_cuisines()}
```

---

### 6.4 Comprehensive Test Suite

| # | Task | File | Test Count |
|---|------|------|------------|
| 6.4.1 | Preprocessor tests | `test_preprocessor.py` | 8–10 tests |
| 6.4.2 | Filter tests | `test_filter.py` | 10–12 tests |
| 6.4.3 | Prompt builder tests | `test_prompt_builder.py` | 5–6 tests |
| 6.4.4 | Response parser tests | `test_response_parser.py` | 6–8 tests |
| 6.4.5 | Recommendation service tests | `test_recommendation.py` | 6–8 tests |
| 6.4.6 | Create test fixtures | `fixtures/sample_restaurants.json` | 15–20 rows |
| 6.4.7 | Create `conftest.py` | Shared fixtures, mock Groq client | — |
| 6.4.8 | Achieve ≥80% coverage | Run `pytest --cov` | — |

#### `conftest.py` — Shared Fixtures

```python
import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock

from src.models.restaurant import Restaurant, BudgetTier


@pytest.fixture
def sample_restaurants():
    """Load frozen test dataset."""
    fixtures_path = Path(__file__).parent / "fixtures" / "sample_restaurants.json"
    with open(fixtures_path) as f:
        data = json.load(f)
    return [Restaurant(**r) for r in data]


@pytest.fixture
def mock_groq_response():
    """Mock successful Groq API response."""
    return {
        "summary": "Test summary",
        "recommendations": [
            {"id": "R001", "rank": 1, "explanation": "Test explanation 1"},
            {"id": "R002", "rank": 2, "explanation": "Test explanation 2"},
        ],
    }


@pytest.fixture
def mock_llm_client(mock_groq_response):
    """Mock LLMClient that returns fixed response."""
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
```

---

### 6.5 Documentation

| # | Task | Details |
|---|------|---------|
| 6.5.1 | Complete `README.md` | Project overview, setup, usage, architecture, API docs |
| 6.5.2 | Add docstrings to all public classes/methods | Google-style docstrings |
| 6.5.3 | Add inline comments for complex logic | Filter pipeline, prompt construction, retry logic |
| 6.5.4 | Update `context.md` and `architecture.md` | Reflect any changes made during implementation |

#### README.md Structure

```markdown
# 🍽️ Zomato AI Restaurant Recommender

## Overview
## Features
## Quick Start
### Prerequisites
### Installation
### Configuration
### Running
  - CLI Mode
  - Streamlit Web UI
  - FastAPI Server
## Architecture
## API Documentation
## Testing
## Project Structure
## Technology Stack
## Contributing
## License
```

---

### 6.6 Phase 5 Verification

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Coverage ≥80%: `pytest tests/ --cov=src --cov-report=term-missing`
- [ ] FastAPI server starts: `uvicorn src.api.routes:app`
- [ ] Health endpoint responds: `GET /api/v1/health`
- [ ] Recommend endpoint works: `POST /api/v1/recommend`
- [ ] README.md is complete and accurate
- [ ] All public classes/methods have docstrings
- [ ] No secrets in source control (`.env` is gitignored)
- [ ] Logging shows filter counts, Groq metrics, and errors

---

## 7. Dependency Graph

```
Phase 0: Setup
    │
    ├─► config.py, requirements.txt, .env.example, directory structure
    │
    ▼
Phase 1: Data Ingestion
    │
    ├─► models/restaurant.py (Restaurant, BudgetTier)
    ├─► data/loader.py (DatasetLoader)
    ├─► data/preprocessor.py (DataPreprocessor)
    ├─► data/repository.py (RestaurantRepository)
    ├─► test_preprocessor.py
    │
    ▼
Phase 2: User Input & Filtering ─────────────────────────────┐
    │                                                         │
    ├─► models/preferences.py (UserPreferences)               │
    ├─► services/filter.py (RestaurantFilter, validation)     │
    ├─► test_filter.py                                        │
    │                                                         │
    ▼                                                         │
Phase 3: Groq LLM Integration ◄──────────────────────────────┘
    │
    ├─► models/recommendation.py (Recommendation, Response, Metadata)
    ├─► services/prompt_builder.py (PromptBuilder)
    ├─► services/llm_client.py (LLMClient — Groq adapter)
    ├─► services/response_parser.py (ResponseParser)
    ├─► services/recommendation.py (RecommendationService — orchestrator)
    ├─► test_prompt_builder.py, test_response_parser.py, test_recommendation.py
    │
    ▼
Phase 4: User Interface
    │
    ├─► ui/cli.py (Terminal interface)
    ├─► ui/streamlit_app.py (Web UI)
    ├─► main.py (Entry point)
    │
    ▼
Phase 5: Hardening
    │
    ├─► api/routes.py, schemas.py, middleware.py (FastAPI REST)
    ├─► Comprehensive error handling across all modules
    ├─► Structured logging across all modules
    ├─► Full test suite (≥80% coverage)
    ├─► README.md, docstrings, inline comments
    └─► Final integration verification
```

---

## 8. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Hugging Face dataset schema changes** | Medium | High | Inspect `df.columns` in Phase 1; document actual column names; add schema validation |
| **Groq rate limits during development** | High | Medium | Use fallback model (`llama-3.1-8b-instant`); cache successful responses during dev |
| **Dataset has unexpected data quality issues** | Medium | High | Robust null handling, type coercion with `errors='coerce'`; log dropped rows |
| **Prompt engineering needs iteration** | High | Medium | Start with structured prompt template; iterate based on Groq output quality |
| **Groq JSON mode produces unexpected schema** | Medium | Medium | `ResponseParser` with schema validation; partial result recovery |
| **Large dataset causes slow filtering** | Low | Low | In-memory pandas is fast for ~50K rows; indexed if needed later |
| **GROQ_API_KEY accidentally committed** | Low | Critical | `.gitignore` includes `.env`; `.env.example` has placeholders; pre-commit hook (optional) |
| **Streamlit version incompatibility** | Low | Medium | Pin Streamlit version in `requirements.txt` |

---

## 9. Definition of Done (Global)

A phase is **done** when ALL of the following are true:

- [ ] All tasks in the phase are implemented
- [ ] All acceptance criteria for each task are met
- [ ] Unit tests pass for all new code
- [ ] No regressions in existing tests
- [ ] Code follows consistent style (PEP 8, Google-style docstrings)
- [ ] No secrets or API keys in source control
- [ ] Logging is present for key operations
- [ ] Error handling covers expected failure modes

The **entire project** is done when:

- [ ] All 5 phases are complete
- [ ] End-to-end flow works: Streamlit form → preferences → filter → Groq → recommendation cards
- [ ] Heuristic fallback works when Groq is unavailable
- [ ] Test suite passes with ≥80% coverage
- [ ] README.md provides complete setup and usage instructions
- [ ] A developer can clone the repo, add `GROQ_API_KEY` to `.env`, and run `streamlit run src/ui/streamlit_app.py` to see working recommendations

---

## Appendix A: Estimated Effort

| Phase | Estimated Effort | Complexity |
|-------|-----------------|------------|
| **Phase 0 — Setup** | 1–2 hours | 🟢 Low |
| **Phase 1 — Data Ingestion** | 3–5 hours | 🟡 Medium |
| **Phase 2 — User Input & Filtering** | 3–4 hours | 🟡 Medium |
| **Phase 3 — Groq LLM Integration** | 5–8 hours | 🔴 High |
| **Phase 4 — User Interface** | 4–6 hours | 🟡 Medium |
| **Phase 5 — Hardening** | 4–6 hours | 🟡 Medium |
| **Total** | **20–31 hours** | — |

---

## Appendix B: Quick Reference — Key Configs

| Config | Value | Where Used |
|--------|-------|------------|
| `GROQ_API_KEY` | Your key | `llm_client.py` |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | `llm_client.py` |
| `GROQ_FALLBACK_MODEL` | `llama-3.1-8b-instant` | `llm_client.py` |
| `GROQ_TEMPERATURE` | `0.3` | `llm_client.py` |
| `MAX_CANDIDATES_FOR_LLM` | `20` | `filter.py` |
| `TOP_K_RECOMMENDATIONS` | `5` | `prompt_builder.py` |
| `DATA_CACHE_PATH` | `./data/zomato_cache.parquet` | `loader.py` |
| `HF_DATASET_NAME` | `ManikaSaini/zomato-restaurant-recommendation` | `loader.py` |

---

*Implementation Plan Version: 1.0 | Created: 2026-06-21 | Based on: `problemStatement.txt` + `context.md` + `architecture.md`*
