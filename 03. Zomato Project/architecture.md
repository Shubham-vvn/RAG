# 🏗️ Architecture: AI-Powered Restaurant Recommendation System

> This document describes the **complete technical architecture** for the Zomato-inspired restaurant recommendation service. The system combines structured restaurant data from Hugging Face with **Groq LLM inference** to produce personalized, explainable recommendations in near real-time.
>
> **Related Documents:**
> - [`context.md`](./context.md) — Product requirements, user input spec, and workflow
> - [`problemStatement.txt`](./problemStatement.txt) — Original problem statement

---

## 1. Executive Summary

This architecture defines a **5-layer recommendation system** that:

1. **Ingests** the Zomato restaurant dataset from Hugging Face and normalizes it into a canonical schema
2. **Collects** user preferences (location, budget, cuisine, rating, free-text) with validation and normalization
3. **Filters** restaurants deterministically using hard constraints to produce a bounded candidate set
4. **Ranks and explains** candidates using Groq's `llama-3.3-70b-versatile` LLM via structured JSON prompts
5. **Presents** ranked recommendations with AI-generated explanations in a clean, card-based UI

The architecture prioritizes **separation of concerns**, **deterministic pre-filtering** (to reduce LLM token cost and hallucination risk), **explainability** (every pick is justified), and **graceful degradation** (fallback to heuristic ranking when the LLM is unavailable).

---

## 2. Architecture Goals & Design Principles

| Goal | Description | Implementation Strategy |
|------|-------------|------------------------|
| **Separation of concerns** | Data loading, filtering, LLM reasoning, and presentation are isolated modules with clear interfaces | Layered architecture: Data → Service → Presentation; each layer communicates via typed models |
| **Deterministic pre-filtering** | Hard constraints (location, budget, rating) are applied *before* the LLM to reduce token cost and hallucination risk | Python-side pandas/list filtering narrows thousands of restaurants to ~15–20 candidates before any LLM call |
| **Explainability** | Every recommendation includes a Groq LLM-generated rationale tied to user preferences | Prompt design forces per-recommendation `explanation` field; enricher joins explanations with structured data |
| **Extensibility** | Swap UI frameworks, data sources, or LLM providers without rewriting core logic | Adapter pattern for LLM client; repository pattern for data access; dependency injection throughout |
| **Testability** | Pure functions for filtering/ranking prep; mockable LLM adapter for unit tests | `LLMClient` is a thin adapter easily mocked; frozen 10–20 row dataset fixtures for deterministic tests |
| **Reliability** | Graceful degradation when LLM fails or returns invalid output | Multi-tier fallback: retry with lower temperature → model fallback → heuristic ranking → clear user messaging |
| **Performance** | Sub-second end-to-end latency for interactive use | Groq LPU provides sub-second inference; in-memory dataset avoids I/O; pre-filtering minimizes prompt tokens |

### Key Design Constraints

- **LLM is NOT used for:** loading data, hard filtering by location/budget/rating, or inventing restaurants not in the candidate list
- **All recommendations must be grounded:** the LLM can only rank and explain restaurants from the provided candidate list
- **Secrets are never committed:** `GROQ_API_KEY` lives in `.env` (gitignored), `.env.example` provides templates

---

## 3. LLM Provider: Groq — Deep Dive

**Groq is the sole LLM provider for this project.** Groq offers ultra-low-latency inference via its proprietary Language Processing Unit (LPU) hardware, making it ideal for interactive recommendation flows where users expect near-instant responses.

### 3.1 Groq Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| **SDK** | `groq` | Official Python client — `pip install groq` |
| **API Key** | `GROQ_API_KEY` | Required; set in `.env`, never committed to source control |
| **Primary Model** | `llama-3.3-70b-versatile` | 70B parameter open-weight model; strong reasoning for ranking + explanation tasks |
| **Fallback Model** | `llama-3.1-8b-instant` | 8B parameter model; faster/cheaper alternative for development, testing, or when primary model is rate-limited |
| **Temperature** | `0.3` | Low for consistent, parseable JSON output; retry with `0.1` on first parse failure |
| **Output Format** | `response_format={"type": "json_object"}` | Enforced where model supports it; ensures structured output |
| **Max Tokens** | Configurable (default: `2048`) | Sufficient for top-5 recommendations with explanations + summary |

### 3.2 Why Groq?

| Advantage | Details |
|-----------|---------|
| **Speed** | Sub-second token generation via Language Processing Unit (LPU) — essential for responsive, interactive UI where users won't wait |
| **Open models** | Runs open-weight LLaMA models, reducing vendor lock-in compared to proprietary models like GPT-4 |
| **JSON mode** | `response_format={"type": "json_object"}` produces structured output, drastically reducing parse failures for recommendation ranking |
| **Cost-effective** | Competitive pricing for high-throughput inference; open-weight models have no per-token licensing overhead |
| **Consistent latency** | LPU hardware provides predictable inference times, unlike GPU-based providers with variable queue depths |

### 3.3 Groq Client Usage

```python
from groq import Groq

client = Groq(api_key=settings.GROQ_API_KEY)

response = client.chat.completions.create(
    model=settings.GROQ_MODEL,            # "llama-3.3-70b-versatile"
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ],
    temperature=settings.GROQ_TEMPERATURE,     # 0.3
    response_format={"type": "json_object"},   # enforced where model supports it
    max_tokens=settings.GROQ_MAX_TOKENS,       # 2048
)

raw_json = response.choices[0].message.content
token_usage = response.usage  # Log: prompt_tokens, completion_tokens, total_tokens
model_used = response.model   # Log: actual model ID used
```

### 3.4 Model Selection Strategy

```
Request arrives
    │
    ▼
Try primary model (llama-3.3-70b-versatile)
    │
    ├─ ✅ Success → parse JSON → return
    │
    ├─ ❌ Invalid JSON → retry with temperature=0.1
    │   │
    │   ├─ ✅ Success → parse JSON → return
    │   └─ ❌ Still invalid → try fallback model
    │
    └─ ❌ 429 Rate Limit → exponential backoff (up to 3 retries)
        │
        ├─ ✅ Retry succeeds → parse JSON → return
        └─ ❌ All retries exhausted → try fallback model
                │
                ├─ ✅ Fallback succeeds → parse JSON → return (log model switch)
                └─ ❌ Fallback also fails → heuristic ranking
                        │
                        └─ Return top-K by rating desc with generic explanation
                           + surface "AI explanation unavailable" in UI
```

### 3.5 Groq-Specific Operational Considerations

- **Rate limiting:** Groq enforces per-minute token and request limits; implement exponential backoff with jitter
- **Token budget:** Pre-filtering to ~15–20 candidates keeps prompt tokens manageable (~1K–2K tokens for candidates JSON)
- **Latency logging:** Log `response.usage` (prompt_tokens, completion_tokens, total_tokens) and wall-clock latency per request
- **Model versioning:** Log `response.model` to track which model version served each request
- **Error codes:** Handle `429` (rate limit), `500/503` (server error), and timeout separately with appropriate retry strategies

---

## 4. System Architecture — Layered Design

### 4.1 Layer Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                       PRESENTATION LAYER                             │
│            Streamlit UI / Gradio / FastAPI + CLI                     │
│                                                                      │
│   ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────┐  │
│   │ PreferenceForm   │  │  ResultsView    │  │  SummaryBanner     │  │
│   │ (Input Widgets)  │  │ (Ranked Cards)  │  │ (AI Summary)       │  │
│   └────────┬────────┘  └────────▲────────┘  └────────▲───────────┘  │
└────────────┼────────────────────┼────────────────────┼───────────────┘
             │ Raw Input          │ RecommendationResponse
             ▼                    │
┌──────────────────────────────────────────────────────────────────────┐
│                       USER INPUT LAYER                               │
│                                                                      │
│   ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│   │ PreferenceForm   │→│PreferenceValidator│→│PreferenceNormalizer│  │
│   │ (Collect)        │  │ (Validate)       │  │ (Normalize)       │  │
│   └─────────────────┘  └──────────────────┘  └───────────┬───────┘  │
└──────────────────────────────────────────────────────────┼───────────┘
             │ UserPreferences (validated, normalized)      │
             ▼                                              │
┌──────────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                              │
│                                                                      │
│   ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│   │  DatasetLoader   │→│ DataPreprocessor  │→│RestaurantRepository│  │
│   │ (Hugging Face)   │  │ (Normalize)      │  │ (In-Memory Query) │  │
│   └─────────────────┘  └──────────────────┘  └───────────┬───────┘  │
└──────────────────────────────────────────────────────────┼───────────┘
             │ list[Restaurant] (full dataset, cached)      │
             ▼                                              │
┌──────────────────────────────────────────────────────────────────────┐
│                      INTEGRATION LAYER                               │
│                                                                      │
│   ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐   │
│   │ RestaurantFilter  │→│ CandidateSelector │→│  PromptBuilder   │   │
│   │ (Hard Filters)    │  │ (Top N, Sort)    │  │ (Craft Prompt)   │   │
│   └──────────────────┘  └──────────────────┘  └───────────┬──────┘  │
└───────────────────────────────────────────────────────────┼──────────┘
             │ Structured Prompt (system + user + candidates + task)
             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                RECOMMENDATION ENGINE (GROQ LLM LAYER)                │
│                                                                      │
│   ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│   │   LLMClient      │→│  ResponseParser   │→│RecommendationEnr. │  │
│   │ (Groq API)       │  │ (Parse JSON)     │  │ (Join + Enrich)   │  │
│   │                   │  │                  │  │                   │  │
│   │ llama-3.3-70b    │  │ Schema Validate  │  │ Merge with full   │  │
│   │ -versatile       │  │ + Error Handle   │  │ Restaurant records │  │
│   └─────────────────┘  └──────────────────┘  └───────────┬───────┘  │
└──────────────────────────────────────────────────────────┼───────────┘
             │ RecommendationResponse
             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     OUTPUT DISPLAY LAYER                              │
│                                                                      │
│   ┌──────────────────────┐  ┌────────────┐  ┌────────────────────┐  │
│   │RecommendationPresenter│→│ ResultsView │→│   SummaryBanner    │  │
│   │ (Format for UI/CLI)  │  │(Rank Cards) │  │(AI Summary Header) │  │
│   └──────────────────────┘  └────────────┘  └────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 Dependency Graph

```
config.py ─────────────────────────────────────┐
    │                                           │
    ▼                                           ▼
models/                                    services/
├── restaurant.py ◄──── data/loader.py     ├── filter.py
├── preferences.py ◄── data/preprocessor.py│   └── uses: models/restaurant
└── recommendation.py   data/repository.py │       uses: models/preferences
                            │              ├── prompt_builder.py
                            │              │   └── uses: models/*, config
                            │              ├── llm_client.py
                            │              │   └── uses: groq SDK, config
                            │              └── recommendation.py (orchestrator)
                            │                  └── uses: filter, prompt_builder,
                            │                       llm_client, models/*
                            │
                            ▼
                    ui/cli.py  OR  ui/streamlit_app.py  OR  api/routes.py
                        └── uses: services/recommendation, models/*
```

---

## 5. Component Architecture — Detailed Design

### 5.1 Data Ingestion Layer

**Responsibility:** Load, normalize, and cache the Zomato dataset once at startup (or on first request). This layer has **zero LLM dependency**.

#### Components

| Component | Role | Key Methods |
|-----------|------|-------------|
| `DatasetLoader` | Fetches `ManikaSaini/zomato-restaurant-recommendation` via Hugging Face `datasets` library | `load() → raw DataFrame` |
| `DataPreprocessor` | Maps raw columns to canonical schema, handles nulls, normalizes text fields, derives budget tiers | `preprocess(raw_df) → clean DataFrame` |
| `RestaurantRepository` | In-memory query interface over the preprocessed DataFrame; provides filtered access | `get_all()`, `get_by_location()`, `get_cuisines()`, `get_locations()` |

#### Canonical Restaurant Schema

```python
from dataclasses import dataclass

@dataclass
class Restaurant:
    id: str              # stable identifier (index or dataset id)
    name: str
    location: str        # city / locality (normalized, title-case)
    cuisines: list[str]  # e.g. ["Italian", "Continental"]
    cost_for_two: int    # numeric cost indicator (INR)
    rating: float        # e.g. 4.2 (0.0–5.0)
    votes: int           # popularity signal for tie-breaking
    rest_type: str       # casual dining, cafe, quick bites, etc.
    budget_tier: str     # derived: "low" | "medium" | "high"
```

#### Raw-to-Canonical Field Mapping

| Raw Field | Canonical Field | Transformation | Error Handling |
|-----------|----------------|----------------|----------------|
| `restaurant_name` | `name` | Direct mapping | Drop if null/empty |
| `location` / `city` | `location` | Trim, title-case, alias map (`"Bengaluru"` → `"Bangalore"`) | Drop if null |
| `cuisines` | `cuisines` | Split `"Italian, Chinese"` → `["Italian", "Chinese"]`; strip whitespace | Default to `["Unknown"]` |
| `average_cost_for_two` | `cost_for_two` | Coerce to int | Drop row if non-numeric |
| `aggregate_rating` | `rating` | Coerce to float; clamp to `[0.0, 5.0]` | Drop row if non-numeric |
| `votes` | `votes` | Coerce to int | Default to `0` |
| `rest_type` | `rest_type` | Trim, title-case | Default to `"Unknown"` |
| *(derived)* | `budget_tier` | From `cost_for_two` using thresholds | Inherit from `cost_for_two` |

#### Budget Tier Derivation

| Tier | `cost_for_two` Range (INR) | `price_range` Equivalent |
|------|---------------------------|--------------------------|
| `low` | ≤ ₹500 | 1 |
| `medium` | ₹501 – ₹1,500 | 2–3 |
| `high` | > ₹1,500 | 4 |

> ⚠️ Thresholds should be tuned after inspecting the actual dataset distribution. Configurable via `config.py` → `BUDGET_THRESHOLDS`.

#### Preprocessing Pipeline (Step-by-Step)

```python
def preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    # 1. Select and rename columns to canonical schema
    df = raw_df.rename(columns=COLUMN_MAP)[CANONICAL_COLUMNS]

    # 2. Drop rows with null name or location
    df = df.dropna(subset=["name", "location"])

    # 3. Parse cuisine strings into lists
    df["cuisines"] = df["cuisines"].apply(
        lambda x: [c.strip() for c in str(x).split(",")] if pd.notna(x) else ["Unknown"]
    )

    # 4. Coerce rating and cost_for_two to numeric; drop invalid
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["cost_for_two"] = pd.to_numeric(df["cost_for_two"], errors="coerce")
    df = df.dropna(subset=["rating", "cost_for_two"])

    # 5. Normalize location (trim, title-case, alias map)
    df["location"] = df["location"].str.strip().str.title().replace(CITY_ALIASES)

    # 6. Derive budget_tier
    df["budget_tier"] = df["cost_for_two"].apply(derive_budget_tier)

    return df
```

#### Caching Strategy

- **First run:** Download from Hugging Face → preprocess → save to `./data/zomato_cache.parquet`
- **Subsequent runs:** Load directly from local parquet cache (bypass Hugging Face download)
- **Cache invalidation:** Manual (delete cache file to force re-download)
- **Runtime:** Hold as in-memory pandas DataFrame or list of `Restaurant` objects for fast filtering

---

### 5.2 User Input Layer

**Responsibility:** Collect, validate, and normalize user preferences before passing downstream. This layer enforces input contracts so downstream components can trust the data.

#### Input Model

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class UserPreferences:
    location: str                # required — city or locality name
    budget: str                  # "low" | "medium" | "high"
    cuisine: Optional[str]       # optional primary cuisine preference
    min_rating: float            # e.g. 3.5 — minimum acceptable rating
    additional: Optional[str]    # free-text: "family-friendly, quick service"
```

#### Components

| Component | Role | Input → Output |
|-----------|------|----------------|
| `PreferenceForm` | UI form (Streamlit widgets) or CLI prompt collecting all preference fields | Raw user input → raw dict |
| `PreferenceValidator` | Enforces required fields, enum values, rating bounds; raises `ValidationError` on failure | Raw dict → validated dict |
| `PreferenceNormalizer` | Lowercases cuisine, maps city aliases, trims free text whitespace | Validated dict → `UserPreferences` |

#### Validation Rules (Enforced by `PreferenceValidator`)

| Field | Rule | On Failure |
|-------|------|------------|
| `location` | Non-empty string; must match ≥1 value in dataset (case-insensitive) | Suggest closest valid locations from dataset |
| `budget` | Must be one of `"low"`, `"medium"`, `"high"` | Reject with enum values listed |
| `min_rating` | Float in `[0.0, 5.0]` | Clamp to valid range with warning |
| `cuisine` | Optional; if provided, fuzzy match against known cuisine vocabulary from dataset | Suggest similar cuisines if no match |
| `additional` | Optional free text; max 500 characters | Truncate silently |

#### Input Fields Reference

| Input | Type | Required | Examples | Downstream Use |
|-------|------|----------|---------|----------------|
| **Location** | String | ✅ Yes | `"Delhi"`, `"Bangalore"` | Hard filter (deterministic) |
| **Budget** | Enum | ✅ Yes | `"low"`, `"medium"`, `"high"` | Hard filter → `budget_tier` match |
| **Cuisine** | String | ❌ No | `"Italian"`, `"North Indian"` | Hard filter (when provided) |
| **Min Rating** | Float [0–5] | ✅ Yes | `3.5`, `4.0` | Hard filter (≥ threshold) |
| **Additional** | Free text | ❌ No | `"family-friendly"`, `"outdoor seating"` | Soft signal → passed to LLM only |

---

### 5.3 Integration Layer

**Responsibility:** Apply hard filters, rank candidates heuristically, and assemble the Groq LLM prompt. This layer sits between structured data and the LLM, ensuring the model only reasons over a **bounded, relevant candidate set**.

#### 5.3.1 Restaurant Filter (Deterministic)

Applies deterministic filters in a fixed sequence to progressively narrow the candidate set:

```
all restaurants (full dataset, e.g. ~50K+ records)
  │
  ├─► filter by location (exact or case-insensitive match)
  │       → typically narrows to ~500–2000 restaurants
  │
  ├─► filter by budget tier (low/medium/high match)
  │       → typically narrows to ~100–500 restaurants
  │
  ├─► filter by min_rating (>= threshold)
  │       → typically narrows to ~50–200 restaurants
  │
  ├─► filter by cuisine (if provided; match if cuisine in restaurant.cuisines)
  │       → typically narrows to ~10–50 restaurants
  │
  ├─► sort by rating desc, then votes desc (deterministic tie-breaking)
  │
  └─► take top N candidates (default N = 15–20, configurable via MAX_CANDIDATES_FOR_LLM)
          → final candidate set sent to LLM
```

| Component | Role | Input → Output |
|-----------|------|----------------|
| `RestaurantFilter` | Executes the full filter pipeline | `(all_restaurants, preferences)` → `list[Restaurant]` |
| `CandidateSelector` | Caps result count, applies deterministic sort/tie-breaking | `list[Restaurant]` → `list[Restaurant]` (top N) |

#### Constraint Relaxation Strategy

If zero candidates remain after filtering, constraints are relaxed **in order** to recover results:

```
0 candidates after full filter
    │
    ├─ Step 1: Drop cuisine filter → re-run
    │   └─ Still 0? Continue to Step 2
    │
    ├─ Step 2: Drop budget filter → re-run
    │   └─ Still 0? Continue to Step 3
    │
    ├─ Step 3: Lower min_rating by 0.5 → re-run
    │   └─ Still 0? Show "no restaurants found for this location"
    │
    └─ At each step: surface a warning to the user explaining which constraint was relaxed
        e.g. "We couldn't find Italian restaurants in your budget. Showing all cuisines instead."
```

#### 5.3.2 Prompt Builder

Constructs a structured prompt to send to Groq. The prompt is designed to produce **deterministic, parseable JSON output** while leveraging the LLM's reasoning capabilities for ranking and explanation.

**Prompt Structure (Detailed):**

```
┌─────────────────────────────────────────────────────────────────┐
│ SYSTEM MESSAGE                                                   │
│                                                                   │
│ Role: "You are a restaurant recommendation assistant             │
│        specializing in Indian cities."                            │
│                                                                   │
│ Instructions:                                                     │
│ - Rank restaurants ONLY from the CANDIDATES list                 │
│ - Do NOT invent or hallucinate restaurants                       │
│ - Return valid JSON matching the specified schema                │
│ - Consider all user preferences when ranking                     │
│ - Provide a specific explanation for each recommendation         │
│ - Explanations should reference the user's stated preferences    │
│ - Write a brief summary of the overall recommendations          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ USER MESSAGE                                                     │
│                                                                   │
│ [User Preferences]                                               │
│ {                                                                 │
│   "location": "Bangalore",                                       │
│   "budget": "medium",                                            │
│   "cuisine": "Italian",                                          │
│   "min_rating": 4.0,                                             │
│   "additional": "family-friendly, outdoor seating"               │
│ }                                                                 │
│                                                                   │
│ [Candidates] (compact JSON, ~15–20 restaurants)                  │
│ [                                                                 │
│   { "id": "R001", "name": "...", "location": "...",             │
│     "cuisines": [...], "cost_for_two": 1200, "rating": 4.5 },   │
│   ...                                                             │
│ ]                                                                 │
│                                                                   │
│ [Task]                                                            │
│ Return the top 5 restaurants as JSON:                             │
│ {                                                                 │
│   "summary": "Brief overall summary...",                         │
│   "recommendations": [                                            │
│     {                                                             │
│       "id": "R001",                                              │
│       "rank": 1,                                                  │
│       "explanation": "Why this restaurant fits..."               │
│     }                                                             │
│   ]                                                               │
│ }                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Prompt Design Principles:**

| Principle | Implementation | Rationale |
|-----------|---------------|-----------|
| **Grounding** | Instruct model to ONLY recommend from provided candidates | Prevents hallucinated restaurant names |
| **Structured output** | Request JSON with explicit schema in prompt | Enables reliable parsing via `ResponseParser` |
| **ID preservation** | Include `restaurant.id` in candidates | Enables enrichment: join LLM explanations with full records |
| **Soft signals** | Pass `additional` preferences as natural language | LLM can reason about "family-friendly" without explicit data column |
| **Compact candidates** | Serialize only essential fields (id, name, cuisines, cost, rating) | Minimize token usage while preserving ranking-relevant data |
| **Task specificity** | Explicit top-K count and output schema | Reduces variance in LLM output format |

---

### 5.4 Recommendation Engine (Groq LLM Layer)

**Responsibility:** Invoke the Groq API, handle retries/fallbacks, parse and validate the JSON response, and merge LLM-generated rankings/explanations with full structured restaurant data.

#### Components

| Component | Role | Input → Output |
|-----------|------|----------------|
| `LLMClient` | Thin adapter over the Groq API via the official `groq` Python SDK; handles retries, model fallback, error classification | `(messages, config)` → `raw JSON string` |
| `RecommendationService` | **Orchestrator** — coordinates the full pipeline: filter → prompt → LLM → parse → enrich | `UserPreferences` → `RecommendationResponse` |
| `ResponseParser` | Parses raw JSON string from Groq; validates against expected schema; handles malformed/partial output | `raw JSON string` → `parsed dict` |
| `RecommendationEnricher` | Joins Groq LLM ranks/explanations with full restaurant records from the repository (adds cuisine, cost, rating, etc.) | `(parsed dict, candidates)` → `list[Recommendation]` |

#### Output Models

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Recommendation:
    rank: int
    name: str
    cuisine: str           # joined cuisine string for display (e.g. "Italian, Continental")
    rating: float
    estimated_cost: int    # cost_for_two in INR
    explanation: str       # Groq LLM-generated explanation

@dataclass
class RecommendationMetadata:
    candidates_considered: int   # how many restaurants passed deterministic filters
    filters_applied: dict        # the actual filter values used
    model: str                   # Groq model ID (e.g. "llama-3.3-70b-versatile")
    latency_ms: float            # wall-clock time for Groq API call
    tokens_used: int             # total tokens consumed (prompt + completion)
    fallback_used: bool          # whether heuristic fallback was triggered
    constraints_relaxed: list    # which constraints were relaxed, if any

@dataclass
class RecommendationResponse:
    summary: Optional[str]                  # Groq LLM-generated summary of recommendations
    recommendations: list[Recommendation]   # ranked list
    metadata: RecommendationMetadata        # operational metadata for logging/debugging
```

#### Reliability Patterns

| Pattern | Trigger | Behavior | Max Attempts |
|---------|---------|----------|--------------|
| **JSON mode** | Every request | Use `response_format={"type": "json_object"}` | — |
| **Temperature retry** | Invalid JSON from Groq | Retry same model with `temperature=0.1` | 1 retry |
| **Model fallback** | Primary model fails after retry | Switch to `llama-3.1-8b-instant` | 1 attempt |
| **Exponential backoff** | Groq 429 rate limit | Wait 1s, 2s, 4s between retries with jitter | 3 retries |
| **Heuristic fallback** | All LLM attempts fail | Return top-K by rating desc with generic explanation | Terminal fallback |
| **Idempotency** | By design | Same preferences + same dataset snapshot → same candidate set | — |

#### Heuristic Fallback Ranking

When the LLM is completely unavailable, the system degrades gracefully:

```python
def heuristic_fallback(candidates: list[Restaurant], top_k: int = 5) -> list[Recommendation]:
    """Fallback when Groq LLM is unavailable."""
    sorted_candidates = sorted(candidates, key=lambda r: (-r.rating, -r.votes))
    return [
        Recommendation(
            rank=i + 1,
            name=r.name,
            cuisine=", ".join(r.cuisines),
            rating=r.rating,
            estimated_cost=r.cost_for_two,
            explanation=f"Ranked #{i+1} based on rating ({r.rating}⭐) and popularity ({r.votes} votes). "
                        f"AI-powered explanation is currently unavailable."
        )
        for i, r in enumerate(sorted_candidates[:top_k])
    ]
```

---

### 5.5 Output Display Layer

**Responsibility:** Render `RecommendationResponse` in a clear, scannable, visually appealing format.

#### Components

| Component | Role | Input → Output |
|-----------|------|----------------|
| `RecommendationPresenter` | Formats `RecommendationResponse` for the target rendering context (UI vs CLI) | `RecommendationResponse` → formatted output |
| `ResultsView` | Renders individual recommendation cards with rank badge, name, cuisine, rating, cost, and AI explanation | `list[Recommendation]` → rendered cards |
| `SummaryBanner` | Renders the optional Groq LLM-generated summary at the top of results | `str` → header banner |

#### Result Card Layout

Each recommendation card displays:

```
┌─────────────────────────────────────────────────────────────┐
│  🏆 #1                                                       │
│                                                               │
│  🍽️  Restaurant Name                                         │
│  🍜  Italian, Continental                                     │
│  ⭐  4.5 / 5.0                                               │
│  💰  ₹1,200 for two                                          │
│                                                               │
│  🤖 AI Recommendation:                                       │
│  "This highly rated Italian restaurant fits perfectly         │
│   within your medium budget. Known for its family-friendly    │
│   atmosphere and spacious outdoor seating area."              │
└─────────────────────────────────────────────────────────────┘
```

#### UX Requirements

| Requirement | Description |
|-------------|-------------|
| **Applied filters banner** | Show location, budget, cuisine, min_rating above results so user sees what was searched |
| **No results state** | Display friendly message with suggestions: "Try broadening your budget or removing cuisine filter" |
| **Loading state** | Show spinner/progress while dataset loads or Groq LLM responds |
| **Fallback indicator** | If AI explanation is unavailable, show a subtle badge: "📊 Ranked by rating (AI unavailable)" |
| **Constraint relaxation notice** | If filters were relaxed, show: "⚠️ Showing all cuisines (no Italian restaurants found in your budget)" |
| **Metadata footer** | Optional: show candidates considered, model used, response time |

---

## 6. Request Flow — Full Sequence

```
User
  │
  │  1. Enter preferences:
  │     Location: "Bangalore"
  │     Budget: "medium"
  │     Cuisine: "Italian"
  │     Min Rating: 4.0
  │     Additional: "family-friendly, outdoor seating"
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ PRESENTATION LAYER                                               │
│  Streamlit form / CLI prompt                                     │
│  → Collect raw input → Submit                                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Raw input dict
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ PreferenceValidator                                              │
│  ✓ location is non-empty and exists in dataset                  │
│  ✓ budget is "low" | "medium" | "high"                          │
│  ✓ min_rating is float in [0.0, 5.0]                            │
│  ✓ cuisine fuzzy-matches known vocabulary (optional)             │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Validated dict
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ PreferenceNormalizer                                             │
│  → location: "bangalore" → "Bangalore" (title-case + alias)    │
│  → cuisine: "italian" → "Italian" (title-case)                  │
│  → additional: trimmed whitespace                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │ UserPreferences (clean)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ RestaurantRepository.get_all()                                   │
│  → Returns full cached dataset (~50K+ records)                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ list[Restaurant]
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ RestaurantFilter                                                 │
│  → location="Bangalore": ~1,200 remain                          │
│  → budget="medium" (501–1500): ~400 remain                      │
│  → min_rating=4.0: ~120 remain                                  │
│  → cuisine="Italian": ~18 remain                                 │
│  → sort by rating desc, votes desc                               │
│  → take top 20                                                   │
│  (If 0 remain → relax cuisine → budget → rating)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │ candidates[] (top 18–20)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ PromptBuilder                                                    │
│  → System: role + JSON format + ranking rules                   │
│  → User: preferences + candidates JSON + task                   │
│  → ~1,500–2,500 tokens total                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Structured prompt (messages list)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ LLMClient → Groq API                                             │
│  → model: "llama-3.3-70b-versatile"                             │
│  → temperature: 0.3                                              │
│  → response_format: {"type": "json_object"}                     │
│  → Latency: ~200–500ms (Groq LPU)                              │
│                                                                   │
│  ┌─ On failure: retry temp=0.1 → fallback model → heuristic    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Raw JSON response from Groq
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ ResponseParser                                                   │
│  → Parse JSON string → validate schema                          │
│  → Extract: summary, recommendations[{id, rank, explanation}]   │
│  → On invalid JSON: trigger retry or fallback                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Parsed recommendation dict
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ RecommendationEnricher                                           │
│  → Join LLM ranks/explanations with full Restaurant records     │
│  → Add: cuisine string, rating, cost_for_two from repository    │
│  → Build: RecommendationResponse with metadata                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ RecommendationResponse
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ PRESENTATION LAYER                                               │
│  → Render SummaryBanner (AI summary)                            │
│  → Render applied filters badge                                  │
│  → Render 5 ResultsView cards (rank, name, cuisine, rating,    │
│    cost, AI explanation)                                         │
│  → Show metadata footer (candidates, model, latency)            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
                         User sees personalized recommendations
```

---

## 7. API Design (Optional REST Layer)

For deployments where the UI is decoupled from the backend, FastAPI provides a lightweight REST interface.

### Endpoints

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/api/v1/recommend` | `POST` | Primary recommendation endpoint | API key (optional) |
| `/api/v1/health` | `GET` | Service status + dataset load state + Groq connectivity | None |
| `/api/v1/locations` | `GET` | Distinct locations from dataset (for UI dropdown population) | None |
| `/api/v1/cuisines` | `GET` | Distinct cuisines from dataset (for UI dropdown population) | None |
| `/api/v1/stats` | `GET` | Dataset statistics: total restaurants, location distribution, rating distribution | None |

### `POST /api/v1/recommend` — Request/Response

**Request (Pydantic model: `RecommendRequest`):**
```json
{
  "location": "Bangalore",
  "budget": "medium",
  "cuisine": "Italian",
  "min_rating": 4.0,
  "additional": "family-friendly, outdoor seating"
}
```

**Response (Pydantic model: `RecommendResponse`):**
```json
{
  "summary": "Based on your preference for Italian cuisine in Bangalore with a medium budget, here are 5 top-rated restaurants that offer family-friendly dining with outdoor seating options...",
  "recommendations": [
    {
      "rank": 1,
      "name": "Trattoria Milano",
      "cuisine": "Italian, Continental",
      "rating": 4.5,
      "estimated_cost": 1200,
      "explanation": "Highest-rated Italian restaurant in Bangalore within your budget. Known for spacious outdoor seating and a welcoming family atmosphere with a dedicated kids' menu."
    },
    {
      "rank": 2,
      "name": "La Piazza",
      "cuisine": "Italian",
      "rating": 4.3,
      "estimated_cost": 1100,
      "explanation": "Authentic Italian dining at an excellent price point. The garden seating area is perfect for families, and their wood-fired pizzas are a crowd favorite."
    }
  ],
  "metadata": {
    "candidates_considered": 18,
    "filters_applied": {
      "location": "Bangalore",
      "budget": "medium",
      "min_rating": 4.0,
      "cuisine": "Italian"
    },
    "model": "llama-3.3-70b-versatile",
    "latency_ms": 342,
    "tokens_used": 1847,
    "fallback_used": false,
    "constraints_relaxed": []
  }
}
```

### `GET /api/v1/health`

```json
{
  "status": "healthy",
  "dataset_loaded": true,
  "dataset_size": 51717,
  "groq_reachable": true,
  "groq_model": "llama-3.3-70b-versatile",
  "uptime_seconds": 3600
}
```

### Error Responses

| HTTP Status | Scenario | Response Body |
|-------------|----------|---------------|
| `400` | Invalid input (missing location, bad budget enum) | `{"error": "validation_error", "details": [...]}` |
| `404` | Location not found in dataset | `{"error": "location_not_found", "suggestions": ["Bangalore", "Bengaluru"]}` |
| `503` | Groq unavailable + heuristic fallback used | Response includes `"fallback_used": true` + warning |
| `500` | Unexpected server error | `{"error": "internal_error", "request_id": "..."}` |

---

## 8. Proposed Module Structure

```
zomato-milestone1/
├── docs/
│   ├── context.md                  ← Product requirements and workflow
│   ├── architecture.md             ← This document
│   └── problemStatement.txt        ← Original problem statement
│
├── src/
│   ├── __init__.py
│   ├── main.py                     # Entry point (CLI or app bootstrap)
│   ├── config.py                   # Env vars, budget thresholds, Groq settings
│   │
│   ├── models/                     # Data transfer objects (dataclasses)
│   │   ├── __init__.py
│   │   ├── restaurant.py           # Restaurant dataclass + budget tier enum
│   │   ├── preferences.py          # UserPreferences dataclass + validation
│   │   └── recommendation.py       # Recommendation, RecommendationResponse, Metadata
│   │
│   ├── data/                       # Data ingestion layer
│   │   ├── __init__.py
│   │   ├── loader.py               # Hugging Face dataset loader + cache management
│   │   ├── preprocessor.py         # Column mapping, normalization, budget derivation
│   │   └── repository.py           # In-memory query interface (get_all, get_locations, etc.)
│   │
│   ├── services/                   # Business logic layer
│   │   ├── __init__.py
│   │   ├── filter.py               # RestaurantFilter + CandidateSelector + constraint relaxation
│   │   ├── prompt_builder.py       # PromptBuilder (crafts system/user messages for Groq)
│   │   ├── llm_client.py           # Groq API adapter (retry, fallback, error handling)
│   │   ├── response_parser.py      # JSON parsing + schema validation from Groq output
│   │   └── recommendation.py       # RecommendationService orchestrator (main pipeline)
│   │
│   ├── api/                        # Optional REST layer
│   │   ├── __init__.py
│   │   ├── routes.py               # FastAPI routes (/recommend, /health, /locations, /cuisines)
│   │   ├── schemas.py              # Pydantic request/response models
│   │   └── middleware.py           # Rate limiting, error handling, request ID
│   │
│   └── ui/                         # Presentation layer
│       ├── __init__.py
│       ├── cli.py                  # Rich terminal interface (click or argparse)
│       └── streamlit_app.py        # Streamlit web UI with form + result cards
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Shared fixtures (frozen dataset, mock Groq client)
│   ├── test_preprocessor.py        # Cuisine parsing, numeric coercion, null handling
│   ├── test_filter.py              # Filter pipeline, constraint relaxation, edge cases
│   ├── test_prompt_builder.py      # Prompt structure, candidate serialization
│   ├── test_response_parser.py     # Valid/invalid JSON, schema validation
│   ├── test_recommendation.py      # Full pipeline with mocked Groq client
│   └── fixtures/
│       └── sample_restaurants.json # Frozen 10–20 row dataset for deterministic tests
│
├── data/                           # Cached dataset (gitignored)
│   └── zomato_cache.parquet
│
├── .env.example                    # Template: GROQ_API_KEY=your_key_here
├── .gitignore                      # Includes: .env, data/, __pycache__/, *.pyc
├── requirements.txt                # All Python dependencies with pinned versions
├── requirements-dev.txt            # Dev dependencies: pytest, black, flake8, mypy
└── README.md                       # Setup guide, usage, architecture overview
```

---

## 9. Technology Stack

| Layer | Technology | Version | Rationale |
|-------|-----------|---------|-----------|
| **Language** | Python | 3.11+ | Strong ecosystem for data processing + LLM integration; type hints, dataclasses |
| **Dataset** | `datasets` (Hugging Face) | Latest | Direct, versioned access to the specified Zomato dataset |
| **Data processing** | `pandas` | 2.x | Filtering, normalization, caching; DataFrame operations for tabular data |
| **LLM Provider** | **Groq** | — | Ultra-low-latency inference via LPU hardware; see Section 3 |
| **LLM Model** | `llama-3.3-70b-versatile` | — | 70B parameter open-weight model; strong reasoning for ranking + explanation |
| **LLM SDK** | `groq` | Latest | Official Groq Python client (`pip install groq`) |
| **Config** | `pydantic-settings` + `.env` | v2.x | Typed config with validation, automatic env var loading |
| **API (optional)** | FastAPI | 0.100+ | Lightweight async REST; auto-generated OpenAPI docs |
| **UI (optional)** | Streamlit | 1.x | Rapid prototyping of form + result cards; live reload |
| **Testing** | `pytest` | 8.x | Unit + integration tests; fixtures, parametrize, mock |
| **Linting** | `black` + `flake8` + `mypy` | — | Consistent formatting, style checking, static type analysis |

### `requirements.txt` (Estimated)

```
groq>=0.5.0
datasets>=2.14.0
pandas>=2.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
fastapi>=0.100.0          # optional
uvicorn>=0.23.0           # optional
streamlit>=1.28.0         # optional
```

---

## 10. Configuration Reference (`config.py`)

All configuration is centralized in `config.py` using `pydantic-settings` for type safety and automatic `.env` loading.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HF_DATASET_NAME` | `str` | `ManikaSaini/zomato-restaurant-recommendation` | Hugging Face dataset identifier |
| `HF_DATASET_SPLIT` | `str` | `train` | Dataset split to load |
| `BUDGET_THRESHOLDS` | `dict` | `{"low": 500, "medium": 1500}` | Cost-for-two INR cutoffs for budget tier derivation |
| `MAX_CANDIDATES_FOR_LLM` | `int` | `20` | Maximum restaurants sent to Groq in a single prompt |
| `TOP_K_RECOMMENDATIONS` | `int` | `5` | Number of top results returned to the user |
| `GROQ_API_KEY` | `str` | `$env` | Groq API key (**required**; from `.env`, never hardcoded) |
| `GROQ_MODEL` | `str` | `llama-3.3-70b-versatile` | Primary Groq model for ranking + explanations |
| `GROQ_FALLBACK_MODEL` | `str` | `llama-3.1-8b-instant` | Fallback Groq model (faster, cheaper, for dev/testing or primary failure) |
| `GROQ_TEMPERATURE` | `float` | `0.3` | Inference temperature; retried at `0.1` on parse failure |
| `GROQ_RETRY_TEMPERATURE` | `float` | `0.1` | Temperature used on JSON parse failure retry |
| `GROQ_MAX_TOKENS` | `int` | `2048` | Max output tokens per Groq request |
| `GROQ_MAX_RETRIES` | `int` | `3` | Max retry attempts for rate limit errors |
| `DATA_CACHE_PATH` | `str` | `./data/zomato_cache.parquet` | Local dataset cache file path |
| `LOG_LEVEL` | `str` | `INFO` | Logging verbosity |

### Configuration Implementation

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Hugging Face
    HF_DATASET_NAME: str = "ManikaSaini/zomato-restaurant-recommendation"
    HF_DATASET_SPLIT: str = "train"

    # Budget
    BUDGET_LOW_MAX: int = 500
    BUDGET_MEDIUM_MAX: int = 1500

    # Candidates & Results
    MAX_CANDIDATES_FOR_LLM: int = 20
    TOP_K_RECOMMENDATIONS: int = 5

    # Groq
    GROQ_API_KEY: str          # Required — no default
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_FALLBACK_MODEL: str = "llama-3.1-8b-instant"
    GROQ_TEMPERATURE: float = 0.3
    GROQ_RETRY_TEMPERATURE: float = 0.1
    GROQ_MAX_TOKENS: int = 2048
    GROQ_MAX_RETRIES: int = 3

    # Data Cache
    DATA_CACHE_PATH: str = "./data/zomato_cache.parquet"

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

---

## 11. Error Handling — Comprehensive Matrix

| Scenario | Detection | Behavior | Fallback | User-Facing Message |
|----------|-----------|----------|----------|---------------------|
| **Dataset download fails** | `ConnectionError`, `HTTPError` from `datasets` | Retry with exponential backoff (3 attempts) | Load from local parquet cache if available | "Loading restaurant data... (using cached data)" |
| **Dataset cache missing** | `FileNotFoundError` on cache path | Force download from Hugging Face | Show error if download also fails | "Unable to load restaurant data. Please check your connection." |
| **No restaurants match filters** | Empty result from `RestaurantFilter` | Relax constraints in order: cuisine → budget → rating | Show results with relaxation notice | "⚠️ Showing all cuisines (no Italian restaurants found in your budget)" |
| **Location not in dataset** | Location not found in repository's location set | Suggest closest matches via fuzzy matching | None — user must re-enter | "Location not found. Did you mean: Bangalore, Bengaluru?" |
| **Groq returns invalid JSON** | `json.JSONDecodeError` in `ResponseParser` | Retry once with `temperature=0.1` | Try fallback model → heuristic ranking | "Generating recommendations..." (transparent to user) |
| **Groq returns wrong schema** | Schema validation failure in `ResponseParser` | Same as invalid JSON | Same | Transparent to user |
| **Groq 429 rate limit** | `groq.RateLimitError` or HTTP 429 | Exponential backoff with jitter (1s, 2s, 4s) up to 3 retries | Fallback model → heuristic ranking | "📊 Recommendations ranked by rating (AI explanation temporarily unavailable)" |
| **Groq timeout** | `groq.APITimeoutError` | Same as 429 handling | Same | Same |
| **Groq server error (500/503)** | `groq.InternalServerError` | Retry once, then fallback | Heuristic ranking | Same as 429 |
| **GROQ_API_KEY missing** | `ValueError` at startup in `config.py` | Fail fast at startup | None — cannot start without key | "Error: GROQ_API_KEY not set. See .env.example" |
| **Empty dataset** | Zero rows after preprocessing | Show clear error | None | "Dataset contains no valid restaurant records." |

---

## 12. Cross-Cutting Concerns

### 12.1 Logging & Observability

| What to Log | Where | Format |
|-------------|-------|--------|
| Dataset load: row count, time taken | `DatasetLoader` | `INFO: Loaded 51,717 restaurants in 2.3s` |
| Filter pipeline: count at each stage | `RestaurantFilter` | `INFO: location=1,200 → budget=400 → rating=120 → cuisine=18` |
| Constraint relaxation events | `RestaurantFilter` | `WARNING: Relaxed cuisine filter (0 results with "Italian")` |
| Groq request: model, prompt tokens | `LLMClient` | `INFO: Groq request model=llama-3.3-70b-versatile prompt_tokens=1,423` |
| Groq response: completion tokens, latency | `LLMClient` | `INFO: Groq response completion_tokens=824 latency_ms=312` |
| Groq retry/fallback events | `LLMClient` | `WARNING: Groq 429 — retrying in 2s (attempt 2/3)` |
| Model switch events | `LLMClient` | `WARNING: Switched to fallback model llama-3.1-8b-instant` |
| Heuristic fallback activation | `RecommendationService` | `WARNING: All LLM attempts failed — using heuristic ranking` |
| Parse failures | `ResponseParser` | `ERROR: Failed to parse Groq JSON — retrying with temp=0.1` |

**Do NOT log:**
- Full prompt text (may contain user preferences — privacy)
- `GROQ_API_KEY` or any credentials
- Full Groq response body (large, and may contain PII in explanations)

### 12.2 Security

| Concern | Mitigation |
|---------|------------|
| **API key exposure** | `GROQ_API_KEY` stored in `.env` (gitignored); `.env.example` provides template with placeholder |
| **Prompt injection** | Validate and sanitize `additional` field before embedding in prompt; limit to 500 chars |
| **User input validation** | All inputs validated by `PreferenceValidator` before use; enums enforced, numeric bounds checked |
| **Rate limiting** | If API is deployed publicly, apply per-IP rate limits via FastAPI middleware |
| **Error detail leakage** | Never expose internal stack traces or Groq error details to end users |
| **Dependency security** | Pin dependency versions in `requirements.txt`; audit with `pip-audit` |

### 12.3 Performance Considerations

| Stage | Expected Latency | Optimization |
|-------|-------------------|-------------|
| Dataset load (cold start) | 5–15s from Hugging Face | Cache to parquet; load from cache on subsequent starts |
| Dataset load (warm start) | 0.5–2s from parquet | Pre-load at container startup |
| Filter pipeline | <10ms | In-memory pandas operations on cached DataFrame |
| Prompt construction | <5ms | String formatting of pre-filtered candidates |
| Groq LLM inference | 200–500ms | Groq LPU hardware; pre-filtering reduces token count |
| Response parsing | <5ms | `json.loads()` on structured response |
| **Total (warm, happy path)** | **~300–600ms** | Sub-second end-to-end for interactive use |

---

## 13. Testing Strategy

### Test Matrix

| Test Type | Scope | Component | Example Test Case |
|-----------|-------|-----------|-------------------|
| **Unit** | Isolated | `DataPreprocessor` | Cuisine string `"Italian, Chinese"` → `["Italian", "Chinese"]` |
| **Unit** | Isolated | `DataPreprocessor` | Non-numeric rating rows are dropped |
| **Unit** | Isolated | `DataPreprocessor` | Location alias `"Bengaluru"` → `"Bangalore"` |
| **Unit** | Isolated | `RestaurantFilter` | Location filter returns only matching restaurants |
| **Unit** | Isolated | `RestaurantFilter` | Budget filter maps "medium" to correct `cost_for_two` range |
| **Unit** | Isolated | `RestaurantFilter` | Constraint relaxation: drops cuisine when 0 results |
| **Unit** | Isolated | `PreferenceValidator` | Rejects budget value "expensive" (not in enum) |
| **Unit** | Isolated | `PreferenceValidator` | Clamps min_rating 6.0 to 5.0 with warning |
| **Unit** | Isolated | `ResponseParser` | Valid Groq JSON → parsed recommendations dict |
| **Unit** | Isolated | `ResponseParser` | Invalid JSON → raises `ParseError` |
| **Unit** | Isolated | `ResponseParser` | Missing `explanation` field → schema validation error |
| **Unit** | Isolated | `PromptBuilder` | Prompt includes all candidate IDs and preference fields |
| **Unit** | Isolated | `PromptBuilder` | Prompt token count stays under budget for 20 candidates |
| **Integration** | Service | `RecommendationService` | Mock Groq client returns fixed JSON; verify full enriched output |
| **Integration** | Service | `RecommendationService` | Mock Groq client fails; verify heuristic fallback activates |
| **Integration** | Pipeline | Full pipeline | Frozen dataset + mock Groq → deterministic `RecommendationResponse` |
| **Snapshot** | Contract | `PromptBuilder` | Prompt text matches expected snapshot (detect prompt regressions) |

### Test Infrastructure

- **Fixtures:** Frozen 10–20 row dataset in `tests/fixtures/sample_restaurants.json`
- **Mocking:** Mock `groq.Groq` client to avoid real API calls in CI/CD
- **Parametrize:** Use `@pytest.mark.parametrize` for filter combinations and edge cases
- **Coverage target:** ≥80% line coverage for `services/` and `data/` modules
- **CI command:** `pytest tests/ -v --cov=src --cov-report=term-missing`

---

## 14. Deployment Topology

### 14.1 Development (Local)

```
Developer Machine
├── Python 3.11+ virtual environment
├── src/main.py (Streamlit / CLI entry point)
├── data/zomato_cache.parquet (local cache)
├── .env (GROQ_API_KEY)
└── Groq API (cloud, via HTTPS)
```

**Quick start:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # Add your GROQ_API_KEY
streamlit run src/ui/streamlit_app.py
```

### 14.2 Minimal Production

```
                    ┌─────────────────────────────┐
                    │         User Browser         │
                    └──────────────┬───────────────┘
                                   │ HTTPS
                                   ▼
                    ┌─────────────────────────────┐
                    │   Streamlit / Static Frontend │
                    │   (or Reverse Proxy / CDN)    │
                    └──────────────┬───────────────┘
                                   │ HTTP/WebSocket
                                   ▼
                    ┌─────────────────────────────┐
                    │      FastAPI Backend          │
                    │  (uvicorn, single instance)   │
                    │                               │
                    │  ┌─ Dataset (pre-loaded)      │
                    │  │  at container startup       │
                    │  │                             │
                    │  └─ In-memory DataFrame       │
                    └──────────────┬───────────────┘
                                   │ HTTPS
                                   ▼
                    ┌─────────────────────────────┐
                    │       Groq API (Cloud)        │
                    │   llama-3.3-70b-versatile     │
                    └─────────────────────────────┘
```

- **Pre-load dataset** at container startup for fast first-request response
- **Single stateless API instance** is sufficient for milestone scope
- **Scale horizontally** later by sharing a read-only dataset snapshot via shared volume or S3
- **Health endpoint** (`/api/v1/health`) for load balancer probes

---

## 15. Implementation Phases

| Phase | Deliverable | Key Activities | Dependencies |
|-------|-------------|----------------|-------------|
| **Phase 1 — Data** | Dataset pipeline | Load Hugging Face dataset, implement preprocessing, cache to parquet, expose `RestaurantRepository` with `get_all()`, `get_locations()`, `get_cuisines()` | None |
| **Phase 2 — Filter** | Preference & filtering | Implement `UserPreferences` model, `PreferenceValidator`, `PreferenceNormalizer`, `RestaurantFilter` with constraint relaxation, `CandidateSelector` | Phase 1 |
| **Phase 3 — Groq LLM** | LLM integration | Build `PromptBuilder`, `LLMClient` (Groq adapter with retry/fallback), `ResponseParser`, `RecommendationEnricher`, `RecommendationService` orchestrator | Phases 1, 2 |
| **Phase 4 — UI** | User interface | CLI interface (click/argparse) and/or Streamlit web UI with preference form + ranked result cards + summary banner | Phases 1–3 |
| **Phase 5 — Hardening** | Production readiness | Error handling, heuristic fallback, logging/observability, pytest suite (unit + integration), README, `.env.example`, `requirements.txt` | Phases 1–4 |

---

## 16. Architecture Decisions — Decision Log

| Decision | Choice | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| **LLM Provider** | **Groq** (`llama-3.3-70b-versatile`) | OpenAI GPT-4, Anthropic Claude, local models via Ollama | Sub-second latency via LPU hardware; open-weight models reduce vendor lock-in; JSON mode support; cost-effective |
| **Pre-filter before LLM** | Yes — hard filters in Python code | Let Groq filter entire dataset from raw input | LLM filtering entire dataset is expensive (~50K records × tokens), unreliable (hallucination), and slow |
| **LLM Output Format** | Structured JSON via `response_format` | Free-form natural language text | JSON is parseable, testable, mappable to typed models; reduces post-processing ambiguity |
| **Data Storage** | In-memory pandas DataFrame | PostgreSQL, SQLite, or other database | Read-only milestone dataset doesn't need persistence layer complexity; DataFrame is fast for filter/sort |
| **Ranking Split** | Heuristic shortlist (top 15–20) + LLM final rank (top 5) | Pure LLM ranking (send all) or pure heuristic (no LLM) | Best of both: deterministic efficiency for narrowing + LLM reasoning quality for final ranking and explanation |
| **UI Approach** | Streamlit for rapid prototyping | React SPA, Next.js, or other frontend framework | Fastest path to interactive demo for milestone 1; extensible architecture allows swap later |
| **Model fallback** | Automatic fallback to `llama-3.1-8b-instant` | Fail hard on primary model failure | Graceful degradation improves reliability; 8B model is still capable for ranking |
| **Heuristic fallback** | Sort by rating desc, generic explanation | No fallback (error page) | Users should always see results, even without AI explanations |
| **Caching** | Local parquet file | Redis, Memcached, or cloud storage | Simplest caching for single-instance deployment; parquet is efficient for tabular data |
| **Testing** | Mocked Groq client + frozen fixtures | Live API calls in tests | Deterministic, fast, no API cost; real integration tested manually |

---

## 17. Related Documents

- [`context.md`](./context.md) — Product requirements, user input specification, success criteria, and full workflow
- [`problemStatement.txt`](./problemStatement.txt) — Original problem statement defining objectives and system workflow

---

*Architecture Version: 2.0 | LLM Provider: Groq (`llama-3.3-70b-versatile`) | Last Updated: 2026-06-21*
