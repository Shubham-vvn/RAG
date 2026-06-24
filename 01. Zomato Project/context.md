# 📋 Project Context: AI-Powered Restaurant Recommendation System (Zomato Use Case)

> **Related Documents:**
> - [`architecture.md`](./architecture.md) — Full technical architecture and component design
> - [`problemStatement.txt`](./problemStatement.txt) — Original problem statement

---

## 1. Project Overview

This project involves designing and building a **production-grade, AI-powered restaurant recommendation service** inspired by Zomato. The system bridges structured restaurant data (sourced from Hugging Face) with the advanced reasoning capabilities of **Groq's LLM inference engine** to deliver personalized, explainable restaurant suggestions in near real-time.

Unlike a simple filter-and-list approach, this system:
- **Understands user intent** via structured preference input and optional free-text signals
- **Deterministically pre-filters** candidates using hard constraints (location, budget, rating, cuisine) before involving the LLM — reducing token cost, latency, and hallucination risk
- **Reasons, ranks, and explains** options using Groq's `llama-3.3-70b-versatile` model running on Groq's Language Processing Unit (LPU) hardware
- **Provides explainable AI** — every recommendation includes a natural-language rationale tied directly to the user's stated preferences
- **Presents results** in a clean, scannable, card-based format with applied-filter context and graceful fallback states

### Why This Matters

Traditional restaurant recommendation systems rely purely on heuristic scoring (rating, popularity, proximity). By integrating an LLM, this system can:
- Consider **soft preferences** like "family-friendly" or "outdoor seating" that aren't easily encoded as database filters
- Generate **human-readable explanations** for why each restaurant was chosen
- **Adapt ranking logic** based on nuanced combinations of preferences that a rule-based system would struggle with

---

## 2. Problem Statement

> *"Build an application that takes user preferences (such as location, budget, cuisine, and ratings), uses a real-world dataset of restaurants, and leverages an LLM to generate personalized, human-like recommendations with clear, useful results."*

### Core Objectives

| # | Objective | Description |
|---|-----------|-------------|
| 1 | **Data Ingestion** | Load and preprocess the Zomato dataset from Hugging Face, extracting restaurant name, location, cuisine, cost, rating, and other relevant fields |
| 2 | **User Input Collection** | Collect structured preferences (location, budget, cuisine, min rating) plus optional free-text preferences (e.g., "quick service", "romantic ambiance") |
| 3 | **Integration Layer** | Filter and prepare relevant restaurant data based on user input; pass structured results into an LLM prompt designed to help the model reason and rank |
| 4 | **Recommendation Engine** | Use the LLM to rank restaurants, provide per-recommendation explanations (why each fits), and optionally summarize the overall set of choices |
| 5 | **Output Display** | Present top recommendations in a user-friendly format showing restaurant name, cuisine, rating, estimated cost, and AI-generated explanation |

### System Workflow (High-Level)

```
1. Data Ingestion
   └─► Load Zomato dataset from Hugging Face
   └─► Preprocess: normalize fields, parse cuisines, derive budget tiers
   └─► Cache in-memory (+ local parquet snapshot for dev)

2. User Input
   └─► Collect: Location, Budget, Cuisine, Min Rating, Additional Preferences
   └─► Validate: required fields, enum values, rating bounds
   └─► Normalize: lowercase, trim, city alias mapping

3. Integration Layer
   └─► Apply deterministic filters (location → budget → rating → cuisine)
   └─► Select top N candidates (15–20) sorted by rating desc, votes desc
   └─► Build structured LLM prompt with candidates + user preferences

4. Recommendation Engine (Groq LLM)
   └─► Send prompt to Groq API (llama-3.3-70b-versatile)
   └─► Parse JSON response; validate schema
   └─► Enrich with full restaurant records from structured data

5. Output Display
   └─► Render ranked recommendation cards with AI explanations
   └─► Show applied filters, summary banner, and fallback states
```

---

## 3. Architecture Goals

| Goal | Description | Implementation Strategy |
|------|-------------|------------------------|
| **Separation of concerns** | Data loading, filtering, LLM reasoning, and presentation are isolated modules with clear interfaces | Layered architecture with distinct data/service/presentation tiers |
| **Deterministic pre-filtering** | Hard constraints (location, budget, rating) are applied *before* the LLM to reduce token cost and hallucination risk | Python-side pandas/list filtering before prompt construction |
| **Explainability** | Every recommendation includes a Groq LLM-generated rationale tied to user preferences | Prompt instructs LLM to explain each pick; explanations stored per-recommendation |
| **Extensibility** | Swap UI frameworks or data sources without rewriting core logic; LLM access is isolated behind a Groq adapter | Adapter pattern for LLM client; repository pattern for data access |
| **Testability** | Pure functions for filtering/ranking prep; mockable LLM adapter for unit tests | Dependency injection for Groq client; frozen test fixtures |
| **Reliability** | Graceful degradation when LLM fails or returns invalid output | Retry with lower temperature → fallback heuristic ranking → clear user messaging |

---

## 4. Dataset

- **Source:** [Hugging Face — ManikaSaini/zomato-restaurant-recommendation](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation)
- **Access:** Via `datasets` Python library (Hugging Face)
- **Split:** Typically `train`
- **Caching:** Load once at startup; persist a local parquet/CSV snapshot to `./data/zomato_cache.parquet` to avoid repeated Hugging Face downloads during development

### Canonical Restaurant Schema

After preprocessing, each restaurant is normalized to this canonical format:

```python
Restaurant = {
    "id": str,              # stable identifier (index or dataset id)
    "name": str,
    "location": str,        # city / locality
    "cuisines": list[str],  # e.g. ["Italian", "Continental"]
    "cost_for_two": int,    # numeric cost indicator (INR)
    "rating": float,        # e.g. 4.2
    "votes": int,           # optional: popularity signal
    "rest_type": str,       # optional: casual dining, cafe, etc.
}
```

### Raw-to-Canonical Field Mapping

| Raw Field | Canonical Field | Transformation |
|-----------|----------------|----------------|
| `restaurant_name` | `name` | Direct mapping |
| `location` / `city` | `location` | Trim, title-case, alias map for city names |
| `cuisines` | `cuisines` | Split `"Italian, Chinese"` → `["Italian", "Chinese"]` |
| `average_cost_for_two` | `cost_for_two` | Coerce to int; drop/impute invalid |
| `aggregate_rating` | `rating` | Coerce to float; drop/impute invalid |
| `votes` | `votes` | Popularity signal for tie-breaking |
| `rest_type` | `rest_type` | Casual dining, café, etc. |

### Preprocessing Pipeline

1. Download dataset split (typically `train`)
2. Select and rename relevant columns to canonical schema
3. Parse cuisine strings into lists (`"Italian, Chinese"` → `["Italian", "Chinese"]`)
4. Coerce `rating` and `cost_for_two` to numeric types; drop or impute invalid rows
5. Normalize location strings (trim, title-case, alias map for city names)
6. Derive `budget_tier` from `cost_for_two` using configurable thresholds:

| Budget Tier | `cost_for_two` Range (INR) | `price_range` |
|-------------|---------------------------|---------------|
| `low` | ≤ ₹500 | 1 |
| `medium` | ₹501 – ₹1,500 | 2–3 |
| `high` | > ₹1,500 | 4 |

> ⚠️ Thresholds should be tuned after inspecting the actual dataset distribution. Configurable via `config.py`.

---

## 5. User Input Specification

### Input Model

```python
UserPreferences = {
    "location": str,           # required — city or locality name
    "budget": str,             # "low" | "medium" | "high"
    "cuisine": str | None,     # optional primary cuisine preference
    "min_rating": float,       # e.g. 3.5 — minimum acceptable rating
    "additional": str | None,  # free-text: "family-friendly, quick service"
}
```

### Input Fields

| Input | Type | Required | Examples | Purpose |
|-------|------|----------|---------|---------|
| **Location** | String | ✅ Yes | `"Delhi"`, `"Bangalore"`, `"Mumbai"` | Primary geographic filter |
| **Budget** | Enum | ✅ Yes | `"low"`, `"medium"`, `"high"` | Maps to `cost_for_two` ranges |
| **Cuisine** | String | ❌ No | `"Italian"`, `"Chinese"`, `"North Indian"` | Hard filter when provided |
| **Min Rating** | Float [0–5] | ✅ Yes | `3.5`, `4.0` | Minimum quality threshold |
| **Additional Preferences** | Free text | ❌ No | `"family-friendly"`, `"quick service"`, `"outdoor seating"` | Passed as soft signal to LLM for ranking/explanation |

### Validation Rules

| Field | Rule |
|-------|------|
| `location` | Non-empty; must match at least one value in the dataset (or suggest closest matches) |
| `budget` | Must be one of `low`, `medium`, `high` |
| `min_rating` | Float in `[0.0, 5.0]` |
| `cuisine` | Optional; fuzzy match against known cuisine vocabulary extracted from the dataset |
| `additional` | Optional free text; passed through to Groq LLM for soft matching |

### Input Components

| Component | Role |
|-----------|------|
| `PreferenceForm` | UI form or CLI prompt collecting all preference fields |
| `PreferenceValidator` | Enforces required fields, enum values, rating bounds |
| `PreferenceNormalizer` | Lowercases cuisine, maps city aliases, trims free text |

---

## 6. System Architecture

### End-to-End Data Flow

```
Hugging Face Dataset
        │
        ▼
  [Load & Preprocess] ──► RestaurantRepository (cached in-memory)
                                │
User Preferences ──► [Validate & Normalize] ──► [Deterministic Filter]
                                                        │
                                                        ▼
                                               [Build Groq Prompt]
                                                        │
                                                        ▼
                                         [Groq LLM — Rank + Explain]
                                         (llama-3.3-70b-versatile)
                                                        │
                                                        ▼
                                              [Parse & Enrich Response]
                                                        │
                                                        ▼
                                        RecommendationResponse ──► UI
```

### Component Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                         │
│          Streamlit UI / Gradio / FastAPI + CLI                 │
└──────────────────────────┬─────────────────────────────────────┘
                           │ User Preferences
                           ▼
┌────────────────────────────────────────────────────────────────┐
│                   USER INPUT LAYER                             │
│   PreferenceForm → PreferenceValidator → PreferenceNormalizer  │
└──────────────────────────┬─────────────────────────────────────┘
                           │ UserPreferences
                           ▼
┌────────────────────────────────────────────────────────────────┐
│                  DATA INGESTION LAYER                          │
│     DatasetLoader → DataPreprocessor → RestaurantRepository    │
│          (Hugging Face datasets → pandas DataFrame)            │
└──────────────────────────┬─────────────────────────────────────┘
                           │ list[Restaurant]
                           ▼
┌────────────────────────────────────────────────────────────────┐
│                  INTEGRATION LAYER                             │
│       RestaurantFilter → CandidateSelector → PromptBuilder     │
└──────────────────────────┬─────────────────────────────────────┘
                           │ Structured Prompt
                           ▼
┌────────────────────────────────────────────────────────────────┐
│              RECOMMENDATION ENGINE (LLM LAYER)                 │
│    LLMClient (Groq) → ResponseParser → RecommendationEnricher  │
│               Model: llama-3.3-70b-versatile                   │
└──────────────────────────┬─────────────────────────────────────┘
                           │ RecommendationResponse
                           ▼
┌────────────────────────────────────────────────────────────────┐
│                  OUTPUT DISPLAY LAYER                          │
│      RecommendationPresenter → ResultsView → SummaryBanner     │
└────────────────────────────────────────────────────────────────┘
```

### Request Flow (Detailed Sequence)

```
User
  │
  │  Enter preferences (location, budget, cuisine, min_rating, additional)
  ▼
Presentation Layer (Streamlit / CLI)
  │
  │  POST /recommend  (or direct function call)
  ▼
Preference Validator
  │  ✓ Validate required fields, enums, rating bounds
  │  ✓ Normalize: lowercase, trim, alias map
  │
  │  UserPreferences (validated)
  ▼
Restaurant Repository  ──►  get_all() / query
  │
  │  restaurants[] (full dataset, cached in-memory)
  ▼
Restaurant Filter
  │  → filter by location (case-insensitive match)
  │  → filter by budget tier
  │  → filter by min_rating
  │  → filter by cuisine (if provided)
  │  → sort by rating desc, then votes desc
  │  → take top N candidates (default N = 15–20)
  │
  │  candidates[] (top N)
  ▼
Prompt Builder
  │  Constructs structured prompt with:
  │  - System instructions (role, JSON format, ranking criteria)
  │  - User preferences (serialized)
  │  - Candidate restaurants (compact JSON array)
  │  - Task description (rank top K, explain each, summarize)
  │
  │  Structured prompt (JSON-compatible)
  ▼
Groq LLMClient  ──►  chat.completions.create(model="llama-3.3-70b-versatile")
  │
  │  JSON response (ranked recommendations with explanations)
  ▼
Response Parser + Enricher
  │  ✓ Parse JSON from Groq response
  │  ✓ Validate response schema
  │  ✓ Join LLM ranks/explanations with full restaurant records
  │
  │  RecommendationResponse
  ▼
Presentation Layer
  │
  │  Render ranked cards with Groq explanations + summary banner
  ▼
User
```

---

## 7. Component Breakdown

### 7.1 Data Ingestion Layer

**Responsibility:** Load, normalize, and cache the Zomato dataset once at startup (or on first request).

| Component | Role |
|-----------|------|
| `DatasetLoader` | Fetches `ManikaSaini/zomato-restaurant-recommendation` via Hugging Face `datasets` library |
| `DataPreprocessor` | Maps raw columns to canonical schema, handles nulls, normalizes text fields |
| `RestaurantRepository` | In-memory query interface over the preprocessed DataFrame |

**Caching strategy:** Load once into a pandas DataFrame or list of `Restaurant` objects. Persist a local parquet/CSV snapshot to `./data/zomato_cache.parquet` to avoid repeated Hugging Face downloads during development.

---

### 7.2 User Input Layer

**Responsibility:** Collect, validate, and normalize user preferences before passing downstream.

| Component | Role |
|-----------|------|
| `PreferenceForm` | UI form or CLI prompt collecting all preference fields |
| `PreferenceValidator` | Enforces required fields, enum values, rating bounds |
| `PreferenceNormalizer` | Lowercases cuisine, maps city aliases, trims free text |

---

### 7.3 Integration Layer

**Responsibility:** Sits between structured data and the LLM. Ensures the model only reasons over a bounded, relevant candidate set.

#### 7.3.1 Restaurant Filter (Deterministic)

Applies hard filters in sequence:
```
all restaurants
  → filter by location (exact or case-insensitive match)
  → filter by budget tier
  → filter by min_rating
  → filter by cuisine (if provided; match if cuisine in restaurant.cuisines)
  → sort by rating desc, then votes desc
  → take top N candidates (default N = 15–20)
```

**Constraint relaxation:** If zero candidates remain after filtering, relax constraints in order: `cuisine` → `budget` → `min_rating`, and surface a warning to the user explaining which constraints were relaxed.

| Component | Role |
|-----------|------|
| `RestaurantFilter` | Executes filter pipeline; returns `list[Restaurant]` |
| `CandidateSelector` | Caps result count and applies tie-breaking logic |

#### 7.3.2 Prompt Builder

Constructs a structured LLM prompt containing:
- **System instructions** — role definition, output format (JSON), ranking criteria
- **User preferences** — serialized `UserPreferences`
- **Candidate restaurants** — compact JSON array of filtered restaurants
- **Task** — rank top K (e.g. 5), explain each pick, optionally summarize

**Design principles:**
- Require JSON output from the LLM for reliable parsing
- Include `restaurant.id` in candidates so explanations map back to structured data
- Instruct the model to **only recommend from the provided list** (no fabrication)
- Pass `additional` preferences as soft signals the LLM may use in ranking/explanation

**Prompt structure (conceptual):**
```
[System]
You are a restaurant recommendation assistant for Indian cities.
Rank restaurants from the CANDIDATES list only. Return valid JSON.

[User Preferences]
{ location, budget, cuisine, min_rating, additional }

[Candidates]
[ { id, name, location, cuisines, cost_for_two, rating }, ... ]

[Task]
Return top 5 restaurants as JSON:
{
  "summary": "...",
  "recommendations": [
    { "id": "...", "rank": 1, "explanation": "..." }
  ]
}
```

---

### 7.4 Recommendation Engine (Groq LLM Layer)

**Responsibility:** Invoke the Groq API, handle retries, parse and validate the response, merge with structured data.

| Component | Role |
|-----------|------|
| `LLMClient` | Thin adapter over the Groq API via the official `groq` Python SDK |
| `RecommendationService` | Orchestrates the full prompt → Groq → parse → enrich pipeline |
| `ResponseParser` | Parses JSON from Groq response; validates schema; handles malformed output |
| `RecommendationEnricher` | Joins Groq LLM ranks/explanations with full restaurant records from the repository |

#### Why Groq?

| Advantage | Details |
|-----------|---------|
| **Speed** | Sub-second token generation via Language Processing Unit (LPU) — essential for responsive, interactive UI |
| **Open models** | Runs open-weight LLaMA models, reducing vendor lock-in |
| **JSON mode** | `response_format={"type": "json_object"}` reduces parse failures for recommendation ranking |
| **Cost-effective** | Competitive pricing for high-throughput inference |

#### Groq Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| **SDK** | `groq` | Official Python client — `pip install groq` |
| **API Key** | `GROQ_API_KEY` | Required; set in `.env`, never committed to source control |
| **Primary Model** | `llama-3.3-70b-versatile` | Strong reasoning for ranking + explanation tasks |
| **Fallback Model** | `llama-3.1-8b-instant` | Faster/cheaper alternative for development and testing |
| **Temperature** | `0.3` | Low for consistent JSON output; retry with `0.1` on parse failure |
| **Output Format** | `response_format={"type": "json_object"}` | Enforced where model supports it |

#### Groq Client Usage

```python
from groq import Groq

client = Groq(api_key=settings.GROQ_API_KEY)

response = client.chat.completions.create(
    model=settings.GROQ_MODEL,          # "llama-3.3-70b-versatile"
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ],
    temperature=settings.GROQ_TEMPERATURE,   # 0.3
    response_format={"type": "json_object"}, # enforced where model supports it
)

raw_json = response.choices[0].message.content
```

#### Reliability Patterns

| Pattern | Purpose |
|---------|---------|
| Structured output / JSON mode | Reduce parse failures from Groq response |
| Retry with temperature reduction (`0.1`) | Recover from invalid JSON on first attempt |
| Fallback heuristic ranking | If Groq fails after retries, return top-K by rating with a generic explanation |
| Idempotency | Same preferences + same dataset snapshot → reproducible candidate set |
| Exponential backoff on Groq 429 | Handle Groq rate limits gracefully before falling back |

#### Groq-Specific Considerations

- Groq offers very low latency — suitable for interactive UI feedback (sub-second responses)
- Use `response_format={"type": "json_object"}` where the selected model supports it
- Handle Groq 429 rate limits with exponential backoff before falling back to heuristic ranking
- Log model ID and latency per request; Groq responses include token usage in `response.usage`

> **🚫 LLM is NOT used for:** loading data, hard filtering by location/budget/rating, or inventing restaurants not in the candidate list.

---

### 7.5 Output Display Layer

**Responsibility:** Render `RecommendationResponse` in a clear, scannable format.

| Component | Role |
|-----------|------|
| `RecommendationPresenter` | Formats `RecommendationResponse` for UI or CLI |
| `ResultsView` | Cards/table showing name, cuisine, rating, cost, Groq-generated explanation |
| `SummaryBanner` | Optional Groq LLM summary at the top of results |

**Each result card must show:**
- 🏆 Rank badge
- 🍽️ Restaurant Name
- 🍜 Cuisine
- ⭐ Rating
- 💰 Estimated Cost for Two
- 🤖 AI-generated explanation (from Groq)

**UX requirements:**
- Show applied filters (location, budget, etc.) above results so the user knows what was searched
- Display "no results" state with suggestions to broaden filters
- Show loading state while dataset loads / Groq LLM responds
- Indicate if AI explanation is unavailable (fallback to heuristic mode)
- Results should be rank-ordered and visually distinct

---

## 8. Data Models

### Output: Recommendation

```python
Recommendation = {
    "rank": int,
    "name": str,
    "cuisine": str,           # joined cuisine string for display
    "rating": float,
    "estimated_cost": int,    # cost_for_two
    "explanation": str,       # Groq LLM-generated
}
```

### Output: RecommendationResponse

```python
RecommendationResponse = {
    "summary": str | None,            # Optional Groq LLM summary of recommendations
    "recommendations": list[Recommendation],
    "metadata": {
        "candidates_considered": int,  # How many restaurants passed filters
        "filters_applied": dict,       # The filters used for this request
        "model": str,                  # e.g. "llama-3.3-70b-versatile"
    }
}
```

---

## 9. API Design (Optional REST Layer)

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/recommend` | `POST` | Primary recommendation endpoint |
| `/api/v1/health` | `GET` | Service status + dataset load state |
| `/api/v1/locations` | `GET` | Distinct locations from dataset (for UI dropdowns) |
| `/api/v1/cuisines` | `GET` | Distinct cuisines extracted from dataset (for UI dropdowns) |

### `POST /api/v1/recommend`

**Request:**
```json
{
  "location": "Bangalore",
  "budget": "medium",
  "cuisine": "Italian",
  "min_rating": 4.0,
  "additional": "family-friendly, outdoor seating"
}
```

**Response:**
```json
{
  "summary": "Based on your preference for Italian cuisine in Bangalore with a medium budget, here are the top picks that match your requirements for family-friendly dining...",
  "recommendations": [
    {
      "rank": 1,
      "name": "Example Ristorante",
      "cuisine": "Italian, Continental",
      "rating": 4.5,
      "estimated_cost": 1200,
      "explanation": "Highly rated Italian spot within your budget, known for family-friendly ambiance and spacious outdoor seating."
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
    "model": "llama-3.3-70b-versatile"
  }
}
```

---

## 10. Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Language** | Python 3.11+ | Strong ecosystem for data processing + LLM integration |
| **Dataset** | `datasets` (Hugging Face) | Direct access to the specified Zomato dataset |
| **Data processing** | `pandas` | Filtering, normalization, caching, DataFrame operations |
| **LLM Provider** | **Groq** | Ultra-low-latency inference via LPU hardware |
| **LLM Model** | `llama-3.3-70b-versatile` | Strong open-weight model for ranking + explanation tasks |
| **LLM SDK** | `groq` | Official Groq Python client (`pip install groq`) |
| **Config** | `pydantic-settings` + `.env` | Typed config and secret management |
| **API (optional)** | FastAPI | Lightweight async REST for frontend decoupling |
| **UI (optional)** | Streamlit or Gradio | Rapid prototyping of preference form + results display |
| **Testing** | `pytest` | Unit tests for filter, parser, preprocessor; mocked Groq client |

---

## 11. Proposed Folder Structure

```
zomato-milestone1/
├── docs/
│   ├── context.md                  ← This document (product requirements & workflow)
│   ├── architecture.md             ← Technical architecture & component design
│   └── problemStatement.txt        ← Original problem statement
├── src/
│   ├── __init__.py
│   ├── main.py                     # Entry point (CLI or app bootstrap)
│   ├── config.py                   # Env vars, budget thresholds, top-K, Groq settings
│   ├── models/
│   │   ├── restaurant.py           # Restaurant dataclass
│   │   ├── preferences.py          # UserPreferences dataclass
│   │   └── recommendation.py       # Recommendation, RecommendationResponse
│   ├── data/
│   │   ├── loader.py               # Hugging Face dataset loader
│   │   ├── preprocessor.py         # Normalization & schema mapping
│   │   └── repository.py           # In-memory query interface
│   ├── services/
│   │   ├── filter.py               # RestaurantFilter + CandidateSelector
│   │   ├── prompt_builder.py       # PromptBuilder (crafts Groq prompts)
│   │   ├── llm_client.py           # Groq API adapter (wraps groq SDK)
│   │   └── recommendation.py       # RecommendationService orchestrator
│   ├── api/
│   │   ├── routes.py               # FastAPI routes (optional)
│   │   └── schemas.py              # Request/response Pydantic models
│   └── ui/
│       ├── cli.py                  # Terminal interface
│       └── streamlit_app.py        # Streamlit web UI (optional)
├── tests/
│   ├── test_filter.py
│   ├── test_preprocessor.py
│   └── test_recommendation.py      # Uses mocked Groq client
├── data/                           # Cached parquet/csv (gitignored)
├── .env.example                    # GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE
├── requirements.txt
└── README.md
```

---

## 12. Configuration Reference (`config.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_DATASET_NAME` | `ManikaSaini/zomato-restaurant-recommendation` | Hugging Face dataset identifier |
| `BUDGET_THRESHOLDS` | `{low: 500, medium: 1500}` | Cost-for-two INR thresholds for budget tier derivation |
| `MAX_CANDIDATES_FOR_LLM` | `20` | Maximum restaurants sent to Groq in a single prompt |
| `TOP_K_RECOMMENDATIONS` | `5` | Number of top results to display to the user |
| `GROQ_API_KEY` | `$env` | Groq API key (from `.env`, never hardcoded) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Primary Groq model for ranking + explanations |
| `GROQ_FALLBACK_MODEL` | `llama-3.1-8b-instant` | Fallback Groq model (faster, cheaper for dev/testing) |
| `GROQ_TEMPERATURE` | `0.3` | Inference temperature (retry with `0.1` on parse failure) |
| `DATA_CACHE_PATH` | `./data/zomato_cache.parquet` | Local dataset cache path |

---

## 13. Error Handling

| Scenario | Behavior | Fallback |
|----------|----------|----------|
| Dataset download fails | Retry with exponential backoff | Load from local parquet/CSV cache if available; show clear error in UI |
| No restaurants match filters | Relax constraints in order: cuisine → budget → rating | Surface warning explaining which constraints were relaxed; prompt user to adjust |
| LLM returns invalid JSON | Retry once with `temperature=0.1` | Fall back to heuristic top-K ranking (by rating desc) with generic explanation |
| Groq 429 rate limit | Exponential backoff (up to 3 retries) | Return heuristic top-K with note that AI explanation is currently unavailable |
| Groq timeout | Same as 429 handling | Same as 429 fallback |
| Unknown location input | Suggest valid locations from dataset | Show list of available locations for user to choose from |
| Empty dataset | Show clear error message | Suggest checking network connectivity or dataset availability |

---

## 14. Cross-Cutting Concerns

### 14.1 Logging & Observability

- Log filter counts at each stage (input size → candidate size after each filter)
- Log Groq LLM latency and token usage (`response.usage`)
- Log which Groq model was used per request (primary vs. fallback)
- Do **not** log full prompts containing sensitive user data
- Optional: trace ID per recommendation request for debugging

### 14.2 Security

- Store `GROQ_API_KEY` in environment variables, never in source control
- Add `.env` to `.gitignore`; provide `.env.example` with placeholder values
- Validate and sanitize all user inputs before building Groq prompts
- Rate-limit API endpoints if deployed publicly
- Never expose internal error details to end users

---

## 15. Testing Strategy

| Test Type | Scope | Example |
|-----------|-------|---------|
| **Unit** | `RestaurantFilter` | Location + budget + rating filters return expected subset |
| **Unit** | `DataPreprocessor` | Cuisine string parsing, numeric coercion, null handling |
| **Unit** | `ResponseParser` | Valid/invalid Groq JSON response handling |
| **Integration** | `RecommendationService` | Mock Groq client returns fixed JSON; verify enriched output |
| **Snapshot** | `PromptBuilder` | Prompt contains all candidates and preference fields correctly |

> Use a frozen subset of the dataset (10–20 rows) in test fixtures for deterministic tests. Mock the `groq` SDK client to avoid real API calls in CI.

---

## 16. Deployment Topology

### Development (Local)

```
Developer Machine
├── Python app (Streamlit / FastAPI + CLI)
├── Cached dataset in ./data/
└── Groq API (cloud, via GROQ_API_KEY)
```

### Minimal Production

```
User → Streamlit / Static Frontend
     → FastAPI Backend
     → Cached Dataset (pre-loaded at container startup)
     → Groq API (cloud)
```

- Pre-load dataset at container startup for fast first-request response
- Single stateless API instance is sufficient for milestone scope
- Scale horizontally later by sharing a read-only dataset snapshot

---

## 17. Implementation Phases

| Phase | Deliverable | Key Activities |
|-------|-------------|----------------|
| **Phase 1 — Data** | Dataset pipeline | Load Hugging Face dataset, preprocess, cache locally, expose `RestaurantRepository` |
| **Phase 2 — Filter** | Preference & filtering | Implement preference validation/normalization and deterministic restaurant filtering |
| **Phase 3 — LLM** | Groq integration | Build prompt builder, Groq API adapter, response parser, recommendation enricher |
| **Phase 4 — UI** | User interface | CLI or Streamlit form + results display with Groq-generated explanations |
| **Phase 5 — Hardening** | Production readiness | Error handling, fallback ranking, logging, tests, README, `.env.example` |

---

## 18. Success Criteria

- [ ] Dataset loads from Hugging Face and preprocesses correctly to canonical schema
- [ ] User inputs are collected, validated, and normalized
- [ ] Deterministic filtering returns a relevant, non-empty candidate list (with graceful constraint relaxation)
- [ ] Groq LLM generates coherent, grounded JSON recommendations from candidates only
- [ ] Each recommendation shows: restaurant name, cuisine, rating, estimated cost, and AI-generated explanation
- [ ] Output is displayed in a clean, rank-ordered, user-friendly format with applied-filter context
- [ ] System handles edge cases: no results, bad JSON from LLM, Groq API errors, unknown location
- [ ] Fallback heuristic ranking works when Groq is unavailable
- [ ] API keys are never stored in source control (`.env` in `.gitignore`)
- [ ] Unit tests pass for filter, preprocessor, and response parser
- [ ] Logging captures filter counts, Groq model used, latency, and token usage

---

## 19. Key Architecture Decisions

| Decision | Choice | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| **LLM Provider** | Groq (`llama-3.3-70b-versatile`) | OpenAI GPT-4, Anthropic Claude, local models (Ollama) | Sub-second latency via LPU; open-weight models; JSON mode support |
| **Pre-filter before LLM** | Yes — hard filters in Python code | Let LLM filter entire dataset | LLM filtering is expensive, unreliable, and hallucination-prone |
| **LLM Output Format** | Structured JSON via Groq | Free-form text | JSON is parseable, testable, and mappable to structured records |
| **Data Storage** | In-memory DataFrame | Database (PostgreSQL, SQLite) | Read-only milestone dataset doesn't need persistence layer complexity |
| **Ranking Split** | Heuristic shortlist + LLM final rank | Pure LLM or pure heuristic | Best of both: deterministic efficiency + LLM reasoning quality |
| **UI Approach** | Streamlit for speed | React SPA | Faster prototyping for milestone 1; swap later via extensible architecture |

---

*Generated on: 2026-06-21 | Based on: `problemStatement.txt` + `architecture.md` | Architecture Version: 1.0 | LLM Provider: Groq (`llama-3.3-70b-versatile`)*
