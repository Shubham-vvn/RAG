# 📋 Project Context: AI-Powered Restaurant Recommendation System (Zomato Use Case)

> **Related Documents:**
> - [`architecture.md`](./architecture.md) — Full technical architecture and component design
> - [`problemStatement.txt`](./problemStatement.txt) — Original problem statement

---

## 1. Project Overview

This project involves designing and building an **AI-powered restaurant recommendation service** inspired by Zomato. The system bridges structured restaurant data (from Hugging Face) with the reasoning capabilities of **Groq's LLM** to deliver personalized, explainable restaurant suggestions.

Rather than returning simple filtered results, the system:
- **Understands user intent** via structured preference input
- **Deterministically filters** candidates using hard constraints before involving the LLM
- **Reasons and ranks** options using Groq's `llama-3.3-70b-versatile` model
- **Explains every recommendation** in natural language tied to user preferences
- **Presents results** in a clean, scannable format

---

## 2. Problem Statement Summary

> *"Build an application that takes user preferences (location, budget, cuisine, ratings), uses a real-world restaurant dataset, and leverages an LLM to generate personalized, human-like recommendations."*

The system must:
- Accept structured and free-text user preferences
- Filter relevant restaurants from a real-world dataset
- Use an LLM to rank and explain results
- Display recommendations with cuisine, rating, cost, and AI rationale

---

## 3. Architecture Goals

| Goal | Description |
|------|-------------|
| **Separation of concerns** | Data loading, filtering, LLM reasoning, and presentation are isolated modules with clear interfaces |
| **Deterministic pre-filtering** | Hard constraints (location, budget, rating) are applied before the LLM to reduce token cost and hallucination risk |
| **Explainability** | Every recommendation includes an LLM-generated rationale tied to user preferences |
| **Extensibility** | Swap UI frameworks or data sources without rewriting core logic; LLM access is isolated behind a Groq adapter |
| **Testability** | Pure functions for filtering/ranking prep; mockable LLM adapter for unit tests |

---

## 4. Dataset

- **Source:** [Hugging Face — ManikaSaini/zomato-restaurant-recommendation](https://huggingface.co/datasets/ManikaSaini/zomato-restaurant-recommendation)
- **Access:** Via `datasets` Python library (Hugging Face)
- **Caching:** Load once at startup; persist a local parquet/CSV snapshot to avoid repeated downloads during development

### Canonical Restaurant Schema

After preprocessing, each restaurant is normalized to:

```python
Restaurant = {
    "id": str,              # stable identifier (index or dataset id)
    "name": str,
    "location": str,        # city / locality
    "cuisines": list[str],  # e.g. ["Italian", "Continental"]
    "cost_for_two": int,    # numeric cost indicator
    "rating": float,        # e.g. 4.2
    "votes": int,           # optional: popularity signal
    "rest_type": str,       # optional: casual dining, cafe, etc.
}
```

### Key Raw Fields to Extract

| Raw Field | Canonical Field | Notes |
|-----------|----------------|-------|
| `restaurant_name` | `name` | Direct mapping |
| `location` / `city` | `location` | Trim, title-case, alias map |
| `cuisines` | `cuisines` | Split `"Italian, Chinese"` → `["Italian", "Chinese"]` |
| `average_cost_for_two` | `cost_for_two` | Coerce to int |
| `aggregate_rating` | `rating` | Coerce to float |
| `votes` | `votes` | Popularity signal |
| `rest_type` | `rest_type` | Casual dining, café, etc. |

---

## 5. User Input Specification

### Input Model

```python
UserPreferences = {
    "location": str,           # required
    "budget": str,             # "low" | "medium" | "high"
    "cuisine": str | None,     # optional primary cuisine
    "min_rating": float,       # e.g. 3.5
    "additional": str | None,  # free-text: "family-friendly, quick service"
}
```

### Input Fields

| Input | Type | Required | Examples |
|-------|------|----------|---------|
| **Location** | String | ✅ Yes | `"Delhi"`, `"Bangalore"`, `"Mumbai"` |
| **Budget** | Enum | ✅ Yes | `"low"`, `"medium"`, `"high"` |
| **Cuisine** | String | ❌ No | `"Italian"`, `"Chinese"`, `"North Indian"` |
| **Min Rating** | Float [0–5] | ✅ Yes | `3.5`, `4.0` |
| **Additional Preferences** | Free text | ❌ No | `"family-friendly"`, `"quick service"` |

### Budget Tier Mapping

| Budget | `cost_for_two` Range (INR) | `price_range` |
|--------|---------------------------|----------------|
| `low` | ≤ ₹500 | 1 |
| `medium` | ₹501 – ₹1,500 | 2–3 |
| `high` | > ₹1,500 | 4 |

> Thresholds should be tuned after inspecting the actual dataset distribution. Configurable in `config.py`.

### Validation Rules

- `location` — non-empty; must match at least one value in the dataset (or suggest closest matches)
- `budget` — one of `low`, `medium`, `high`
- `min_rating` — float in `[0.0, 5.0]`
- `cuisine` — optional; fuzzy match against known cuisine vocabulary extracted from dataset
- `additional` — optional free text; passed through to LLM as soft signal

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
                                               [Build LLM Prompt]
                                                        │
                                                        ▼
                                             [Groq LLM — Rank + Explain]
                                                        │
                                                        ▼
                                              [Parse & Enrich Response]
                                                        │
                                                        ▼
                                        RecommendationResponse ──► UI
```

### Component Overview

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

---

## 7. Component Breakdown

### 7.1 Data Ingestion Layer

Loads, normalizes, and caches the dataset once at startup.

| Component | Role |
|-----------|------|
| `DatasetLoader` | Fetches `ManikaSaini/zomato-restaurant-recommendation` via Hugging Face `datasets` |
| `DataPreprocessor` | Maps raw columns to canonical schema, handles nulls, normalizes text |
| `RestaurantRepository` | In-memory query interface over the preprocessed DataFrame |

**Preprocessing steps:**
1. Download dataset split (typically `train`)
2. Select and rename relevant columns to canonical schema
3. Parse cuisine strings into lists (`"Italian, Chinese"` → `["Italian", "Chinese"]`)
4. Coerce `rating` and `cost_for_two` to numeric; drop/impute invalid rows
5. Normalize location strings (trim, title-case, alias map)
6. Derive `budget_tier` from `cost_for_two` using configurable thresholds

---

### 7.2 User Input Layer

Collects, validates, and normalizes user preferences before passing downstream.

| Component | Role |
|-----------|------|
| `PreferenceForm` | UI form or CLI prompt collecting all fields |
| `PreferenceValidator` | Enforces required fields, enum values, rating bounds |
| `PreferenceNormalizer` | Lowercases cuisine, maps city aliases, trims free text |

---

### 7.3 Integration Layer

Sits between structured data and the LLM. Ensures the model only reasons over a bounded, relevant candidate set.

#### 7.3.1 Restaurant Filter (Deterministic)

Applies hard filters in sequence:
```
all restaurants
  → filter by location (case-insensitive match)
  → filter by budget tier
  → filter by min_rating
  → filter by cuisine (if provided; match if cuisine in restaurant.cuisines)
  → sort by rating desc, then votes desc
  → take top N candidates (default N = 15–20)
```

**Constraint relaxation:** If zero candidates remain, relax in order: `cuisine` → `budget` → `min_rating`, and surface a warning to the user.

| Component | Role |
|-----------|------|
| `RestaurantFilter` | Executes filter pipeline; returns `list[Restaurant]` |
| `CandidateSelector` | Caps result count and applies tie-breaking |

#### 7.3.2 Prompt Builder

Constructs a structured LLM prompt containing:
- **System instructions** — role, output format (JSON), ranking criteria
- **User preferences** — serialized `UserPreferences`
- **Candidate restaurants** — compact JSON array of filtered restaurants
- **Task** — rank top K (e.g. 5), explain each pick, optionally summarize

**Design principles:**
- Require JSON output from the LLM for reliable parsing
- Include `restaurant.id` in candidates so explanations map back to structured data
- Instruct the model to **only recommend from the provided list** (no fabrication)
- Pass `additional` preferences as soft signals for ranking/explanation

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

### 7.4 Recommendation Engine (LLM Layer)

Invokes Groq, handles retries, parses JSON, and merges with structured data.

| Component | Role |
|-----------|------|
| `LLMClient` | Thin adapter over Groq API via official `groq` Python SDK |
| `RecommendationService` | Orchestrates prompt → LLM → parse → enrich pipeline |
| `ResponseParser` | Parses JSON; validates schema; handles malformed output |
| `RecommendationEnricher` | Joins LLM ranks/explanations with full restaurant records |

#### Groq Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| SDK | `groq` | `pip install groq` |
| API Key | `GROQ_API_KEY` | Set in `.env`, never committed |
| Primary Model | `llama-3.3-70b-versatile` | Strong reasoning for ranking + explanation |
| Fallback Model | `llama-3.1-8b-instant` | Faster/cheaper for dev |
| Temperature | `0.3` | Low for consistent JSON; retry with `0.1` on parse failure |

**Client usage (conceptual):**
```python
from groq import Groq

client = Groq(api_key=settings.GROQ_API_KEY)

response = client.chat.completions.create(
    model=settings.GROQ_MODEL,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ],
    temperature=settings.GROQ_TEMPERATURE,
    response_format={"type": "json_object"},  # when supported by model
)
```

**Reliability patterns:**

| Pattern | Purpose |
|---------|---------|
| Structured output / JSON mode | Reduce parse failures |
| Retry with temperature reduction | Recover from invalid JSON |
| Fallback heuristic ranking | If LLM fails, return top-K by rating with generic explanation |
| Idempotency | Same preferences + same dataset snapshot → reproducible candidate set |

> **LLM is NOT used for:** loading data, hard filtering, or inventing restaurants not in the candidate list.

---

### 7.5 Output Display Layer

Renders `RecommendationResponse` in a clear, scannable format.

| Component | Role |
|-----------|------|
| `RecommendationPresenter` | Formats `RecommendationResponse` for UI or CLI |
| `ResultsView` | Cards/table showing name, cuisine, rating, cost, explanation |
| `SummaryBanner` | Optional LLM summary at the top |

**Each result card must show:**
- 🏆 Rank badge
- 🍽️ Restaurant Name
- 🍜 Cuisine
- ⭐ Rating
- 💰 Estimated Cost for Two
- 🤖 AI-generated explanation

**UX requirements:**
- Show applied filters (location, budget, etc.) above results
- Display "no results" state with suggestions to broaden filters
- Show loading state while dataset loads / LLM responds

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
    "explanation": str,       # LLM-generated
}
```

### Output: RecommendationResponse

```python
RecommendationResponse = {
    "summary": str | None,
    "recommendations": list[Recommendation],
    "metadata": {
        "candidates_considered": int,
        "filters_applied": dict,
        "model": str,
    }
}
```

---

## 9. API Design (Optional REST Layer)

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
  "summary": "Based on your preference for Italian cuisine in Bangalore...",
  "recommendations": [
    {
      "rank": 1,
      "name": "Example Ristorante",
      "cuisine": "Italian, Continental",
      "rating": 4.5,
      "estimated_cost": 1200,
      "explanation": "Highly rated Italian spot within your budget, known for family-friendly ambiance."
    }
  ],
  "metadata": {
    "candidates_considered": 18,
    "filters_applied": { "location": "Bangalore", "budget": "medium", "min_rating": 4.0, "cuisine": "Italian" },
    "model": "llama-3.3-70b-versatile"
  }
}
```

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/recommend` | Primary recommendation endpoint |
| `GET /api/v1/health` | Service status + dataset load state |
| `GET /api/v1/locations` | Distinct locations (for UI dropdowns) |
| `GET /api/v1/cuisines` | Distinct cuisines from dataset |

---

## 10. Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Language** | Python 3.11+ | Strong ecosystem for data + LLM integration |
| **Dataset** | `datasets` (Hugging Face) | Direct access to the specified dataset |
| **Data processing** | `pandas` | Filtering, normalization, caching |
| **LLM** | Groq (`llama-3.3-70b-versatile`) | Fast, low-latency inference for ranking + explanations |
| **LLM SDK** | `groq` | Official Groq Python client for chat completions |
| **Config** | `pydantic-settings` + `.env` | Typed config and secret management |
| **API (optional)** | FastAPI | Lightweight async REST for frontend decoupling |
| **UI (optional)** | Streamlit or Gradio | Rapid prototyping of preference form + results |
| **Testing** | `pytest` | Unit tests for filter, parser, preprocessor |

---

## 11. Proposed Folder Structure

```
zomato-milestone1/
├── docs/
│   ├── context.md                  ← This document
│   ├── architecture.md             ← Technical architecture
│   └── problemStatement.txt        ← Original problem statement
├── src/
│   ├── __init__.py
│   ├── main.py                     # Entry point (CLI or app bootstrap)
│   ├── config.py                   # Env vars, budget thresholds, top-K
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
│   │   ├── prompt_builder.py       # PromptBuilder
│   │   ├── llm_client.py           # Groq API adapter
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
│   └── test_recommendation.py
├── data/                           # Cached parquet/csv (gitignored)
├── .env.example                    # GROQ_API_KEY and model config template
├── requirements.txt
└── README.md
```

---

## 12. Configuration Reference (`config.py`)

| Variable | Purpose |
|----------|---------|
| `HF_DATASET_NAME` | Hugging Face dataset identifier |
| `BUDGET_THRESHOLDS` | Dict mapping budget tiers to `cost_for_two` ranges |
| `MAX_CANDIDATES_FOR_LLM` | Max restaurants sent to LLM (default: 15–20) |
| `TOP_K_RECOMMENDATIONS` | Top K results to display (default: 5) |
| `GROQ_API_KEY` | Groq API key (from env) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` (default) |
| `GROQ_TEMPERATURE` | `0.3` (retry with `0.1` on parse failure) |
| `DATA_CACHE_PATH` | Local parquet/CSV cache path |

---

## 13. Error Handling

| Scenario | Behavior |
|----------|---------|
| Dataset download fails | Retry with backoff; show clear error in UI |
| No restaurants match filters | Relax constraints (cuisine → budget → rating) or prompt user to adjust |
| LLM returns invalid JSON | Retry once with lower temperature; fallback to heuristic ranking |
| Groq 429 rate limit / timeout | Retry with exponential backoff; return heuristic top-K with note that AI explanation is unavailable |
| Unknown location input | Suggest valid locations from dataset |

---

## 14. Testing Strategy

| Test Type | Scope | Example |
|-----------|-------|---------|
| **Unit** | `RestaurantFilter` | Location + budget + rating filters return expected subset |
| **Unit** | `Preprocessor` | Cuisine string parsing, numeric coercion |
| **Unit** | `ResponseParser` | Valid/invalid LLM JSON handling |
| **Integration** | `RecommendationService` | Mock LLM returns fixed JSON; verify enriched output |
| **Snapshot** | `PromptBuilder` | Prompt contains all candidates and preference fields |

> Use a frozen subset of the dataset (10–20 rows) in test fixtures for deterministic tests.

---

## 15. Implementation Phases

| Phase | Deliverable |
|-------|-------------|
| **Phase 1 — Data** | Load Hugging Face dataset, preprocess, cache, expose repository |
| **Phase 2 — Filter** | Implement preference validation and deterministic filtering |
| **Phase 3 — LLM** | Prompt builder, Groq client, response parser, enricher |
| **Phase 4 — UI** | CLI or Streamlit form + results display |
| **Phase 5 — Hardening** | Error handling, fallback ranking, tests, README |

---

## 16. Success Criteria

- [ ] Dataset loads from Hugging Face and preprocesses correctly
- [ ] User inputs are collected and validated
- [ ] Deterministic filtering returns a relevant, non-empty candidate list
- [ ] Groq LLM generates coherent, grounded JSON recommendations
- [ ] Each recommendation shows: name, cuisine, rating, cost, AI explanation
- [ ] Output is displayed in a clean, rank-ordered, user-friendly format
- [ ] System handles edge cases: no results, bad JSON from LLM, API errors, unknown location
- [ ] API keys are never stored in source control

---

## 17. Key Architecture Decisions

| Decision | Choice | Alternatives Considered |
|----------|--------|------------------------|
| **LLM Provider** | Groq (`llama-3.3-70b-versatile`) | OpenAI, Anthropic, local models |
| **Pre-filter before LLM** | Yes — hard filters in code | Let LLM filter entire dataset (expensive, unreliable) |
| **LLM Output Format** | Structured JSON | Free-form text (harder to parse) |
| **Data Storage** | In-memory DataFrame | Database (unnecessary for read-only milestone dataset) |
| **Ranking Split** | Heuristic shortlist + LLM final rank | Pure LLM or pure heuristic |
| **UI Approach** | Streamlit for speed | React SPA (more effort for milestone 1) |

---

*Generated on: 2026-06-19 | Based on: `problemStatement.txt` + `architecture.md`*
