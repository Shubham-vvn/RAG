# Architecture: AI-Powered Restaurant Recommendation System

> This document describes the technical architecture for the Zomato-inspired restaurant recommendation service defined in [`context.md`](./context.md). The system combines structured restaurant data from Hugging Face with **Groq LLM inference** to produce personalized, explainable recommendations.

---

## 1. Architecture Goals

| Goal | Description |
|------|-------------|
| **Separation of concerns** | Data loading, filtering, LLM reasoning, and presentation are isolated modules with clear interfaces |
| **Deterministic pre-filtering** | Hard constraints (location, budget, rating) are applied before the LLM to reduce token cost and hallucination risk |
| **Explainability** | Every recommendation includes a Groq LLM-generated rationale tied to user preferences |
| **Extensibility** | Swap UI frameworks or data sources without rewriting core logic; LLM access is isolated behind a Groq adapter |
| **Testability** | Pure functions for filtering/ranking prep; mockable Groq LLM adapter for unit tests |

---

## 2. LLM Provider: Groq

**Groq is the sole LLM provider for this project.**

Groq offers ultra-low-latency inference via its Language Processing Unit (LPU), making it ideal for interactive recommendation flows where users expect near-instant responses.

| Setting | Value | Notes |
|---------|-------|-------|
| **SDK** | `groq` | Official Python client — `pip install groq` |
| **API Key** | `GROQ_API_KEY` | Required; set in `.env`, never committed to source control |
| **Primary Model** | `llama-3.3-70b-versatile` | Strong reasoning for ranking + explanation tasks |
| **Fallback Model** | `llama-3.1-8b-instant` | Faster/cheaper alternative for development/testing |
| **Temperature** | `0.3` | Low for consistent JSON output; retry with `0.1` on parse failure |
| **Output Format** | `response_format={"type": "json_object"}` | Enforced where model supports it |

### Why Groq?
- **Speed**: Sub-second token generation — essential for a responsive UI
- **Open models**: Runs open-weight LLaMA models, reducing vendor lock-in
- **JSON mode**: Structured output reduces parse failures for recommendation ranking
- **Cost-effective**: Competitive pricing for high-throughput inference

---

## 3. Component Architecture

### 3.1 Data Ingestion Layer

**Responsibility:** Load, normalize, and cache the Zomato dataset once at startup (or on first request).

| Component | Role |
|-----------|------|
| `DatasetLoader` | Fetches `ManikaSaini/zomato-restaurant-recommendation` via Hugging Face `datasets` library |
| `DataPreprocessor` | Maps raw columns to a canonical schema, handles nulls, normalizes text fields |
| `RestaurantRepository` | In-memory query interface over the preprocessed DataFrame |

#### Canonical Restaurant Schema

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

#### Preprocessing Steps

1. Download dataset split (typically `train`)
2. Select and rename relevant columns to the canonical schema
3. Parse cuisine strings into lists (`"Italian, Chinese"` → `["Italian", "Chinese"]`)
4. Coerce `rating` and `cost_for_two` to numeric types; drop or impute invalid rows
5. Normalize location strings (trim, title-case, alias map for city names)
6. Derive `budget_tier` from `cost_for_two` using configurable thresholds:

| Tier | `cost_for_two` Range (INR) |
|------|---------------------------|
| `low` | ≤ 500 |
| `medium` | 501 – 1500 |
| `high` | > 1500 |

> Thresholds should be tuned after inspecting the actual dataset distribution. Configurable via `config.py`.

**Caching strategy:** Load once into a pandas DataFrame or list of `Restaurant` objects. Persist a local parquet/CSV snapshot to avoid repeated Hugging Face downloads during development.

---

### 3.2 User Input Layer

**Responsibility:** Collect, validate, and normalize user preferences.

#### Input Model

```python
UserPreferences = {
    "location": str,           # required
    "budget": str,             # "low" | "medium" | "high"
    "cuisine": str | None,     # optional primary cuisine
    "min_rating": float,       # e.g. 3.5
    "additional": str | None,  # free-text: "family-friendly, quick service"
}
```

| Component | Role |
|-----------|------|
| `PreferenceForm` | UI form or CLI prompt collecting all preference fields |
| `PreferenceValidator` | Enforces required fields, enum values, rating bounds |
| `PreferenceNormalizer` | Lowercases cuisine, maps city aliases, trims free text |

#### Validation Rules

- `location` — non-empty; must match at least one value in the dataset (or suggest closest matches)
- `budget` — one of `low`, `medium`, `high`
- `min_rating` — float in `[0.0, 5.0]`
- `cuisine` — optional; fuzzy match against known cuisine vocabulary extracted from the dataset
- `additional` — optional free text passed through to Groq LLM for soft matching

---

### 3.3 Integration Layer

**Responsibility:** Apply hard filters, rank candidates heuristically, and assemble the Groq LLM prompt.

This layer sits between structured data and the LLM. It ensures the model only reasons over a bounded, relevant candidate set.

#### 3.3.1 Restaurant Filter

Applies deterministic filters in sequence:

```
all restaurants
  → filter by location (exact or case-insensitive match)
  → filter by budget tier
  → filter by min_rating
  → filter by cuisine (if provided; match if cuisine in restaurant.cuisines)
  → sort by rating desc, then votes desc
  → take top N candidates (default N = 15–20)
```

| Component | Role |
|-----------|------|
| `RestaurantFilter` | Executes filter pipeline; returns `list[Restaurant]` |
| `CandidateSelector` | Caps result count and applies tie-breaking |

> If zero candidates remain, relax constraints in order: `cuisine` → `budget` → `min_rating`, and surface a warning to the user.

#### 3.3.2 Prompt Builder

Constructs a structured prompt to send to Groq. The prompt contains:

- **System instructions** — role definition, output format (JSON), ranking criteria
- **User preferences** — serialized `UserPreferences`
- **Candidate restaurants** — compact JSON array of filtered restaurants
- **Task** — rank top K (e.g. 5), explain each pick, optionally summarize

**Design principles:**
- Require JSON output from Groq LLM for reliable parsing
- Include `restaurant.id` in candidates so explanations map back to structured data
- Instruct the model to **only recommend from the provided list** (no fabrication)
- Pass `additional` preferences as soft signals the LLM may use in ranking/explanation

**Example prompt structure (conceptual):**

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
    {
      "id": "...",
      "rank": 1,
      "explanation": "..."
    }
  ]
}
```

---

### 3.4 Recommendation Engine (Groq LLM Layer)

**Responsibility:** Invoke the Groq API, handle retries, parse and validate the response, merge with structured data.

| Component | Role |
|-----------|------|
| `LLMClient` | Thin adapter over the Groq API via the official `groq` Python SDK |
| `RecommendationService` | Orchestrates prompt → Groq → parse → enrich pipeline |
| `ResponseParser` | Parses JSON; validates schema; handles malformed output |
| `RecommendationEnricher` | Joins Groq LLM ranks/explanations with full restaurant records |

#### Output Models

```python
Recommendation = {
    "rank": int,
    "name": str,
    "cuisine": str,           # joined cuisine string for display
    "rating": float,
    "estimated_cost": int,    # cost_for_two
    "explanation": str,       # Groq LLM-generated
}

RecommendationResponse = {
    "summary": str | None,
    "recommendations": list[Recommendation],
    "metadata": {
        "candidates_considered": int,
        "filters_applied": dict,
        "model": str,             # e.g. "llama-3.3-70b-versatile"
    }
}
```

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
| Fallback heuristic ranking | If Groq fails, return top-K by rating with a generic explanation |
| Idempotency | Same preferences + same dataset snapshot → reproducible candidate set |
| Exponential backoff on Groq 429 | Handle Groq rate limits gracefully before falling back |

**Groq-specific considerations:**
- Groq offers very low latency — suitable for interactive UI feedback (sub-second responses)
- Use `response_format={"type": "json_object"}` where the selected model supports it
- Handle Groq 429 rate limits with exponential backoff before falling back to heuristic ranking
- Log model ID and latency per request; Groq responses include token usage in `response.usage`

> **Groq LLM is NOT used for:** loading data, hard filtering by location/budget/rating, or inventing restaurants not in the candidate list.

---

### 3.5 Output Display Layer

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
- Show applied filters (location, budget, etc.) above results
- Display "no results" state with suggestions to broaden filters
- Show loading state while dataset loads / Groq LLM responds
- Indicate if AI explanation is unavailable (fallback to heuristic mode)

---

## 4. Request Flow (Sequence)

```
User
  │
  │  Enter preferences
  ▼
Presentation Layer
  │
  │  POST /recommend
  ▼
Preference Validator
  │
  │  UserPreferences
  ▼
Restaurant Repository  ──►  get_all() / query
  │
  │  restaurants[]
  ▼
Restaurant Filter
  │
  │  candidates[] (top N)
  ▼
Prompt Builder
  │
  │  Structured prompt (JSON)
  ▼
Groq LLMClient  ──►  chat.completions.create(model="llama-3.3-70b-versatile")
  │
  │  JSON response
  ▼
Response Parser + Enricher
  │
  │  RecommendationResponse
  ▼
Presentation Layer
  │
  │  Render ranked cards with Groq explanations
  ▼
User
```

---

## 5. Proposed Module Structure

```
zomato-milestone1/
├── docs/
│   ├── context.md                  ← Product requirements and workflow
│   ├── architecture.md             ← This document
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

## 6. Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Language** | Python 3.11+ | Strong ecosystem for data + LLM integration |
| **Dataset** | `datasets` (Hugging Face) | Direct access to the specified dataset |
| **Data processing** | `pandas` | Filtering, normalization, caching |
| **LLM Provider** | **Groq** | Ultra-low-latency inference via LPU hardware |
| **LLM Model** | `llama-3.3-70b-versatile` | Strong open-weight model for ranking + explanation |
| **LLM SDK** | `groq` | Official Groq Python client (`pip install groq`) |
| **Config** | `pydantic-settings` + `.env` | Typed config and secret management |
| **API (optional)** | FastAPI | Lightweight async REST for frontend decoupling |
| **UI (optional)** | Streamlit or Gradio | Rapid prototyping of preference form + results |
| **Testing** | `pytest` | Unit tests for filter, parser, preprocessor, mocked Groq |

---

## 7. API Design (Optional REST Layer)

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
  "summary": "Based on your preference for Italian cuisine in Bangalore with a medium budget...",
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

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/recommend` | Primary recommendation endpoint |
| `GET /api/v1/health` | Service status + dataset load state |
| `GET /api/v1/locations` | Distinct locations from dataset (for UI dropdowns) |
| `GET /api/v1/cuisines` | Distinct cuisines extracted from dataset |

---

## 8. Data Flow Summary

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
                                              [Parse & Enrich]
                                                        │
                                                        ▼
                                        RecommendationResponse ──► UI
```

---

## 9. Cross-Cutting Concerns

### 9.1 Configuration (`config.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_DATASET_NAME` | `ManikaSaini/zomato-restaurant-recommendation` | Hugging Face dataset path |
| `BUDGET_THRESHOLDS` | `{low: 500, medium: 1500}` | Cost-for-two INR thresholds |
| `MAX_CANDIDATES_FOR_LLM` | `20` | Max restaurants sent to Groq |
| `TOP_K_RECOMMENDATIONS` | `5` | Top K results to display |
| `GROQ_API_KEY` | `$env` | Groq API key (from `.env`) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Primary Groq model |
| `GROQ_FALLBACK_MODEL` | `llama-3.1-8b-instant` | Fallback Groq model |
| `GROQ_TEMPERATURE` | `0.3` | Inference temperature |
| `DATA_CACHE_PATH` | `./data/zomato_cache.parquet` | Local dataset cache |

### 9.2 Error Handling

| Scenario | Behavior |
|----------|---------|
| Dataset download fails | Retry with backoff; load from local cache if available |
| No restaurants match filters | Relax constraints (cuisine → budget → rating); surface warning |
| Groq returns invalid JSON | Retry once with `temperature=0.1`; fallback to heuristic top-K |
| Groq 429 rate limit | Exponential backoff; then return heuristic top-K with note that AI explanation is unavailable |
| Groq timeout | Same as 429 handling |
| Unknown location input | Suggest valid locations from dataset |

### 9.3 Logging & Observability

- Log filter counts (input size → candidate size)
- Log Groq LLM latency and token usage (`response.usage`)
- Log which Groq model was used per request
- Do **not** log full prompts containing sensitive user data
- Optional: trace ID per recommendation request

### 9.4 Security

- Store `GROQ_API_KEY` in environment variables, never in source control
- Add `.env` to `.gitignore`; provide `.env.example` with placeholder values
- Validate and sanitize all user inputs before building Groq prompts
- Rate-limit API endpoints if deployed publicly

---

## 10. Deployment Topology

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

- Pre-load dataset at container startup
- Single stateless API instance is sufficient for milestone scope
- Scale horizontally later by sharing a read-only dataset snapshot

---

## 11. Testing Strategy

| Test Type | Scope | Example |
|-----------|-------|---------|
| **Unit** | `RestaurantFilter` | Location + budget + rating filters return expected subset |
| **Unit** | `DataPreprocessor` | Cuisine string parsing, numeric coercion |
| **Unit** | `ResponseParser` | Valid/invalid Groq JSON response handling |
| **Integration** | `RecommendationService` | Mock Groq client returns fixed JSON; verify enriched output |
| **Snapshot** | `PromptBuilder` | Prompt contains all candidates and preference fields |

> Use a frozen subset of the dataset (10–20 rows) in test fixtures for deterministic tests. Mock the `groq` SDK client to avoid real API calls in CI.

---

## 12. Implementation Phases

| Phase | Deliverable |
|-------|-------------|
| **Phase 1 — Data** | Load Hugging Face dataset, preprocess, cache, expose repository |
| **Phase 2 — Filter** | Implement preference validation and deterministic filtering |
| **Phase 3 — Groq LLM** | Prompt builder, Groq API adapter, response parser, enricher |
| **Phase 4 — UI** | CLI or Streamlit form + results display with Groq explanations |
| **Phase 5 — Hardening** | Error handling, Groq fallback ranking, tests, README |

---

## 13. Architecture Decisions

| Decision | Choice | Alternatives Considered |
|----------|--------|------------------------|
| **LLM Provider** | **Groq** (`llama-3.3-70b-versatile`) | OpenAI GPT-4, Anthropic Claude, local models (Ollama) |
| **Pre-filter before LLM** | Yes — hard filters in Python code | Let Groq filter entire dataset (expensive, unreliable) |
| **LLM Output Format** | Structured JSON via Groq | Free-form text (harder to parse) |
| **Data Storage** | In-memory DataFrame | Database (unnecessary for read-only milestone dataset) |
| **Ranking Split** | Heuristic shortlist + Groq final rank | Pure LLM or pure heuristic |
| **UI Approach** | Streamlit for speed | React SPA (more effort for milestone 1) |

---

## 14. Related Documents

- [`context.md`](./context.md) — Product requirements and full workflow
- [`problemStatement.txt`](./problemStatement.txt) — Original problem statement

---

*Architecture Version: 1.0 | LLM Provider: Groq (`llama-3.3-70b-versatile`) | Generated: 2026-06-19*
