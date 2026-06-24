# 🗺️ Phase-Wise Implementation Plan
## AI-Powered Restaurant Recommendation System (Zomato Use Case)

> **Based on:** [`architecture.md`](./architecture.md) · [`context.md`](./context.md)
> **LLM Provider:** Groq (`llama-3.3-70b-versatile`)
> **Stack:** Python 3.11+ · Hugging Face Datasets · pandas · groq SDK · Streamlit · pytest

---

## Overview

| Phase | Name | Focus | Key Output |
|-------|------|-------|-----------|
| [Phase 1](#phase-1--data-layer) | Data Layer | Load, preprocess, cache dataset | `DatasetLoader`, `DataPreprocessor`, `RestaurantRepository` |
| [Phase 2](#phase-2--filter-layer) | Filter Layer | Validate inputs, filter candidates | `PreferenceValidator`, `RestaurantFilter`, `CandidateSelector` |
| [Phase 3](#phase-3--groq-llm-layer) | Groq LLM Layer | Prompt → Groq → Parse → Enrich | `PromptBuilder`, `LLMClient`, `ResponseParser`, `RecommendationService` |
| [Phase 4](#phase-4--ui-layer) | UI Layer | CLI + Streamlit interface | `cli.py`, `streamlit_app.py` |
| [Phase 5](#phase-5--hardening) | Hardening | Errors, fallback, tests, README | Full test suite, robust error handling |

---

## Phase 1 — Data Layer

> **Goal:** Load the Zomato dataset from Hugging Face, normalize it to a canonical schema, and expose it through an in-memory repository.

### 1.1 Environment Setup

| Task | Detail |
|------|--------|
| Initialize project structure | Create `zomato-milestone1/` folder layout as defined in `architecture.md` §5 |
| Create `requirements.txt` | `datasets`, `pandas`, `groq`, `python-dotenv`, `pydantic-settings`, `streamlit`, `pytest`, `pyarrow` |
| Create `.env.example` | Include `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_TEMPERATURE`, `DATA_CACHE_PATH` |
| Create `src/config.py` | Load all env vars using `pydantic-settings`; define `BUDGET_THRESHOLDS`, `MAX_CANDIDATES_FOR_LLM`, `TOP_K_RECOMMENDATIONS` |

**`config.py` variables to define:**
```python
HF_DATASET_NAME      = "ManikaSaini/zomato-restaurant-recommendation"
BUDGET_THRESHOLDS    = {"low": 500, "medium": 1500}   # INR cost_for_two
MAX_CANDIDATES_FOR_LLM = 20
TOP_K_RECOMMENDATIONS  = 5
GROQ_API_KEY           = os.getenv("GROQ_API_KEY")
GROQ_MODEL             = "llama-3.3-70b-versatile"
GROQ_FALLBACK_MODEL    = "llama-3.1-8b-instant"
GROQ_TEMPERATURE       = 0.3
DATA_CACHE_PATH        = "./data/zomato_cache.parquet"
```

---

### 1.2 Data Models

**File:** `src/models/restaurant.py`

Define the `Restaurant` dataclass with the canonical schema:

```python
@dataclass
class Restaurant:
    id: str
    name: str
    location: str
    cuisines: list[str]
    cost_for_two: int
    rating: float
    votes: int = 0
    rest_type: str = ""
    budget_tier: str = ""   # derived: "low" | "medium" | "high"
```

---

### 1.3 Dataset Loader

**File:** `src/data/loader.py` — `DatasetLoader`

| Step | Implementation Detail |
|------|-----------------------|
| Load from HF | `load_dataset("ManikaSaini/zomato-restaurant-recommendation", split="train")` |
| Convert to DataFrame | `dataset.to_pandas()` |
| Cache to parquet | Save to `DATA_CACHE_PATH`; on next startup, load from cache if file exists |
| Error handling | Catch network errors; fall back to cached file if available; raise `DataLoadError` if neither works |

**Caching logic:**
```python
if Path(DATA_CACHE_PATH).exists():
    df = pd.read_parquet(DATA_CACHE_PATH)
else:
    df = load_from_huggingface()
    df.to_parquet(DATA_CACHE_PATH)
```

---

### 1.4 Data Preprocessor

**File:** `src/data/preprocessor.py` — `DataPreprocessor`

| Step | Detail |
|------|--------|
| Column selection | Keep only relevant columns; rename to canonical names |
| Cuisine parsing | Split `"Italian, Chinese"` → `["italian", "chinese"]` (lowercase for matching) |
| Numeric coercion | `pd.to_numeric(df["rating"], errors="coerce")` for rating and cost |
| Null handling | Drop rows with null `name`, `location`, or `rating`; fill `votes` with 0 |
| Location normalization | `.strip().title()` on location strings; apply city alias map (e.g., `"Bengaluru"` → `"Bangalore"`) |
| Budget tier derivation | Assign `"low"`, `"medium"`, `"high"` based on `cost_for_two` vs `BUDGET_THRESHOLDS` |
| Output | Return `list[Restaurant]` |

**Column rename map (inspect dataset to confirm exact names):**
```python
COLUMN_MAP = {
    "name"                 : "name",
    "location"             : "location",
    "cuisines"             : "cuisines",
    "average_cost_for_two" : "cost_for_two",
    "aggregate_rating"     : "rating",
    "votes"                : "votes",
    "rest_type"            : "rest_type",
}
```

---

### 1.5 Restaurant Repository

**File:** `src/data/repository.py` — `RestaurantRepository`

Thin query interface over the preprocessed list:

```python
class RestaurantRepository:
    def get_all(self) -> list[Restaurant]: ...
    def get_locations(self) -> list[str]: ...      # distinct locations
    def get_cuisines(self) -> list[str]: ...       # distinct cuisines
    def find_by_location(self, location: str) -> list[Restaurant]: ...
```

---

### 1.6 Phase 1 Acceptance Criteria

- [ ] `DatasetLoader` successfully loads the Hugging Face dataset
- [ ] Local parquet cache is created on first load; subsequent runs load from cache
- [ ] `DataPreprocessor` produces a clean `list[Restaurant]` with no null `name/location/rating`
- [ ] Cuisine field is a `list[str]` (not a raw comma string)
- [ ] `budget_tier` is correctly derived for all rows
- [ ] `RestaurantRepository.get_locations()` returns a deduplicated list
- [ ] Unit tests pass: `tests/test_preprocessor.py`

---

## Phase 2 — Filter Layer

> **Goal:** Accept user preferences, validate them, and apply deterministic hard filters to produce a bounded candidate list for the LLM.

### 2.1 Data Models

**File:** `src/models/preferences.py`

```python
@dataclass
class UserPreferences:
    location: str
    budget: str           # "low" | "medium" | "high"
    cuisine: str | None
    min_rating: float
    additional: str | None
```

---

### 2.2 Preference Validator & Normalizer

**File:** `src/services/filter.py` — `PreferenceValidator`, `PreferenceNormalizer`

| Validator Rule | Implementation |
|----------------|---------------|
| `location` non-empty | Raise `ValidationError` if blank |
| `location` exists in dataset | Check against `repo.get_locations()`; suggest closest if not found |
| `budget` is valid enum | Must be one of `["low", "medium", "high"]` |
| `min_rating` in `[0.0, 5.0]` | Clamp or raise error |
| `cuisine` fuzzy match | Optional; match against `repo.get_cuisines()` with `difflib` or simple `in` check |

**Normalizer steps:**
- Lowercase `cuisine` string
- Strip and title-case `location`
- Strip leading/trailing whitespace from `additional`

---

### 2.3 Restaurant Filter

**File:** `src/services/filter.py` — `RestaurantFilter`

Apply hard filters **in sequence** (order matters for performance):

```
Step 1: Filter by location  (case-insensitive match)
Step 2: Filter by budget_tier
Step 3: Filter by min_rating  (restaurant.rating >= min_rating)
Step 4: Filter by cuisine     (if preference.cuisine is not None)
Step 5: Sort by rating DESC, then votes DESC
Step 6: Take top N (MAX_CANDIDATES_FOR_LLM = 20)
```

**Constraint relaxation** (if result count == 0):
```
Relax cuisine  → if still 0 results:
Relax budget   → if still 0 results:
Relax rating   → return results + warn user which constraints were relaxed
```

---

### 2.4 Candidate Selector

**File:** `src/services/filter.py` — `CandidateSelector`

- Enforce `MAX_CANDIDATES_FOR_LLM` cap
- Apply tie-breaking: equal rating → prefer higher votes
- Return `list[Restaurant]` ready for prompt building

---

### 2.5 Phase 2 Acceptance Criteria

- [ ] `PreferenceValidator` rejects blank location, invalid budget, out-of-range rating
- [ ] `PreferenceValidator` suggests valid locations when input doesn't match dataset
- [ ] `RestaurantFilter` returns correct subsets for all filter combinations
- [ ] Constraint relaxation triggers correctly when 0 candidates found
- [ ] Result list is capped at `MAX_CANDIDATES_FOR_LLM`
- [ ] Unit tests pass: `tests/test_filter.py` (use 20-row frozen dataset fixture)

---

## Phase 3 — Groq LLM Layer

> **Goal:** Build the LLM prompt, invoke Groq's API, parse the JSON response, and enrich it with full restaurant records.

### 3.1 Data Models

**File:** `src/models/recommendation.py`

```python
@dataclass
class Recommendation:
    rank: int
    name: str
    cuisine: str          # joined display string
    rating: float
    estimated_cost: int
    explanation: str      # Groq LLM-generated

@dataclass
class RecommendationResponse:
    summary: str | None
    recommendations: list[Recommendation]
    metadata: dict        # candidates_considered, filters_applied, model
```

---

### 3.2 Prompt Builder

**File:** `src/services/prompt_builder.py` — `PromptBuilder`

Builds two message objects (`system`, `user`) to pass to Groq:

**System prompt template:**
```
You are a restaurant recommendation assistant for Indian cities.
You must ONLY recommend restaurants from the CANDIDATES list provided.
Do not invent or hallucinate restaurants. Return ONLY valid JSON.
```

**User prompt template:**
```
User Preferences:
- Location: {location}
- Budget: {budget}
- Cuisine: {cuisine or "No preference"}
- Minimum Rating: {min_rating}
- Additional Preferences: {additional or "None"}

Candidates (JSON array):
{json.dumps(candidates, indent=2)}

Task: Rank the top {TOP_K} restaurants from the candidates above.
Return your response as valid JSON matching this exact schema:
{
  "summary": "Brief overall summary (1-2 sentences)",
  "recommendations": [
    {
      "id": "<restaurant id>",
      "rank": 1,
      "explanation": "Why this restaurant fits the user's preferences (2-3 sentences)"
    }
  ]
}
```

**Design rules:**
- Include only `id`, `name`, `location`, `cuisines`, `cost_for_two`, `rating` in the candidate JSON (keep tokens low)
- Pass `restaurant.id` so the enricher can look up full records
- Instruct the model to include exactly `TOP_K_RECOMMENDATIONS` entries

---

### 3.3 Groq LLM Client

**File:** `src/services/llm_client.py` — `LLMClient`

```python
from groq import Groq

class LLMClient:
    def __init__(self, config: Settings):
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model  = config.GROQ_MODEL
        self.temperature = config.GROQ_TEMPERATURE

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
```

**Retry logic:**
```
Attempt 1: temperature=0.3
  → If JSON parse fails:
Attempt 2: temperature=0.1
  → If still fails:
Fallback: heuristic ranking (return top-K by rating, generic explanation)
```

**Groq error handling:**

| Error | Behavior |
|-------|---------|
| `groq.RateLimitError` (429) | Exponential backoff (1s, 2s, 4s); then fallback |
| `groq.APITimeoutError` | Same as rate limit |
| `groq.APIConnectionError` | Immediate fallback |
| `json.JSONDecodeError` | Retry with lower temperature |

---

### 3.4 Response Parser

**File:** `src/services/recommendation.py` — `ResponseParser`

- Parse `response_text` as JSON
- Validate required keys: `summary`, `recommendations`
- Validate each recommendation has: `id`, `rank`, `explanation`
- Handle extra/missing fields gracefully (use `.get()`)
- Raise `ParseError` if structure is fundamentally broken → triggers retry/fallback

---

### 3.5 Recommendation Enricher

**File:** `src/services/recommendation.py` — `RecommendationEnricher`

- Takes parsed LLM output (list of `{id, rank, explanation}`)
- Looks up each `id` in the candidate list from Phase 2
- Builds full `Recommendation` objects with structured fields from the dataset + LLM explanation
- Builds `RecommendationResponse` with metadata:
  - `candidates_considered`: len(candidate list)
  - `filters_applied`: dict of active filters
  - `model`: Groq model name used

---

### 3.6 Recommendation Service (Orchestrator)

**File:** `src/services/recommendation.py` — `RecommendationService`

Ties everything together:
```
preferences
  → validate (PreferenceValidator)
  → filter   (RestaurantFilter → CandidateSelector)
  → build    (PromptBuilder)
  → invoke   (LLMClient.complete)
  → parse    (ResponseParser)
  → enrich   (RecommendationEnricher)
  → return   (RecommendationResponse)
```

---

### 3.7 Phase 3 Acceptance Criteria

- [ ] `PromptBuilder` produces a well-formed system + user prompt with all candidate data
- [ ] `LLMClient` successfully calls Groq API and receives a JSON string
- [ ] `ResponseParser` correctly parses valid Groq JSON into structured objects
- [ ] `ResponseParser` triggers retry on malformed JSON, then fallback
- [ ] `RecommendationEnricher` correctly maps `id` → full restaurant data
- [ ] `RecommendationService` returns a valid `RecommendationResponse` end-to-end
- [ ] Fallback path (heuristic top-K) works when Groq is unavailable
- [ ] Unit tests pass: `tests/test_recommendation.py` (with mocked `groq` SDK)
- [ ] Groq 429 backoff and fallback tested with mock raising `RateLimitError`

---

## Phase 4 — UI Layer

> **Goal:** Build a user-facing interface — both a CLI for quick testing and a Streamlit web app for the primary demo.

### 4.1 CLI Interface

**File:** `src/ui/cli.py`

| Feature | Detail |
|---------|--------|
| Interactive prompts | `input()` for location, budget, cuisine, min_rating, additional |
| Validation feedback | Print validation errors and re-prompt |
| Results display | Numbered list with name, cuisine, rating, cost, explanation |
| Loading indicator | Print `"🔍 Searching restaurants..."` while processing |
| Fallback notice | Print `"⚠️ AI explanation unavailable — showing top results by rating"` when Groq fails |

**Sample CLI output:**
```
🍽️  Recommendation #1: Spice Garden
   Cuisine  : North Indian
   Rating   : 4.5 ⭐
   Cost/2   : ₹800
   Location : Bangalore, Koramangala

   🤖 Why: Spice Garden is an excellent choice for your medium budget...
```

---

### 4.2 Streamlit Web App

**File:** `src/ui/streamlit_app.py`

**Layout:**

```
┌──────────────────────────────────────────────────────┐
│  🍽️  Zomato AI Recommender                          │
├──────────────────────────────────────────────────────┤
│  Sidebar: Preference Form                            │
│  ├── Location (selectbox from dataset)               │
│  ├── Budget (radio: low / medium / high)             │
│  ├── Cuisine (selectbox, optional)                   │
│  ├── Min Rating (slider 0.0 – 5.0)                  │
│  ├── Additional Preferences (text_area)              │
│  └── [🔍 Find Restaurants] button                   │
├──────────────────────────────────────────────────────┤
│  Main Area: Results                                  │
│  ├── Summary banner (Groq-generated)                 │
│  ├── Applied filters badge row                       │
│  └── Recommendation cards (1 per restaurant)         │
│       ├── Rank badge | Name | Rating ⭐              │
│       ├── Cuisine · Cost for Two                     │
│       └── 🤖 AI Explanation (expandable)             │
└──────────────────────────────────────────────────────┘
```

**Streamlit components to use:**

| UI Element | Streamlit Component |
|------------|-------------------|
| Location dropdown | `st.selectbox` (populated from `repo.get_locations()`) |
| Budget selection | `st.radio` |
| Cuisine dropdown | `st.selectbox` with "No preference" option |
| Min rating | `st.slider(0.0, 5.0, 3.5, 0.1)` |
| Additional prefs | `st.text_area` |
| Loading state | `st.spinner("Asking Groq AI...")` |
| Recommendation card | `st.container` + `st.columns` |
| No results | `st.warning` with relaxed filter suggestion |
| Error state | `st.error` with fallback notice |

**State management:**
- Load dataset once using `@st.cache_resource` on `DatasetLoader`
- Store `RecommendationResponse` in `st.session_state` to avoid re-fetching on widget interaction

---

### 4.3 Entry Point

**File:** `src/main.py`

```python
# Bootstraps config, loads dataset, wires services
# Usage:
#   python src/main.py --cli       → start CLI
#   streamlit run src/main.py      → start Streamlit (default)
```

---

### 4.4 Phase 4 Acceptance Criteria

- [ ] CLI collects all 5 preference fields and displays ranked results
- [ ] CLI shows loading message and fallback notice when appropriate
- [ ] Streamlit app loads without errors; location/cuisine dropdowns are populated from dataset
- [ ] Recommendation cards display: name, cuisine, rating, cost, AI explanation
- [ ] Summary banner appears above cards
- [ ] Applied filters shown above results
- [ ] "No results" state handled with helpful message
- [ ] Groq spinner shown during API call
- [ ] Dataset loads once and is cached across Streamlit reruns

---

## Phase 5 — Hardening

> **Goal:** Make the system production-ready with robust error handling, full test coverage, complete fallback paths, and a polished README.

### 5.1 Error Handling (Full Matrix)

| Scenario | Where Handled | Behavior |
|----------|--------------|---------|
| HF dataset unreachable | `DatasetLoader` | Load from parquet cache; raise `DataLoadError` if no cache |
| Parquet cache corrupt | `DatasetLoader` | Re-download and overwrite cache |
| Zero filter results | `RestaurantFilter` | Relax constraints in order: cuisine → budget → rating; surface which were relaxed |
| Groq returns invalid JSON | `ResponseParser` | Retry once at `temperature=0.1`; then heuristic fallback |
| Groq 429 rate limit | `LLMClient` | Exponential backoff (1s, 2s, 4s); then heuristic fallback |
| Groq API timeout | `LLMClient` | Same as 429 |
| Unknown location | `PreferenceValidator` | Return closest 3 valid locations as suggestions |
| Missing `GROQ_API_KEY` | `config.py` startup | Raise `ConfigurationError` with clear message |

### 5.2 Heuristic Fallback Ranking

When Groq is unavailable, `RecommendationService` returns a `RecommendationResponse` where:
- Restaurants are sorted by `rating DESC`, `votes DESC`
- `explanation` = `"Top-rated restaurant matching your filters. AI explanation temporarily unavailable."`
- `summary` = `None`
- `metadata.model` = `"heuristic_fallback"`

---

### 5.3 Full Test Suite

**File structure:**

```
tests/
├── conftest.py               # 20-row frozen dataset fixture; mock LLM client
├── test_preprocessor.py      # cuisine parsing, numeric coercion, null handling
├── test_filter.py            # location/budget/rating/cuisine filters; relaxation
├── test_prompt_builder.py    # prompt contains all candidates, preference fields
├── test_response_parser.py   # valid JSON, missing fields, malformed JSON
└── test_recommendation.py    # end-to-end with mocked Groq; fallback path
```

**Mock pattern for Groq:**
```python
@pytest.fixture
def mock_groq_client(mocker):
    return mocker.patch("src.services.llm_client.Groq", autospec=True)
```

**Test coverage targets:**

| Module | Target Coverage |
|--------|----------------|
| `preprocessor.py` | ≥ 90% |
| `filter.py` | ≥ 90% |
| `prompt_builder.py` | ≥ 80% |
| `response_parser.py` | ≥ 85% (including error paths) |
| `recommendation.py` | ≥ 80% |

---

### 5.4 Logging

Add structured logging throughout:

```python
import logging
logger = logging.getLogger(__name__)

# In LLMClient
logger.info("Groq request: model=%s candidates=%d", model, len(candidates))
logger.info("Groq response: latency=%.2fs tokens=%d", latency, usage.total_tokens)

# In RestaurantFilter
logger.debug("Filter result: input=%d after_location=%d after_budget=%d final=%d",
             total, after_loc, after_budget, final)
```

---

### 5.5 README

**File:** `README.md`

Sections to include:
1. Project overview (1 paragraph)
2. Prerequisites (`Python 3.11+`, `pip`, Groq API key)
3. Setup instructions (`git clone`, `pip install -r requirements.txt`, copy `.env.example` → `.env`)
4. Running the app (Streamlit and CLI commands)
5. Running tests (`pytest tests/ -v`)
6. Project structure (tree)
7. Architecture overview (link to `docs/architecture.md`)
8. Configuration reference (table of env vars)

---

### 5.6 Phase 5 Acceptance Criteria

- [ ] All error scenarios in the matrix handled without unhandled exceptions
- [ ] Heuristic fallback returns valid `RecommendationResponse` when Groq fails
- [ ] All unit tests pass (`pytest tests/ -v`)
- [ ] Coverage ≥ 80% across service modules
- [ ] `GROQ_API_KEY` missing raises a clear `ConfigurationError` at startup
- [ ] Logging emits filter counts and Groq latency/tokens per request
- [ ] README contains working setup and run instructions
- [ ] `.env` is in `.gitignore`; `.env.example` has all required keys documented

---

## Dependency Graph

```
Phase 1 (Data)
   └──► Phase 2 (Filter)
             └──► Phase 3 (Groq LLM)
                      └──► Phase 4 (UI)
                                └──► Phase 5 (Hardening)
```

Each phase builds on the previous. Phases 3 and 4 can be partially parallelized
(prompt builder and UI skeleton can be drafted while Groq integration is in progress).

---

## File Creation Checklist

| File | Phase | Status |
|------|-------|--------|
| `src/config.py` | 1 | ⬜ |
| `src/models/restaurant.py` | 1 | ⬜ |
| `src/data/loader.py` | 1 | ⬜ |
| `src/data/preprocessor.py` | 1 | ⬜ |
| `src/data/repository.py` | 1 | ⬜ |
| `src/models/preferences.py` | 2 | ⬜ |
| `src/services/filter.py` | 2 | ⬜ |
| `src/models/recommendation.py` | 3 | ⬜ |
| `src/services/prompt_builder.py` | 3 | ⬜ |
| `src/services/llm_client.py` | 3 | ⬜ |
| `src/services/recommendation.py` | 3 | ⬜ |
| `src/ui/cli.py` | 4 | ⬜ |
| `src/ui/streamlit_app.py` | 4 | ⬜ |
| `src/main.py` | 4 | ⬜ |
| `tests/conftest.py` | 5 | ⬜ |
| `tests/test_preprocessor.py` | 5 | ⬜ |
| `tests/test_filter.py` | 5 | ⬜ |
| `tests/test_prompt_builder.py` | 5 | ⬜ |
| `tests/test_response_parser.py` | 5 | ⬜ |
| `tests/test_recommendation.py` | 5 | ⬜ |
| `.env.example` | 1 | ⬜ |
| `requirements.txt` | 1 | ⬜ |
| `README.md` | 5 | ⬜ |

---

## `requirements.txt` (Recommended)

```
# Data
datasets>=2.18.0
pandas>=2.2.0
pyarrow>=15.0.0

# LLM
groq>=0.9.0

# Config
pydantic-settings>=2.2.0
python-dotenv>=1.0.0

# UI
streamlit>=1.33.0

# Testing
pytest>=8.0.0
pytest-mock>=3.12.0

# Optional (fuzzy location/cuisine matching)
thefuzz>=0.22.1
```

---

*Implementation Plan v1.0 | Based on architecture.md §12 + context.md §15 | Generated: 2026-06-19*
