# 🍽️ Zomato AI Restaurant Recommender

An AI-powered restaurant recommendation system that combines structured filter searches over real Zomato datasets with the reasoning capabilities of **Groq's LLMs** to deliver personalized, explainable restaurant suggestions.

## 🚀 Features
- **Deterministic Pre-filtering:** Filters restaurants based on hard rules (Location, Budget, Rating, Cuisine) before prompt construction, optimizing token usage and eliminating hallucinated suggestions.
- **Constraint Relaxation:** Automatically relaxes search boundaries sequentially (dropping cuisine, then budget, then rating thresholds) when query combinations yield zero matches.
- **AI-Powered Explanations:** Generates human-like explanations matching user natural language preferences using Groq's `llama-3.3-70b-versatile` model.
- **Heuristic Fallback:** Automatically degrades to a review/popularity sorting heuristic if the Groq LLM layer experiences API limits or connection errors.
- **Multimodal Interfaces:** Exposes terminal CLI, a Streamlit web app, and FastAPI REST endpoints.

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10+
- A valid **Groq API Key** (Get one at [console.groq.com](https://console.groq.com/))

### Installation
1. Clone the project repository.
2. Initialize virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

### Configuration
Create a `.env` file in the project root:
```env
# Groq API Configuration
GROQ_API_KEY=your_actual_groq_api_key_here
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

---

## 🏃 Running the Application

All modes can be launched through the unified entry point using the python executable:

### 1. Interactive CLI
Launch the terminal interface:
```bash
.venv/bin/python -m src.main --mode cli
```

### 2. Streamlit Web App
Launch the browser UI:
```bash
.venv/bin/python -m src.main --mode web
```

### 3. FastAPI REST API
Launch the REST server:
```bash
.venv/bin/python -m src.main --mode api
```
Interactive swagger docs are available at `http://localhost:8000/docs`.

---

## 🧪 Running Tests
Verify unit and integration logic:
```bash
.venv/bin/pytest tests/
```

---

## 📁 Project Structure
```
zomato-recommendation/
├── src/
│   ├── main.py                     # Entry routing coordinator
│   ├── config.py                   # Central settings loader
│   ├── models/
│   │   ├── restaurant.py           # Restaurant schema and tiers
│   │   ├── preferences.py          # UserPreferences validation
│   │   └── recommendation.py       # Recommendation output formats
│   ├── data/
│   │   ├── loader.py               # Hugging Face ingest & caching
│   │   ├── preprocessor.py         # Schema parsing & normalizer
│   │   └── repository.py           # In-memory query wrapper
│   ├── services/
│   │   ├── filter.py               # Pre-filtering & relaxation
│   │   ├── prompt_builder.py       # Groq system/user prompts
│   │   ├── llm_client.py           # Groq SDK adapter (retries)
│   │   ├── response_parser.py      # JSON schemas validations
│   │   └── recommendation.py       # Main recommendation orchestrator
│   └── api/
│       ├── routes.py               # FastAPI endpoint paths
│       ├── schemas.py              # Pydantic payloads
│       └── middleware.py           # Exception maps & request tracing
├── tests/                          # Automated pytest suite
├── data/                           # Cached dataset parquet files (gitignored)
├── .env.example                    # Template environment variables
├── requirements.txt                # Pinned production requirements
└── requirements-dev.txt            # Development & testing libraries
```
