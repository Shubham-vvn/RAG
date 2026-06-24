# Walkthrough: NestAI Rental Property Recommendation System

This walkthrough summarizes the completed implementation of **NestAI**, an AI-powered rental property recommendation system for the Indian housing market using Python, Pandas, Streamlit, and Groq API.

---

## 1. What Was Accomplished

1. **Environment Setup & Data Portability**: Set up folder structures (`./data` and `./src`), relocated the raw Kaggle dataset, and configured virtual environments and imports.
2. **Offline Preprocessing**: Created a cleaning pipeline (`src/preprocess.py`) to parse unstructured `"Floor"` string anomalies, handle outliers, and output clean tables (`data/house_rent_clean.csv`).
3. **Groq Offline Enrichment**: Developed a batch generator (`src/enrich.py`) with Groq (`llama-3.1-8b-instant`) to enrich properties with short descriptions, tenant reviews, and amenities. Added a checkpoint-saving cache system and a local mock fallback runner.
4. **Programmatic Filtering & Shortlisting**: Built a query builder (`src/filter.py`) implementing 100% reliable hard filters, alongside a candidate selection heuristic prioritizing properties by budget proximity and size-to-rent ratio.
5. **LLM Reasoning & Recommendation**: Engineered prompt templates (`src/recommender.py`) with Groq for reasoning over natural language soft preferences, ranking options, and explaining compromises. Included a robust local ranking fallback if API limits are exceeded.
6. **Premium UI Theme (NestAI)**: Configured the frontend web application (`app.py`) with custom CSS, Jakarta Sans typography, base64 local image cards, star ratings, and clean pill layouts.

---

## 2. Project File Structure

The project has been organized into a portable structure:

- [data/](file:///Users/shubhamthakur/Downloads/nextleap%20antigravity%20projects/04.%20House%20Rental%20Project%20/data/)
  - `House_Rent_Dataset.csv` (Raw source)
  - `house_rent_clean.csv` (Programmatically cleaned)
  - `house_rent_enriched.csv` (Offline LLM synthetic cache)
  - [images/](file:///Users/shubhamthakur/Downloads/nextleap%20antigravity%20projects/04.%20House%20Rental%20Project%20/data/images/) (`bandra.png`, `juhu.png`, `powai.png`, `generic.png`)
- [src/](file:///Users/shubhamthakur/Downloads/nextleap%20antigravity%20projects/04.%20House%20Rental%20Project%20/src/)
  - `preprocess.py` (Parser and cleaner)
  - `enrich.py` (Offline Groq enrichment batcher)
  - `filter.py` (Hard filters and candidate scoring)
  - `recommender.py` (Groq prompt integration and structured JSON ranking)
- [tests/](file:///Users/shubhamthakur/Downloads/nextleap%20antigravity%20projects/04.%20House%20Rental%20Object/tests/)
  - `test_preprocess.py` (Floor parser tests)
  - `test_filter.py` (Hard filter tests)
- [app.py](file:///Users/shubhamthakur/Downloads/nextleap%20antigravity%20projects/04.%20House%20Rental%20Project%20/app.py) (Streamlit UI entrypoint)
- [.env](file:///Users/shubhamthakur/Downloads/nextleap%20antigravity%20projects/04.%20House%20Rental%20Project%20/.env) (Credentials file)
- [requirements.txt](file:///Users/shubhamthakur/Downloads/nextleap%20antigravity%20projects/04.%20House%20Rental%20Project%20/requirements.txt) (Python dependencies)

---

## 3. Verification & Testing

### Automated Tests
Ran unit test suites to confirm floor parser and filtering logic. All 13 tests passed successfully.
```bash
python -m pytest
```
Output:
```
tests/test_filter.py .......                                             [ 53%]
tests/test_preprocess.py ......                                          [100%]
============================== 13 passed in 2.60s ==============================
```

### Manual Verification
The Streamlit application server is successfully running in headless mode on port `8503`.
- **Local URL:** [http://localhost:8503](http://localhost:8503)

The interface properly executes:
1. Programmatic hard filters (filtering by City, Budget, BHK selection, Furnishing Status, and Tenant Type).
2. On-the-fly mock enrichment generation for records not batch-enriched offline.
3. Natural language query reasoning using Groq API (or local ranking fallback).
4. Premium responsive rendering with custom Jakarta Sans typography, property card list items, base64 local image tags, star ratings, and dynamic why-it-fits highlighting.
